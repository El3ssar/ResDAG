"""
Direct Ridge Readout Layer
==========================

This module provides :class:`RidgeReadoutLayer`, which solves the ridge
normal equations directly with a Cholesky factorisation (``'cholesky'``) or a
general LU solve (``'solve'``) instead of the iterative Conjugate Gradient of
:class:`CGReadoutLayer`.

See Also
--------
resdag.layers.readouts.ReadoutLayer : Base readout layer.
resdag.layers.readouts.CGReadoutLayer : Iterative CG ridge readout.
resdag.layers.readouts.SVDReadoutLayer : SVD filter-factor readout.
resdag.layers.readouts.PinvReadoutLayer : lstsq / pseudo-inverse readout.
"""

import torch

from ._solve_utils import build_ridge_problem, finalize_fit
from .base import ReadoutLayer


class RidgeReadoutLayer(ReadoutLayer):
    """Readout layer with a direct ridge-regression solver.

    For the readout widths typical of reservoir computing
    (``in_features = 100``–``2000``) the ``(F, F)`` Gram matrix factorises
    exactly and almost instantly. A direct Cholesky solve is both faster and
    *exact* (up to floating point), whereas :class:`CGReadoutLayer` only
    approximates the solution and can under-converge on the ill-conditioned
    concatenated-input readouts produced by ``ott_esn`` / ``power_augmented``.

    The solver minimises the same ridge objective as :class:`CGReadoutLayer`,

    .. math::

        \\lVert X W - Y \\rVert_2^2 + \\alpha \\lVert W \\rVert_2^2 ,

    by forming the regularised normal equations ``(XᵀX + αI) W = Xᵀy`` and
    solving them with one of:

    - ``solver='cholesky'`` (default) — ``torch.linalg.cholesky`` followed by
      ``torch.linalg.cholesky_solve``. Exploits the symmetric positive
      definiteness of ``XᵀX + αI`` (guaranteed for ``alpha > 0``) and is the
      fastest, most numerically stable option for well-conditioned fits.
    - ``solver='solve'`` — ``torch.linalg.solve`` (LU). Slightly slower but
      does not require positive definiteness, so it tolerates ``alpha == 0``
      on a full-rank Gram.

    Solver-selection guide
    ----------------------
    - **Default to** ``RidgeReadoutLayer(solver='cholesky')`` for any
      well-conditioned readout with ``alpha > 0`` and ``F <= 2000``. It is the
      fastest exact option and matches :class:`CGReadoutLayer` to ``< 1e-8``.
    - Use ``solver='solve'`` when you need ``alpha == 0`` (pure least squares)
      and the Gram is full rank but you don't want the SVD machinery.
    - If the Gram is **rank deficient** (high-degree NG-RC feature maps, more
      features than samples), a Cholesky/LU solve will fail or be unstable —
      reach for :class:`SVDReadoutLayer` (filter factors) or
      :class:`PinvReadoutLayer` (lstsq / pinv) instead.

    Parameters
    ----------
    in_features : int
        Size of input features (reservoir state dimension).
    out_features : int
        Size of output features (prediction dimension).
    bias : bool, default=True
        Whether to include a bias term. When ``True`` the data is centered and
        an unpenalised intercept is recovered; when ``False`` the raw normal
        equations are solved with no intercept.
    name : str, optional
        Name for this readout layer. Used for identification in multi-readout
        architectures and as the key into ``targets`` by :class:`ESNTrainer`.
    trainable : bool, default=False
        If ``True``, weights are trainable via backpropagation.
    alpha : float, default=1e-6
        L2 regularization strength. Must be non-negative. ``alpha == 0`` is
        only safe with ``solver='solve'`` on a full-rank Gram.
    solver : {'cholesky', 'solve'}, default='cholesky'
        Direct solver to use for the normal equations.
    use_float64 : bool, default=True
        If ``True`` the small ``(F, F)`` factorisation runs in ``float64`` and
        the result is cast back to the parameter dtype.
    gram_dtype : torch.dtype, optional
        Dtype for forming the Gram matrix and right-hand side (the heavy
        ``(N, F)`` matmuls). Default ``None`` is automatic: ``float64`` on CPU
        (cheap there), the input dtype on CUDA (``float64`` matmuls are
        crippled on consumer GPUs). Pass ``torch.float64`` to force
        full-precision Gram formation on any device.

    Attributes
    ----------
    weight : torch.nn.Parameter
        Weight matrix of shape ``(out_features, in_features)``.
    bias : torch.nn.Parameter or None
        Bias vector of shape ``(out_features,)``, or ``None`` if ``bias=False``.
    alpha : float
        L2 regularization strength.
    solver : str
        The selected direct solver.

    Examples
    --------
    >>> readout = RidgeReadoutLayer(in_features=200, out_features=3, alpha=1e-6)
    >>> states = torch.randn(8, 50, 200)  # (batch, time, features)
    >>> targets = torch.randn(8, 50, 3)
    >>> readout.fit(states, targets)
    >>> output = readout(states)
    >>> output.shape
    torch.Size([8, 50, 3])

    See Also
    --------
    CGReadoutLayer : Iterative Conjugate Gradient ridge readout.
    SVDReadoutLayer : SVD filter-factor readout for rank-deficient Gram.
    PinvReadoutLayer : lstsq / pseudo-inverse readout.
    resdag.training.ESNTrainer : Trainer that uses this for fitting.
    """

    _VALID_SOLVERS = ("cholesky", "solve")

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        name: str | None = None,
        trainable: bool = False,
        alpha: float = 1e-6,
        solver: str = "cholesky",
        use_float64: bool = True,
        gram_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(in_features, out_features, bias, name, trainable)
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}.")
        if solver not in self._VALID_SOLVERS:
            raise ValueError(f"solver must be one of {self._VALID_SOLVERS}, got {solver!r}.")
        self.alpha = alpha
        self.solver = solver
        self.use_float64 = use_float64
        self.gram_dtype = gram_dtype

    def _fit_impl(
        self,
        states: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Solve ridge regression directly on flattened inputs.

        Parameters
        ----------
        states : torch.Tensor
            Flattened state matrix of shape ``(N, in_features)``.
        targets : torch.Tensor
            Flattened target matrix of shape ``(N, out_features)``.

        Returns
        -------
        coefs : torch.Tensor
            Coefficient matrix of shape ``(in_features, out_features)``.
        intercept : torch.Tensor or None
            Intercept vector of shape ``(out_features,)``, or ``None`` when the
            layer was built with ``bias=False``.
        """
        problem = build_ridge_problem(
            states,
            targets,
            fit_intercept=self.bias is not None,
            use_float64=self.use_float64,
            gram_dtype=self.gram_dtype,
        )

        eye = torch.eye(
            problem.n_features,
            dtype=problem.solve_dtype,
            device=problem.gram.device,
        )
        a = problem.gram + self.alpha * eye

        if self.solver == "cholesky":
            factor = torch.linalg.cholesky(a)
            coefs = torch.cholesky_solve(problem.rhs, factor)
        else:  # "solve"
            coefs = torch.linalg.solve(a, problem.rhs)

        return finalize_fit(problem, coefs)

    def __repr__(self) -> str:
        """Return string representation."""
        name_str = f", name='{self._name}'" if self._name is not None else ""
        return (
            f"{self.__class__.__name__}("
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
            f"{name_str}, "
            f"alpha={self.alpha}, "
            f"solver='{self.solver}'"
            f")"
        )
