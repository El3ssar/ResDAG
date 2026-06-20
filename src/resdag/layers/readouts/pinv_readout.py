"""
Pseudo-Inverse Readout Layer
============================

This module provides :class:`PinvReadoutLayer`, which solves the
least-squares readout via ``torch.linalg.lstsq`` (default) or an explicit
Moore–Penrose pseudo-inverse (``torch.linalg.pinv``), both with an ``rcond``
cutoff for rank-deficient design matrices.

See Also
--------
resdag.layers.readouts.ReadoutLayer : Base readout layer.
resdag.layers.readouts.RidgeReadoutLayer : Cholesky / solve direct readout.
resdag.layers.readouts.SVDReadoutLayer : SVD filter-factor readout.
"""

import torch

from ._solve_utils import build_ridge_problem, finalize_fit
from .base import ReadoutLayer


class PinvReadoutLayer(ReadoutLayer):
    """Readout layer solved via least squares / pseudo-inverse.

    Solves the (optionally centered) least-squares problem
    ``min_W ||X W - Y||`` directly, without forming the Gram matrix, using one
    of:

    - ``solver='lstsq'`` (default) — ``torch.linalg.lstsq`` with the ``gelsd``
      driver, a divide-and-conquer SVD-based least-squares solver that returns
      the minimum-norm solution for rank-deficient ``X`` and honours the
      ``rcond`` cutoff.
    - ``solver='pinv'`` — form the Moore–Penrose pseudo-inverse
      ``torch.linalg.pinv(X, rcond=...)`` and apply it to ``Y``. Equivalent in
      result but materialises the ``(F, N)`` pseudo-inverse explicitly.

    This is a pure least-squares readout: there is **no Tikhonov penalty**
    (``alpha``). Regularisation comes only from the ``rcond`` truncation of
    small singular values. If you need an explicit ridge penalty use
    :class:`RidgeReadoutLayer` or :class:`SVDReadoutLayer` (which exposes both
    ``alpha`` and ``rcond``).

    Solver-selection guide
    ----------------------
    - **Reach for** :class:`PinvReadoutLayer` for unregularised least squares
      on a possibly rank-deficient design matrix when you want the
      minimum-norm solution and don't need a Tikhonov ``alpha``.
    - ``solver='lstsq'`` (default) is the right choice almost always — it never
      materialises the pseudo-inverse. Use ``solver='pinv'`` only if you
      specifically want the pseudo-inverse matrix itself.
    - For an explicit ridge penalty, prefer :class:`SVDReadoutLayer`
      (filter factors, ``alpha`` + ``rcond``) or :class:`RidgeReadoutLayer`
      (Cholesky, fast, full rank).

    Parameters
    ----------
    in_features : int
        Size of input features (reservoir state dimension).
    out_features : int
        Size of output features (prediction dimension).
    bias : bool, default=True
        Whether to include a bias term. When ``True`` the data is centered and
        an unpenalised intercept is recovered; when ``False`` the raw design
        matrix is used with no intercept.
    name : str, optional
        Name for this readout layer. Used for identification in multi-readout
        architectures and as the key into ``targets`` by :class:`ESNTrainer`.
    trainable : bool, default=False
        If ``True``, weights are trainable via backpropagation.
    rcond : float, default=1e-15
        Relative cutoff for small singular values. Singular values at or below
        ``rcond * s_max`` are dropped, making rank-deficient problems well
        posed.
    solver : {'lstsq', 'pinv'}, default='lstsq'
        Whether to solve with ``torch.linalg.lstsq`` or an explicit
        ``torch.linalg.pinv``.
    use_float64 : bool, default=True
        If ``True`` the least-squares solve runs in ``float64`` and the result
        is cast back to the parameter dtype.
    gram_dtype : torch.dtype, optional
        Dtype for forming / casting the design matrix. Default ``None`` is
        automatic: ``float64`` on CPU, the input dtype on CUDA. Pass
        ``torch.float64`` to force full precision on any device.

    Attributes
    ----------
    weight : torch.nn.Parameter
        Weight matrix of shape ``(out_features, in_features)``.
    bias : torch.nn.Parameter or None
        Bias vector of shape ``(out_features,)``, or ``None`` if ``bias=False``.
    rcond : float
        Relative singular-value cutoff.
    solver : str
        The selected least-squares solver.

    Examples
    --------
    >>> readout = PinvReadoutLayer(in_features=200, out_features=3)
    >>> states = torch.randn(8, 50, 200)  # (batch, time, features)
    >>> targets = torch.randn(8, 50, 3)
    >>> readout.fit(states, targets)
    >>> output = readout(states)
    >>> output.shape
    torch.Size([8, 50, 3])

    See Also
    --------
    SVDReadoutLayer : SVD filter-factor readout with explicit ``alpha``.
    RidgeReadoutLayer : Cholesky / solve direct ridge readout.
    CGReadoutLayer : Iterative Conjugate Gradient ridge readout.
    resdag.training.ESNTrainer : Trainer that uses this for fitting.
    """

    _VALID_SOLVERS = ("lstsq", "pinv")

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        name: str | None = None,
        trainable: bool = False,
        rcond: float = 1e-15,
        solver: str = "lstsq",
        use_float64: bool = True,
        gram_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(in_features, out_features, bias, name, trainable)
        if rcond < 0:
            raise ValueError(f"rcond must be non-negative, got {rcond}.")
        if solver not in self._VALID_SOLVERS:
            raise ValueError(f"solver must be one of {self._VALID_SOLVERS}, got {solver!r}.")
        self.rcond = rcond
        self.solver = solver
        self.use_float64 = use_float64
        self.gram_dtype = gram_dtype

    def _fit_impl(
        self,
        states: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Solve the readout via least squares / pinv on flattened inputs.

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
        # No Tikhonov penalty here: the normal-equations Gram is unused, so
        # build the problem only to share the centering / precision machinery.
        problem = build_ridge_problem(
            states,
            targets,
            fit_intercept=self.bias is not None,
            use_float64=self.use_float64,
            gram_dtype=self.gram_dtype,
        )

        x = problem.X.to(problem.solve_dtype)
        y = problem.y.to(problem.solve_dtype)
        if problem.fit_intercept:
            assert problem.x_mean is not None and problem.y_mean is not None
            x = x - problem.x_mean.to(problem.solve_dtype)
            y = y - problem.y_mean.to(problem.solve_dtype)

        if self.solver == "lstsq" and x.device.type == "cpu":
            # gelsd is SVD-based and handles rank-deficient X with rcond, but
            # it is CPU-only — torch.linalg.lstsq on CUDA supports only the
            # full-rank ``gels`` driver, which silently mishandles rank
            # deficiency. On CUDA we therefore route ``lstsq`` through the
            # SVD-based pseudo-inverse, which is equally robust there.
            result = torch.linalg.lstsq(x, y, rcond=self.rcond, driver="gelsd")
            coefs = result.solution
        else:  # "pinv", or "lstsq" on a non-CPU device
            coefs = torch.linalg.pinv(x, rcond=self.rcond) @ y

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
            f"rcond={self.rcond}, "
            f"solver='{self.solver}'"
            f")"
        )
