"""
Cholesky Readout Layer
======================

This module provides :class:`CholeskyReadoutLayer`, a direct ridge readout that
factorises the regularised normal equations ``(XᵀX + αI)`` with
:func:`torch.linalg.cholesky` and back-substitutes with
:func:`torch.cholesky_solve`.

It is the readout the streaming / minibatch path needs: the same closed-form
ridge solve that :class:`IncrementalRidgeReadout` performs in
:meth:`~IncrementalRidgeReadout.finalize`, packaged as a standalone
single-shot readout. It reuses the shared centering + auto ``gram_dtype``
policy (float64 on CPU, input dtype on CUDA) so the GPU-slowness footgun stays
solved.

See Also
--------
resdag.layers.readouts.ReadoutLayer : Base readout layer.
resdag.layers.readouts.CGReadoutLayer : Iterative CG ridge readout.
resdag.layers.readouts.RidgeReadoutLayer : Cholesky / LU direct readout (#70).
resdag.layers.readouts.IncrementalRidgeReadout : Streaming partial_fit readout.
"""

import torch

from ._solve_utils import build_ridge_problem, finalize_fit
from .base import ReadoutLayer


class CholeskyReadoutLayer(ReadoutLayer):
    """Readout layer that solves ridge regression via a Cholesky factorisation.

    The regularised normal equations ``(XᵀX + αI) W = Xᵀy`` are formed (with
    analytic centering when a bias is fitted) and solved with
    :func:`torch.linalg.cholesky` followed by :func:`torch.cholesky_solve`. The
    ``(F, F)`` system is symmetric positive definite for ``alpha > 0``, so the
    Cholesky factorisation is the fastest, most numerically stable exact solve
    for the readout widths typical of reservoir computing
    (``in_features = 100``–``2000``).

    Unlike :class:`CGReadoutLayer`, which only *approximates* the solution
    iteratively, this layer returns the exact ridge solution (up to floating
    point) in a single shot. It is the single-batch counterpart of the
    streaming :class:`IncrementalRidgeReadout`: both ultimately Cholesky-solve
    the same regularised Gram system, so a full-batch ``CholeskyReadoutLayer``
    fit and an accumulated ``IncrementalRidgeReadout`` fit over the same data
    agree to within floating-point tolerance.

    The solver minimises the same ridge objective as :class:`CGReadoutLayer`,

    .. math::

        \\lVert X W - Y \\rVert_2^2 + \\alpha \\lVert W \\rVert_2^2 .

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
        L2 regularization strength. Must be non-negative. ``alpha > 0``
        guarantees the Gram is positive definite; with ``alpha == 0`` the
        factorisation succeeds only on a full-rank Gram (reach for
        :class:`SVDReadoutLayer` if the Gram may be rank deficient).
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

    Examples
    --------
    >>> readout = CholeskyReadoutLayer(in_features=200, out_features=3, alpha=1e-6)
    >>> states = torch.randn(8, 50, 200)  # (batch, time, features)
    >>> targets = torch.randn(8, 50, 3)
    >>> readout.fit(states, targets)
    >>> output = readout(states)
    >>> output.shape
    torch.Size([8, 50, 3])

    See Also
    --------
    CGReadoutLayer : Iterative Conjugate Gradient ridge readout.
    RidgeReadoutLayer : Direct ridge readout with a ``solver`` switch.
    IncrementalRidgeReadout : Streaming partial_fit / finalize counterpart.
    resdag.training.ESNTrainer : Trainer that uses this for fitting.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        name: str | None = None,
        trainable: bool = False,
        alpha: float = 1e-6,
        use_float64: bool = True,
        gram_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(in_features, out_features, bias, name, trainable)
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}.")
        self.alpha = alpha
        self.use_float64 = use_float64
        self.gram_dtype = gram_dtype

    def _fit_impl(
        self,
        states: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Solve ridge regression via a Cholesky factorisation on flattened inputs.

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
        factor = torch.linalg.cholesky(a)
        coefs = torch.cholesky_solve(problem.rhs, factor)

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
            f"alpha={self.alpha}"
            f")"
        )
