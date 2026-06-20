"""
SVD Readout Layer
=================

This module provides :class:`SVDReadoutLayer`, which solves the ridge /
least-squares readout via the singular value decomposition of the (centered)
design matrix using Tikhonov filter factors. Unlike a Cholesky solve, it is
robust to rank-deficient Gram matrices and supports ``alpha == 0``.

See Also
--------
resdag.layers.readouts.ReadoutLayer : Base readout layer.
resdag.layers.readouts.RidgeReadoutLayer : Cholesky / solve direct readout.
resdag.layers.readouts.PinvReadoutLayer : lstsq / pseudo-inverse readout.
"""

import torch

from ._solve_utils import build_ridge_problem, finalize_fit
from .base import ReadoutLayer


class SVDReadoutLayer(ReadoutLayer):
    """Readout layer solved via SVD with Tikhonov filter factors.

    The (optionally centered) design matrix ``X`` is decomposed as
    ``X = U diag(s) Vᵀ`` and the ridge solution is reconstructed with
    *filter factors*

    .. math::

        W = V \\, \\operatorname{diag}\\!\\left(\\frac{s}{s^2 + \\alpha}\\right)
            U^\\top Y .

    For ``alpha > 0`` this is exactly the ridge solution; for ``alpha == 0`` it
    collapses to the minimum-norm least-squares solution, with singular values
    at or below an ``rcond`` cutoff dropped so a **rank-deficient** Gram (more
    features than independent samples, or the near-collinear columns produced
    by high-degree NG-RC feature maps) is handled gracefully instead of
    blowing up the way a Cholesky / normal-equations solve would.

    Working from the SVD of ``X`` directly (rather than the Gram ``XᵀX``)
    squares the effective condition number's *square root* away, so the small
    singular values that an iterative CG solver under-resolves are recovered
    faithfully — giving a strictly lower train residual on the ill-conditioned
    fits where CG stalls.

    Solver-selection guide
    ----------------------
    - **Reach for** :class:`SVDReadoutLayer` when the Gram is **rank deficient
      or severely ill-conditioned**: ``alpha == 0`` least squares, high-degree
      NG-RC feature maps, or wide concatenated-input readouts where CG
      under-converges. It is the most robust of the readouts.
    - For a **well-conditioned** readout with ``alpha > 0`` and ``F <= 2000``,
      :class:`RidgeReadoutLayer` (``solver='cholesky'``) is faster and equally
      accurate — prefer it there.
    - :class:`PinvReadoutLayer` is the closely-related lstsq / pinv route; SVD
      here gives explicit control over the Tikhonov filter and ``rcond``.

    Parameters
    ----------
    in_features : int
        Size of input features (reservoir state dimension).
    out_features : int
        Size of output features (prediction dimension).
    bias : bool, default=True
        Whether to include a bias term. When ``True`` the data is centered and
        an unpenalised intercept is recovered; when ``False`` the raw design
        matrix is decomposed with no intercept.
    name : str, optional
        Name for this readout layer. Used for identification in multi-readout
        architectures and as the key into ``targets`` by :class:`ESNTrainer`.
    trainable : bool, default=False
        If ``True``, weights are trainable via backpropagation.
    alpha : float, default=0.0
        Tikhonov (L2) regularization strength. Must be non-negative. ``0`` is
        fully supported and yields the minimum-norm least-squares solution.
    rcond : float, default=1e-15
        Relative cutoff for small singular values. Singular values at or below
        ``rcond * s_max`` are treated as zero (their filter factor is set to
        zero), which is what makes rank-deficient problems well posed. Only
        active for ``alpha == 0``; for ``alpha > 0`` the Tikhonov filter
        already damps small singular values.
    use_float64 : bool, default=True
        If ``True`` the SVD runs in ``float64`` and the result is cast back to
        the parameter dtype.
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
    alpha : float
        Tikhonov regularization strength.
    rcond : float
        Relative singular-value cutoff.

    Examples
    --------
    >>> readout = SVDReadoutLayer(in_features=200, out_features=3, alpha=0.0)
    >>> states = torch.randn(8, 50, 200)  # (batch, time, features)
    >>> targets = torch.randn(8, 50, 3)
    >>> readout.fit(states, targets)
    >>> output = readout(states)
    >>> output.shape
    torch.Size([8, 50, 3])

    See Also
    --------
    RidgeReadoutLayer : Cholesky / solve direct readout (faster, full rank).
    PinvReadoutLayer : lstsq / pseudo-inverse readout.
    CGReadoutLayer : Iterative Conjugate Gradient ridge readout.
    resdag.training.ESNTrainer : Trainer that uses this for fitting.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        name: str | None = None,
        trainable: bool = False,
        alpha: float = 0.0,
        rcond: float = 1e-15,
        use_float64: bool = True,
        gram_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(in_features, out_features, bias, name, trainable)
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}.")
        if rcond < 0:
            raise ValueError(f"rcond must be non-negative, got {rcond}.")
        self.alpha = alpha
        self.rcond = rcond
        self.use_float64 = use_float64
        self.gram_dtype = gram_dtype

    def _fit_impl(
        self,
        states: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Solve the readout via SVD filter factors on flattened inputs.

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

        # Operate on the centered design matrix when an intercept is fitted so
        # the SVD sees the same problem the Gram-based solvers do. The means
        # carried by ``problem`` reproduce the analytic centering exactly.
        x = problem.X.to(problem.solve_dtype)
        y = problem.y.to(problem.solve_dtype)
        if problem.fit_intercept:
            assert problem.x_mean is not None and problem.y_mean is not None
            x = x - problem.x_mean.to(problem.solve_dtype)
            y = y - problem.y_mean.to(problem.solve_dtype)

        # Economy SVD of the (N, F) design matrix.
        u, s, vh = torch.linalg.svd(x, full_matrices=False)

        # Tikhonov filter factors s / (s^2 + alpha). For alpha == 0 this is
        # 1 / s on the retained singular values and 0 on those at/below the
        # rcond cutoff (the minimum-norm least-squares pseudo-inverse).
        if self.alpha > 0:
            filt = s / (s * s + self.alpha)
        else:
            cutoff = self.rcond * s.max() if s.numel() > 0 else s.new_zeros(())
            inv_s = torch.where(s > cutoff, 1.0 / s, torch.zeros_like(s))
            filt = inv_s

        # W = V diag(filt) Uᵀ y  with vh = Vᵀ, so V = vh.T.
        coefs = vh.T @ (filt.unsqueeze(1) * (u.T @ y))

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
            f"rcond={self.rcond}"
            f")"
        )
