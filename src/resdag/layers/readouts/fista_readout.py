"""
FISTA Readout Layer (Lasso / Elastic-Net)
=========================================

This module provides :class:`FISTAReadoutLayer`, a sparse / robust readout that
solves an L1 (lasso) or elastic-net regularised least-squares problem with
**FISTA** — the Fast Iterative Shrinkage-Thresholding Algorithm of Beck &
Teboulle (2009). Unlike the ridge readouts (:class:`RidgeReadoutLayer`,
:class:`CholeskyReadoutLayer`, ...), whose L2 penalty only *shrinks* weights,
the L1 term here drives irrelevant coefficients **exactly to zero**, yielding a
sparse readout that ignores uninformative reservoir states.

The old name :class:`FISTAReadout` is retained as a deprecated alias.

The solve is batched over **all** output columns at once (one FISTA run, not a
per-output loop as in RC.jl's Float64-only implementation), runs on GPU or CPU,
and supports ``float32`` / ``float64``.

Objective
---------
The layer minimises the sklearn ``ElasticNet`` objective (see the
:class:`FISTAReadoutLayer` docstring for the exact scaling / parameterisation),

.. math::

    \\min_W \\; \\lVert X W - Y \\rVert_2^2
        + \\alpha\\,\\rho\\,\\lVert W \\rVert_1
        + \\tfrac{1}{2}\\,\\alpha\\,(1-\\rho)\\,\\lVert W \\rVert_2^2 ,

with ``rho = l1_ratio``. ``l1_ratio=1`` is pure lasso; ``l1_ratio=0`` is pure
ridge (FISTA then converges to the :class:`RidgeReadoutLayer` solution). The
intercept is fitted by centering and is **never** penalised.

See Also
--------
resdag.layers.readouts.ReadoutLayer : Base readout layer.
resdag.layers.readouts.RidgeReadoutLayer : Direct ridge (L2-only) readout.
resdag.layers.readouts.SVDReadoutLayer : SVD filter-factor readout.

References
----------
.. [1] A. Beck and M. Teboulle, "A Fast Iterative Shrinkage-Thresholding
   Algorithm for Linear Inverse Problems," SIAM J. Imaging Sci., 2(1),
   183-202, 2009.
"""

import math
import warnings
from typing import Any

import torch

from ._solve_utils import build_ridge_problem, finalize_fit
from .base import ReadoutLayer


class FISTAReadoutLayer(ReadoutLayer):
    r"""Sparse / robust readout via L1 (lasso) or elastic-net FISTA.

    Solves the elastic-net least-squares problem

    .. math::

        \min_W \; \lVert X W - Y \rVert_2^2
            + \alpha\,\rho\,\lVert W \rVert_1
            + \tfrac{1}{2}\,\alpha\,(1-\rho)\,\lVert W \rVert_2^2

    (``rho = l1_ratio``) with the accelerated proximal-gradient method FISTA
    [1]_, batched across every output column simultaneously. ``l1_ratio=1``
    gives pure lasso (maximally sparse); ``l1_ratio=0`` gives pure ridge, in
    which case FISTA converges to the same solution as
    :class:`RidgeReadoutLayer`. The intercept is recovered by centering and is
    left un-penalised.

    Objective and sklearn parameterisation
    --------------------------------------
    This layer minimises the **un-normalised sum-of-squares** objective above.
    ``sklearn.linear_model.ElasticNet`` instead minimises a
    *sample-size-normalised* objective:

    .. math::

        \frac{1}{2 N} \lVert X w - y \rVert_2^2
            + \alpha_{\mathrm{sk}}\,\rho\,\lVert w \rVert_1
            + \tfrac{1}{2}\,\alpha_{\mathrm{sk}}\,(1-\rho)\,\lVert w \rVert_2^2 ,

    and ``sklearn.linear_model.Lasso(alpha=a)`` is the ``l1_ratio=1`` special
    case, i.e. it minimises
    :math:`\tfrac{1}{2N}\lVert X w - y\rVert^2 + a\,\lVert w\rVert_1`.

    Multiplying sklearn's objective by ``2 N`` shows the two objectives
    coincide when

    .. math::

        \alpha_{\text{resdag}} = 2 N \, \alpha_{\mathrm{sk}} ,

    where ``N`` is the number of (flattened) samples. So to reproduce a
    ``Lasso(alpha=a)`` / ``ElasticNet(alpha=a, l1_ratio=rho)`` fit with this
    layer, pass ``alpha = 2 * N * a`` (and the same ``l1_ratio``). This scaling
    is what makes the FISTA coefficients match sklearn's to solver tolerance;
    see the tests for the round-trip.

    FISTA details
    -------------
    The smooth part
    :math:`f(W) = \lVert X W - Y\rVert^2 + \tfrac{1}{2}\alpha(1-\rho)\lVert W\rVert^2`
    has gradient :math:`2 X^\top X W - 2 X^\top Y + \alpha(1-\rho) W`, and the
    non-smooth L1 part has the soft-threshold proximal operator
    :math:`\operatorname{prox}(Z) = \operatorname{sign}(Z)\max(|Z| - \tau, 0)`
    with threshold :math:`\tau = \text{step}\cdot\alpha\,\rho`. The constant
    step size is :math:`1/L` with Lipschitz constant

    .. math::

        L = 2\,\lambda_{\max}(X^\top X) + \alpha\,(1-\rho) ,

    where :math:`\lambda_{\max}` is the top eigenvalue of the Gram, obtained
    from :func:`torch.linalg.eigvalsh` (small ``(F, F)`` system) or, when
    ``power_iters > 0``, a few power iterations. Momentum follows the standard
    recurrence :math:`t_{k+1} = (1 + \sqrt{1 + 4 t_k^2})/2`. Iteration stops on
    the relative change :math:`\lVert W_{k+1} - W_k\rVert / \lVert W_k\rVert <
    \text{tol}` or after ``max_iter`` steps (a non-converged run warns).

    Only the Gram :math:`X^\top X` and :math:`X^\top Y` enter the iteration, so
    the per-step cost is one ``(F, F) @ (F, T)`` matmul independent of the
    sample count ``N`` — cheap for the readout widths typical of reservoir
    computing.

    Parameters
    ----------
    in_features : int
        Size of input features (reservoir state dimension).
    out_features : int
        Size of output features (prediction dimension).
    bias : bool, default=True
        Whether to include a bias term. When ``True`` the data is centered and
        an unpenalised intercept is recovered; when ``False`` the raw normal
        equations are used with no intercept.
    name : str, optional
        Name for this readout layer. Used for identification in multi-readout
        architectures and as the key into ``targets`` by :class:`ESNTrainer`.
    trainable : bool, default=False
        If ``True``, weights are trainable via backpropagation.
    alpha : float, default=1e-3
        Overall regularisation strength (must be non-negative). See the
        parameterisation note above for the relation to sklearn's ``alpha``.
        ``alpha == 0`` reduces to ordinary least squares (via FISTA).
    l1_ratio : float, default=1.0
        Elastic-net mixing in ``[0, 1]``. ``1.0`` is pure lasso, ``0.0`` is
        pure ridge, values in between blend the two.
    max_iter : int, default=1000
        Maximum number of FISTA iterations.
    tol : float, default=1e-6
        Relative-change convergence tolerance on the coefficient matrix. A run
        that reaches ``max_iter`` without meeting it emits a ``RuntimeWarning``.
    use_float64 : bool, default=True
        If ``True`` the iteration runs in ``float64`` and the result is cast
        back to the parameter dtype.
    gram_dtype : torch.dtype, optional
        Dtype for forming the Gram matrix and right-hand side (the heavy
        ``(N, F)`` matmuls). Default ``None`` forms the Gram in the solve dtype
        (``float64`` unless ``use_float64=False``) on every device; pass
        ``torch.float32`` to opt into faster, lower-precision GPU Gram
        formation.
    power_iters : int, default=0
        If ``> 0``, estimate the top Gram eigenvalue with this many power
        iterations instead of the exact :func:`torch.linalg.eigvalsh`. The
        estimate is scaled up by ``(1 + power_iter_margin)`` to initialize the
        step size. Backtracking verifies the quadratic majorizer before each
        update; the margin alone is not a guaranteed spectral upper bound.
    power_iter_margin : float, default=0.05
        Initial margin applied to a power-iteration eigenvalue estimate.

    Attributes
    ----------
    weight : torch.nn.Parameter
        Weight matrix of shape ``(out_features, in_features)``.
    bias : torch.nn.Parameter or None
        Bias vector of shape ``(out_features,)``, or ``None`` if ``bias=False``.
    alpha : float
        Overall regularisation strength.
    l1_ratio : float
        Elastic-net mixing parameter.
    n_iter_ : int
        Number of FISTA iterations run by the last :meth:`fit` (``0`` before
        the first fit).
    converged_ : bool
        Whether the last :meth:`fit` met the ``tol`` stopping criterion.

    Examples
    --------
    >>> readout = FISTAReadoutLayer(in_features=200, out_features=3, alpha=1e-2)
    >>> states = torch.randn(8, 50, 200)  # (batch, time, features)
    >>> targets = torch.randn(8, 50, 3)
    >>> readout.fit(states, targets)
    >>> output = readout(states)
    >>> output.shape
    torch.Size([8, 50, 3])

    See Also
    --------
    RidgeReadoutLayer : Direct ridge (L2-only) readout.
    SVDReadoutLayer : SVD filter-factor readout.
    resdag.training.ESNTrainer : Trainer that uses this for fitting.

    References
    ----------
    .. [1] A. Beck and M. Teboulle, "A Fast Iterative Shrinkage-Thresholding
       Algorithm for Linear Inverse Problems," SIAM J. Imaging Sci., 2(1),
       183-202, 2009.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        name: str | None = None,
        trainable: bool = False,
        *,
        alpha: float = 1e-3,
        l1_ratio: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-6,
        use_float64: bool = True,
        gram_dtype: torch.dtype | None = None,
        power_iters: int = 0,
        power_iter_margin: float = 0.05,
    ) -> None:
        super().__init__(in_features, out_features, bias, name, trainable)
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}.")
        if not 0.0 <= l1_ratio <= 1.0:
            raise ValueError(f"l1_ratio must be in [0, 1], got {l1_ratio}.")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}.")
        if tol < 0:
            raise ValueError(f"tol must be non-negative, got {tol}.")
        if power_iters < 0:
            raise ValueError(f"power_iters must be non-negative, got {power_iters}.")
        if power_iter_margin < 0:
            raise ValueError(f"power_iter_margin must be non-negative, got {power_iter_margin}.")
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.use_float64 = use_float64
        self.gram_dtype = gram_dtype
        self.power_iters = power_iters
        self.power_iter_margin = power_iter_margin
        # Diagnostics populated by _fit_impl.
        self.n_iter_: int = 0
        self.converged_: bool = False

    def _top_eigenvalue(self, gram: torch.Tensor) -> float:
        """Largest eigenvalue of the symmetric PSD Gram matrix.

        Uses an exact symmetric eigenvalue solve by default (the Gram is only
        ``(F, F)``); with ``power_iters > 0`` it runs that many power
        iterations and inflates the estimate by ``(1 + power_iter_margin)``.
        The estimate need not be an upper bound; proximal-step backtracking
        verifies the quadratic majorizer before accepting an update.

        Parameters
        ----------
        gram : torch.Tensor
            Symmetric PSD Gram matrix ``XᵀX`` of shape ``(F, F)``.

        Returns
        -------
        float
            An estimate of ``λ_max(XᵀX)`` used to initialize backtracking.
        """
        if gram.shape[0] == 0:
            return 0.0
        if self.power_iters <= 0:
            return float(torch.linalg.eigvalsh(gram)[-1].item())

        # Power iteration on the PSD Gram converges to the top eigenvalue.
        v = torch.ones(gram.shape[0], dtype=gram.dtype, device=gram.device)
        v = v / torch.linalg.norm(v)
        eig = 0.0
        for _ in range(self.power_iters):
            w = gram @ v
            w_norm = float(torch.linalg.norm(w).item())
            if w_norm == 0.0:
                # A valid nonzero Gram can annihilate this initial vector.
                # The absolute row-sum norm is a safe spectral upper bound.
                return float(gram.abs().sum(dim=1).max().item())
            v = w / w_norm
            eig = w_norm
        # Inflate to guard against a low estimate (few iterations under-shoot).
        return eig * (1.0 + self.power_iter_margin)

    def _fit_impl(
        self,
        states: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Solve the elastic-net readout via FISTA on flattened inputs.

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

        # FISTA only needs the (centered) Gram and Xᵀy: the gradient of the
        # smooth part is 2·gram·W - 2·rhs + l2·W, so the O(N F) design matrix
        # never re-enters the loop. ``build_ridge_problem`` already folded the
        # (un-penalised) intercept centering into ``gram`` / ``rhs``, so the L1
        # penalty applied below acts only on the slopes — never the intercept.
        gram = problem.gram  # (F, F), solve dtype
        rhs = problem.rhs  # (F, T), solve dtype

        l1 = self.alpha * self.l1_ratio
        l2 = self.alpha * (1.0 - self.l1_ratio)

        lambda_max = self._top_eigenvalue(gram)
        lipschitz = 2.0 * lambda_max + l2
        if not math.isfinite(lipschitz):
            raise RuntimeError(
                "FISTA requires finite curvature; scale the features or use float64."
            )
        # Degenerate Gram (all-zero states / zero features): nothing to solve.
        if lipschitz <= 0.0:
            self.n_iter_ = 0
            self.converged_ = True
            coefs = torch.zeros_like(rhs)
            return finalize_fit(problem, coefs)

        upper_bound = lipschitz
        if self.power_iters > 0:
            upper_bound = (2.0 * float(gram.abs().sum(dim=1).max().item()) + l2) * (
                1.0 + 64 * torch.finfo(gram.dtype).eps
            )

        w = torch.zeros_like(rhs)
        z = w.clone()
        t = 1.0

        layer_name = (
            f"{type(self).__name__}('{self._name}')"
            if self._name is not None
            else type(self).__name__
        )

        converged = False
        n_iter = 0
        for k in range(self.max_iter):
            n_iter = k + 1
            # Gradient of the smooth part f at z: 2(gram·z - rhs) + l2·z.
            grad = 2.0 * (gram @ z - rhs)
            if l2 != 0.0:
                grad = grad + l2 * z
            # Proximal gradient step + soft-threshold (prox of the L1 term).
            while True:
                step = 1.0 / lipschitz
                w_new = _soft_threshold(z - step * grad, step * l1)
                if self.power_iters <= 0:
                    break
                delta_z = w_new - z
                norm_squared = delta_z.square().sum()
                curvature = 2.0 * (delta_z * (gram @ delta_z)).sum() + l2 * norm_squared
                majorizer = lipschitz * norm_squared
                finite = torch.isfinite(curvature) & torch.isfinite(majorizer)
                if bool(finite & (curvature <= majorizer * (1 + 32 * torch.finfo(gram.dtype).eps))):
                    break
                if lipschitz >= upper_bound or not math.isfinite(upper_bound):
                    raise RuntimeError(
                        "FISTA could not form a finite safe step; scale features/targets "
                        "or use a higher working precision."
                    )
                lipschitz = min(2.0 * lipschitz, upper_bound)

            # Nesterov momentum extrapolation.
            t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
            z = w_new + ((t - 1.0) / t_new) * (w_new - w)

            # Relative-change stopping criterion.
            w_norm = torch.linalg.norm(w)
            delta = torch.linalg.norm(w_new - w)
            w = w_new
            t = t_new
            if w_norm > 0:
                if float((delta / w_norm).item()) < self.tol:
                    converged = True
                    break
            elif float(delta.item()) < self.tol:
                # W is still (near) zero — the iterate has settled at the
                # origin (e.g. a penalty large enough to zero every weight).
                converged = True
                break

        self.n_iter_ = n_iter
        self.converged_ = converged
        if not converged:
            warnings.warn(
                f"{layer_name}: FISTA did not converge in max_iter={self.max_iter} "
                f"iterations (relative change still >= tol={self.tol:g}). The fitted "
                f"weights are the last iterate. Increase max_iter, loosen tol, or "
                f"raise alpha for a better-conditioned problem.",
                RuntimeWarning,
                stacklevel=3,
            )

        return finalize_fit(problem, w)

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
            f"l1_ratio={self.l1_ratio}"
            f")"
        )


class FISTAReadout(FISTAReadoutLayer):
    """Deprecated alias for :class:`FISTAReadoutLayer`.

    Retained for backward compatibility.  Instantiating this class emits a
    :class:`DeprecationWarning` and constructs a :class:`FISTAReadoutLayer`;
    the two are otherwise identical (``FISTAReadout`` is a subclass, so
    ``isinstance(FISTAReadout(...), FISTAReadoutLayer)`` holds).

    .. deprecated::
        Use :class:`FISTAReadoutLayer` instead.  ``FISTAReadout`` will be
        removed in a future release.

    See Also
    --------
    FISTAReadoutLayer : The canonical class this aliases.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "FISTAReadout is deprecated and will be removed in a future release; "
            "use FISTAReadoutLayer instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


def _soft_threshold(z: torch.Tensor, threshold: float) -> torch.Tensor:
    """Soft-thresholding operator ``sign(z) · max(|z| - threshold, 0)``.

    This is the proximal operator of the scaled L1 norm ``threshold · ‖·‖₁``
    and the shrinkage step of FISTA: entries with magnitude at or below
    ``threshold`` are set to *exactly* zero, which is what produces the sparse
    lasso solution.

    Parameters
    ----------
    z : torch.Tensor
        Input tensor.
    threshold : float
        Non-negative shrinkage threshold.

    Returns
    -------
    torch.Tensor
        The soft-thresholded tensor, same shape as ``z``.
    """
    if threshold <= 0.0:
        return z
    return torch.sign(z) * torch.clamp(z.abs() - threshold, min=0.0)
