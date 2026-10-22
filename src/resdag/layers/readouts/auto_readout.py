"""Automatic, precision-aware ridge regression without changing its objective."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Literal, cast

import torch

from ._solve_utils import LeastSquaresProblem, build_least_squares_problem, finalize_fit
from .base import ReadoutLayer
from .svd_readout import _ridge_filter

Solver = Literal["auto", "primal", "dual", "qr", "svd"]
_SOLVERS = ("auto", "primal", "dual", "qr", "svd")


@dataclass(frozen=True)
class ReadoutSolveReport:
    """Diagnostics for the last successful algebraic solve.

    Residuals describe the working-precision solution before its cast back to
    the input/parameter dtype. ``relative_optimality_residual`` measures the
    ridge first-order equations; ``relative_prediction_residual`` measures
    training prediction error, which need not be small for a regularized fit.
    Conditioning and roundoff quantities are conservative numerical estimates,
    not guarantees about forecast accuracy. This report is ordinary Python
    metadata; it adds no objects to a PyTorch ``state_dict``.
    """

    requested_solver: str
    solver: str
    alpha: float
    input_device: str
    input_dtype: torch.dtype
    solve_device: str
    solve_dtype: torch.dtype
    n_samples: int
    n_features: int
    condition_estimate: float
    roundoff_estimate: float | None
    relative_optimality_residual: float
    relative_prediction_residual: float
    fallback_reasons: tuple[str, ...]
    precision_warning: str | None


class _UnsafeSolveError(RuntimeError):
    """A method failed its numerical checks without modifying parameters."""


def _norm(value: torch.Tensor) -> float:
    """Frobenius norm with scaling so tiny targets do not become zero."""
    scale = float(value.abs().max()) if value.numel() else 0.0
    if scale == 0.0 or not math.isfinite(scale):
        return scale
    return float(torch.linalg.vector_norm(value / scale)) * scale


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0 if numerator == 0 else math.inf
    result = numerator / denominator
    return result if math.isfinite(result) else math.inf


def _triangular_solve(factor: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Cholesky substitution using operations supported natively on MPS."""
    lower = torch.linalg.solve_triangular(factor, rhs, upper=False)
    return cast(torch.Tensor, torch.linalg.solve_triangular(factor.T, lower, upper=True))


class AutoReadoutLayer(ReadoutLayer):
    r"""Fit the same L2 readout with an automatic or explicitly chosen solver.

    Every method minimizes ``||X W + b - Y||² + alpha ||W||²``. The intercept
    is unpenalized. Automatic recovery never increases ``alpha`` and never
    switches to an L1 or online-learning objective.

    Automatic fitting first tries a smaller primal (features by features) or
    dual (samples by samples) Cholesky system. It accepts that route only
    when a full inverse-norm conditioning estimate, a conservative bound for
    Gram formation roundoff and a scale-relative residual pass their checks.
    Rejected tall systems use augmented QR, then SVD if rank/conditioning
    makes triangular substitution unreliable. Wide least-squares problems
    use SVD for their minimum-norm solution. These are numerical guards, not
    accuracy guarantees; inspect :attr:`last_solve` to see the selected path.

    Parameters
    ----------
    in_features, out_features : int
        Input and output dimensions, as for :class:`ReadoutLayer`.
    bias : bool, default=True
        Fit an unpenalized intercept by centering the training data.
    name : str, optional
        Readout name used by custom graphs and training targets.
    trainable : bool, default=False
        Allow ordinary PyTorch gradients through the fitted linear layer.
        Algebraic fitting itself does not build a differentiation graph.
    alpha : float, default=1e-6
        Finite nonnegative L2 penalty. The requested value is never inflated.
    solver : {"auto", "primal", "dual", "qr", "svd"}, default="auto"
        Force an algorithm or let the layer choose. A forced algorithm is
        never replaced by another. Unsafe forced normal equations warn;
        unrepresentable products, failed factorization or unreliable forced
        QR raises an error.
    solve_dtype : torch.dtype, optional
        Working precision: ``torch.float32`` or ``torch.float64``. By default
        CPU/CUDA use float64 and MPS first tries native float32. Parameters
        retain their normal PyTorch dtype and device.
    rcond : float, optional
        Finite nonnegative relative singular-value cutoff for unregularized
        SVD and QR rank checks. Default: working epsilon times max(N, F).
        Positive-ridge SVD uses every singular value and the exact penalty.
    cpu_fallback : bool, default=True
        Allow an explicit CPU float64 bridge when MPS needs a stable
        decomposition, or another accelerator lacks a required operation.
        Each bridge warns and is recorded in :attr:`last_solve`. ``False``
        raises instead. A forced method stays the same across the bridge.

    Examples
    --------
    >>> readout = AutoReadoutLayer(100, 3, alpha=1e-6)
    >>> readout.fit(torch.randn(200, 100), torch.randn(200, 3))
    >>> readout.last_solve.solver in ("primal", "qr", "svd")
    True
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        name: str | None = None,
        trainable: bool = False,
        alpha: float = 1e-6,
        solver: Solver = "auto",
        solve_dtype: torch.dtype | None = None,
        rcond: float | None = None,
        cpu_fallback: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias, name, trainable)
        self.alpha = alpha
        self.solver = solver
        self.solve_dtype = solve_dtype
        self.rcond = rcond
        self.cpu_fallback = cpu_fallback
        self.last_solve: ReadoutSolveReport | None = None
        self._validate_options()

    def _validate_options(self) -> None:
        if not math.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError(f"alpha must be non-negative and finite, got {self.alpha}.")
        for name, value in (("rcond", self.rcond),):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be non-negative and finite, got {value}.")
        if self.solver not in _SOLVERS:
            raise ValueError(f"solver must be one of {_SOLVERS}, got {self.solver!r}.")
        if self.solve_dtype not in (None, torch.float32, torch.float64):
            raise ValueError("solve_dtype must be torch.float32, torch.float64 or None.")
        if not isinstance(self.cpu_fallback, bool):
            raise TypeError("cpu_fallback must be a bool.")

    def _cutoff(self, x: torch.Tensor) -> float:
        return self.rcond if self.rcond is not None else torch.finfo(x.dtype).eps * max(x.shape)

    def _gram_solve(
        self, x: torch.Tensor, y: torch.Tensor, *, dual: bool
    ) -> tuple[torch.Tensor, float, float]:
        x_norm, y_norm = _norm(x), _norm(y)
        if (
            x_norm > 0
            and y_norm > 0
            and math.log(x_norm) + math.log(y_norm) < math.log(torch.finfo(x.dtype).tiny)
        ):
            raise _UnsafeSolveError("normal-equations products risk subnormal underflow")
        data = x.T if dual else x
        gram = data.T @ data
        if not torch.isfinite(gram).all() or self.alpha > torch.finfo(x.dtype).max:
            raise _UnsafeSolveError("Gram or regularization is outside the working dtype range")
        gram.diagonal().add_(self.alpha)
        factor = torch.linalg.cholesky(gram)
        identity = torch.eye(gram.shape[0], dtype=x.dtype, device=x.device)
        inverse = _triangular_solve(factor, identity)
        inverse_norm = _norm(inverse)
        # An inverse residual includes all off-diagonal correlations. A ratio
        # of Cholesky diagonal entries alone cannot detect these problems.
        inverse_error = _norm(identity - gram @ inverse)
        inverse_bound = _ratio(inverse_norm, 1.0 - inverse_error)
        if inverse_error >= 1.0:
            inverse_bound = math.inf
        condition = _norm(gram) * inverse_bound
        eps = torch.finfo(x.dtype).eps
        reduction = data.shape[0]
        gamma = reduction * eps / (1.0 - reduction * eps) if reduction * eps < 1 else math.inf
        uncertainty = gamma * x_norm * x_norm + eps * self.alpha
        roundoff = uncertainty * inverse_bound + eps * gram.shape[0] * condition
        rhs = y if dual else x.T @ y
        coefs = _triangular_solve(factor, rhs)
        if dual:
            coefs = x.T @ coefs
        return coefs, condition, roundoff

    def _qr_solve(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, float]:
        n, f = x.shape
        if self.alpha > 0:
            root = math.sqrt(self.alpha)
            if root > torch.finfo(x.dtype).max:
                raise _UnsafeSolveError("sqrt(alpha) is outside the working dtype range")
            x_scale = float(x.abs().max())
            if x_scale and root * math.sqrt(torch.finfo(x.dtype).eps) > x_scale:
                raise _UnsafeSolveError(
                    "QR data block is too small relative to the regularizer; select solver='svd'"
                )
            augmented = torch.cat((x, root * torch.eye(f, dtype=x.dtype, device=x.device)))
            rhs = torch.cat((y, y.new_zeros((f, y.shape[1]))))
        elif n < f:
            raise _UnsafeSolveError("unregularized wide QR is underdetermined; select solver='svd'")
        else:
            augmented, rhs = x, y
        q, r = torch.linalg.qr(augmented, mode="reduced")
        singular = torch.linalg.svdvals(r)
        largest, smallest = float(singular.max()), float(singular.min())
        if smallest <= self._cutoff(augmented) * largest or smallest == 0:
            raise _UnsafeSolveError("QR triangular factor is numerically rank deficient")
        condition = _ratio(largest, smallest)
        if condition * torch.finfo(x.dtype).eps > math.sqrt(torch.finfo(x.dtype).eps):
            raise _UnsafeSolveError("QR triangular factor exceeds the precision conditioning guard")
        return torch.linalg.solve_triangular(r, q.T @ rhs, upper=True), condition**2

    def _svd_solve(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, float]:
        u, s, vh = torch.linalg.svd(x, full_matrices=False)
        if self.alpha > 0:
            filt = _ridge_filter(s, self.alpha)
        else:
            keep = s > self._cutoff(x) * s.max()
            safe = torch.where(keep, s, torch.ones_like(s))
            filt = torch.where(keep, 1.0 / safe, torch.zeros_like(s))
        coefs = vh.T @ (filt[:, None] * (u.T @ y))
        largest = float(s.max())
        smallest = float(s.min()) if x.shape[0] >= x.shape[1] else 0.0
        root = math.sqrt(self.alpha)
        scale = max(largest, root)
        condition = (
            _ratio(
                (largest / scale) ** 2 + (root / scale) ** 2,
                (smallest / scale) ** 2 + (root / scale) ** 2,
            )
            if scale
            else math.inf
        )
        return coefs, condition

    def _residuals(
        self, x: torch.Tensor, y: torch.Tensor, coefs: torch.Tensor
    ) -> tuple[float, float]:
        error = x @ coefs - y
        xn, yn, wn = _norm(x), _norm(y), _norm(coefs)
        # Normalize each term before multiplication. A gradient formed from
        # X.T @ error can underflow to zero even when W itself is representable.
        # Log-scaled scalar weights keep the denominator meaningful as well.
        log_x = math.log(xn) if xn > 0 else -math.inf
        log_y = math.log(yn) if yn > 0 else -math.inf
        log_w = math.log(wn) if wn > 0 else -math.inf
        logs = (
            2 * log_x + log_w,
            log_x + log_y,
            math.log(self.alpha) + log_w if self.alpha > 0 else -math.inf,
        )
        largest = max(logs)
        if largest == -math.inf:
            optimality = 0.0
        elif not math.isfinite(largest):
            optimality = math.inf
        else:
            weights = [math.exp(value - largest) for value in logs]
            total = sum(weights)
            a, b, c = (value / total for value in weights)
            normalized_x = x / xn if xn else torch.zeros_like(x)
            normalized_y = y / yn if yn else torch.zeros_like(y)
            normalized_w = coefs / wn if wn else torch.zeros_like(coefs)
            gradient = (
                a * (normalized_x.T @ (normalized_x @ normalized_w))
                - b * (normalized_x.T @ normalized_y)
                + c * normalized_w
            )
            optimality = _norm(gradient)
        return optimality, _ratio(_norm(error), yn)

    def solve(
        self, states: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Solve flattened training data and record the actual numerical path."""
        self._validate_options()
        reasons: list[str] = []
        device = states.device
        dtype = self.solve_dtype or (torch.float32 if device.type == "mps" else torch.float64)

        def bridge(reason: str) -> None:
            nonlocal device, dtype
            if not self.cpu_fallback:
                raise RuntimeError(
                    f"AutoReadoutLayer needs a CPU solve: {reason}. "
                    "Set cpu_fallback=True or choose a supported precision/solver."
                )
            reasons.append(reason)
            warnings.warn(
                f"AutoReadoutLayer: using CPU float64 fitting ({reason}); "
                f"fitted parameters remain on {states.device}.",
                RuntimeWarning,
                stacklevel=3,
            )
            device, dtype = torch.device("cpu"), torch.float64

        if device.type == "mps" and dtype == torch.float64:
            bridge("MPS does not support float64")

        problem: LeastSquaresProblem | None = None

        def prepare() -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal problem
            problem = build_least_squares_problem(
                states.to(device=device),
                targets.to(device=device),
                fit_intercept=self.bias is not None,
                use_float64=dtype == torch.float64,
                gram_dtype=dtype,
            )
            x, y = problem.X, problem.y
            if problem.fit_intercept:
                assert problem.x_mean is not None and problem.y_mean is not None
                x, y = x - problem.x_mean, y - problem.y_mean
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise RuntimeError(
                    "AutoReadoutLayer centering exceeded the working dtype range; rescale the data."
                )
            return x, y

        x, y = prepare()
        requested = self.solver
        method = requested
        condition, roundoff = math.inf, None
        if requested == "auto":
            method = "primal" if x.shape[0] >= x.shape[1] else "dual"
            if method == "dual" and self.alpha == 0:
                method = "svd"
        if method in ("primal", "dual"):
            try:
                try:
                    coefs, condition, roundoff = self._gram_solve(x, y, dual=method == "dual")
                except NotImplementedError:
                    if device.type == "cpu":
                        raise
                    bridge(f"{device.type} does not implement a required {method} operation")
                    x, y = prepare()
                    coefs, condition, roundoff = self._gram_solve(x, y, dual=method == "dual")
                residual, _ = self._residuals(x, y, coefs)
                tol = math.sqrt(torch.finfo(dtype).eps)
                safe = math.isfinite(roundoff) and roundoff <= tol and residual <= tol
                if not safe:
                    reason = f"{method} normal equations failed the precision/conditioning guard"
                    if requested != "auto":
                        warnings.warn(
                            f"AutoReadoutLayer: forced {reason}.", RuntimeWarning, stacklevel=2
                        )
                    else:
                        raise _UnsafeSolveError(reason)
            except (torch.linalg.LinAlgError, _UnsafeSolveError) as exc:
                if requested != "auto":
                    raise RuntimeError(f"Forced {method} solve failed: {exc}") from exc
                reasons.append(str(exc))
                method = "qr" if x.shape[0] >= x.shape[1] else "svd"
        if method in ("qr", "svd"):
            if device.type == "mps":
                bridge(f"MPS {method} requires a stable CPU decomposition")
                x, y = prepare()
            try:
                solve_decomposition = self._qr_solve if method == "qr" else self._svd_solve
                try:
                    coefs, condition = solve_decomposition(x, y)
                except NotImplementedError:
                    if device.type == "cpu":
                        raise
                    bridge(f"{device.type} does not implement a required {method} operation")
                    x, y = prepare()
                    coefs, condition = solve_decomposition(x, y)
            except (torch.linalg.LinAlgError, _UnsafeSolveError) as exc:
                if requested != "auto" or method == "svd":
                    raise RuntimeError(f"{method} solve failed: {exc}") from exc
                reasons.append(str(exc))
                method = "svd"
                coefs, condition = self._svd_solve(x, y)
            roundoff = None
        if not torch.isfinite(coefs).all():
            raise RuntimeError(
                "AutoReadoutLayer produced non-finite coefficients; rescale the data."
            )
        residual, prediction_residual = self._residuals(x, y, coefs)
        precision_warning = None
        if (
            self.alpha > 0
            and method in ("qr", "svd")
            and math.sqrt(condition) * torch.finfo(dtype).eps > math.sqrt(torch.finfo(dtype).eps)
        ):
            precision_warning = (
                "The regularized design is severely ill-conditioned at the working precision. "
                "Fitted coefficients may be inaccurate even with small training/optimality "
                "residuals. Rescale features, choose stronger regularization explicitly, "
                "or use higher precision; alpha was not changed."
            )
            warnings.warn(f"AutoReadoutLayer: {precision_warning}", RuntimeWarning, stacklevel=2)
        assert problem is not None
        coefs_out, intercept = finalize_fit(problem, coefs)
        self.last_solve = ReadoutSolveReport(
            requested_solver=requested,
            solver=method,
            alpha=self.alpha,
            input_device=str(states.device),
            input_dtype=states.dtype,
            solve_device=str(device),
            solve_dtype=dtype,
            n_samples=x.shape[0],
            n_features=x.shape[1],
            condition_estimate=condition,
            roundoff_estimate=roundoff,
            relative_optimality_residual=residual,
            relative_prediction_residual=prediction_residual,
            fallback_reasons=tuple(reasons),
            precision_warning=precision_warning,
        )
        return coefs_out.to(states.device), (
            intercept.to(states.device) if intercept is not None else None
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, alpha={self.alpha}, solver={self.solver!r}"
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.extra_repr()})"
