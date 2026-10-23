"""Opt-in, CPU float64 ridge fitting with SciPy's matrix-free LSMR solver."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

if TYPE_CHECKING:
    from scipy.sparse.linalg import LinearOperator

from .base import ReadoutLayer

_STOP_REASONS = {
    0: "zero solution",
    1: "compatible-system tolerance reached",
    2: "least-squares tolerance reached",
    3: "condition limit exceeded",
    4: "compatible-system machine precision reached",
    5: "least-squares machine precision reached",
    6: "machine condition limit exceeded",
    7: "iteration limit reached",
}
_ACCEPTED_STOPS = frozenset((0, 1, 2, 4, 5))


@dataclass(frozen=True)
class LSMRColumnReport:
    """One output's solve against a unit-scaled target, before parameter casting.

    Residual/operator/solution norms are SciPy's diagnostics for that normalized
    system. Multiply the solution by ``target_scale`` to recover original units.
    Condition estimates and stopping tests do not guarantee forecast accuracy.
    """

    output: int
    stop_code: int
    stop_reason: str
    iterations: int
    residual_norm: float
    normal_residual_norm: float
    operator_norm: float
    condition_estimate: float
    solution_norm: float
    target_scale: float


@dataclass(frozen=True)
class LSMRSolveReport:
    """Diagnostics from the last successful LSMR solve; not checkpoint state."""

    alpha: float
    input_device: str
    input_dtype: torch.dtype
    solve_device: str
    solve_dtype: torch.dtype
    n_samples: int
    n_features: int
    chunk_size: int
    cpu_bridge: bool
    columns: tuple[LSMRColumnReport, ...]


def _nonnegative(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and non-negative")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _centered_operator(
    design: np.ndarray, mean: np.ndarray | None, chunk_size: int
) -> LinearOperator:
    """Apply centered rows in bounded scratch storage, never form a Gram matrix."""
    from scipy.sparse.linalg import LinearOperator

    n, f = design.shape
    scratch = np.empty((min(n, chunk_size), f), dtype=np.float64) if mean is not None else None

    def block(start: int, stop: int) -> np.ndarray:
        values = design[start:stop]
        if mean is None:
            return values
        assert scratch is not None
        return cast(np.ndarray, np.subtract(values, mean, out=scratch[: stop - start]))

    def matvec(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector).reshape(f)
        result = np.empty(n, dtype=np.float64)
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            result[start:stop] = block(start, stop) @ vector
        return result

    def rmatvec(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector).reshape(n)
        result = np.zeros(f, dtype=np.float64)
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            result += block(start, stop).T @ vector[start:stop]
        return result

    return LinearOperator((n, f), matvec=matvec, rmatvec=rmatvec, dtype=np.float64)


class LSMRReadoutLayer(ReadoutLayer):
    r"""Fit ``||X W + b - Y||² + alpha ||W||²`` without a Gram matrix.

    This explicit solver uses SciPy ``LinearOperator`` and ``lsmr`` on CPU in
    float64, one output at a time. ``damp=sqrt(alpha)`` preserves the requested
    ridge penalty. With a bias, row blocks are centered on demand and the
    unpenalized intercept is recovered afterwards. Targets are scaled per output
    to keep tiny target magnitudes meaningful to relative stopping tests.

    The supplied feature matrix remains resident. Working storage additionally
    includes its CPU float64 representation, at most ``chunk_size * in_features``
    centered scratch entries and linear-size solver vectors. This avoids a
    feature-by-feature Gram matrix, QR augmentation and a complete retained
    centered design; it does not make reservoir feature extraction streaming.

    Accelerator inputs require an explicit CPU bridge: the default
    ``cpu_fallback=True`` allows it with a warning, while ``False`` rejects the
    transfer. Fitted parameters remain on the layer's device and dtype; ordinary
    forward calls are Torch operations and retain upstream gradients.

    Parameters
    ----------
    in_features, out_features : int
        Feature and prediction widths.
    bias : bool, default=True
        Fit an unpenalized intercept by centering the supervised rows.
    name : str, optional
        Readout name for custom DAG training targets.
    trainable : bool, default=False
        Allow gradient updates of fitted readout parameters after fitting.
    alpha : float, default=1e-6
        Finite non-negative L2 penalty; never increased automatically.
    atol, btol : float, default=1e-8
        Finite non-negative SciPy backward-error stopping tolerances. These are
        relative tests, not the absolute residual cutoff used by CGReadoutLayer.
    conlim : float, default=1e8
        Finite non-negative estimated condition limit. Zero disables this limit.
    max_iter : int or None, default=1000
        Positive iteration budget per output. None uses SciPy's min(N, F).
        Exhaustion or a condition-limit stop raises without publishing weights.
    chunk_size : int, default=1024
        Maximum rows centered in scratch storage during each operator product.
    cpu_fallback : bool, default=True
        Permit the warned CPU float64 bridge for accelerator feature tensors.

    Attributes
    ----------
    last_solve : LSMRSolveReport or None
        Diagnostics from the last successful solve. Reports are runtime metadata
        and are not preserved in portable checkpoints.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        name: str | None = None,
        trainable: bool = False,
        alpha: float = 1e-6,
        atol: float = 1e-8,
        btol: float = 1e-8,
        conlim: float = 1e8,
        max_iter: int | None = 1000,
        chunk_size: int = 1024,
        cpu_fallback: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias, name, trainable)
        self.alpha, self.atol, self.btol = alpha, atol, btol
        self.conlim, self.max_iter = conlim, max_iter
        self.chunk_size, self.cpu_fallback = chunk_size, cpu_fallback
        self.last_solve: LSMRSolveReport | None = None
        self._validate_options()

    def _validate_options(self) -> None:
        for name in ("alpha", "atol", "btol", "conlim"):
            _nonnegative(getattr(self, name), name)
        _positive_integer(self.chunk_size, "chunk_size")
        if self.max_iter is not None:
            _positive_integer(self.max_iter, "max_iter")
        if not isinstance(self.cpu_fallback, bool):
            raise ValueError("cpu_fallback must be a bool")

    def solve(
        self, states: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Solve validated flattened features without publishing partial output fits."""
        from scipy.sparse.linalg import lsmr

        self._validate_options()
        bridge = states.device.type != "cpu"
        if bridge:
            if not self.cpu_fallback:
                raise RuntimeError(
                    "LSMRReadoutLayer requires CPU float64 fitting; cpu_fallback=False"
                )
            warnings.warn(
                "LSMRReadoutLayer: copying features/targets to CPU float64 for SciPy fitting; "
                "fitted parameters remain on their original device.",
                RuntimeWarning,
                stacklevel=2,
            )
        # Transfer before casting: MPS cannot create even a transient float64
        # tensor while resolving a combined device/dtype conversion.
        x = states.detach().cpu().double().numpy()
        y = targets.detach().cpu().double().numpy()
        mean_x = x.mean(axis=0) if self.bias is not None else None
        mean_y = y.mean(axis=0) if self.bias is not None else np.zeros(y.shape[1])
        if (mean_x is not None and not np.isfinite(mean_x).all()) or not np.isfinite(mean_y).all():
            raise RuntimeError(
                "LSMRReadoutLayer centering exceeded float64 range; rescale the data"
            )
        operator = _centered_operator(x, mean_x, self.chunk_size)
        coefficients = np.empty((x.shape[1], y.shape[1]), dtype=np.float64)
        reports = []
        for output in range(y.shape[1]):
            rhs = y[:, output] - mean_y[output]
            scale = float(np.max(np.abs(rhs)))
            if not math.isfinite(scale):
                raise RuntimeError(
                    "LSMRReadoutLayer targets exceeded float64 range; rescale the data"
                )
            normalized = rhs / scale if scale else rhs
            result = lsmr(
                operator,
                normalized,
                damp=math.sqrt(self.alpha),
                atol=self.atol,
                btol=self.btol,
                conlim=self.conlim,
                maxiter=self.max_iter,
            )
            weights, stop, iterations, normr, normar, norma, conda, normx = result
            reason = _STOP_REASONS.get(stop, "unknown stopping reason")
            if (
                stop not in _ACCEPTED_STOPS
                or not all(
                    math.isfinite(value) and value >= 0
                    for value in (normr, normar, norma, conda, normx)
                )
                or not np.isfinite(weights).all()
            ):
                raise RuntimeError(
                    f"LSMRReadoutLayer output {output}: {reason} (stop={stop}, "
                    f"iterations={iterations}, condition={conda:g}); no weights were published. "
                    "Adjust tolerances, max_iter/conlim, scaling, or choose a direct solver."
                )
            if stop == 0 and np.any(operator.rmatvec(normalized) != 0):
                raise RuntimeError(
                    "LSMRReadoutLayer reported a zero solution for a nonzero normal residual; "
                    "rescale features to avoid float64 norm underflow."
                )
            coefficients[:, output] = weights * scale
            reports.append(
                LSMRColumnReport(
                    output,
                    int(stop),
                    reason,
                    int(iterations),
                    float(normr),
                    float(normar),
                    float(norma),
                    float(conda),
                    float(normx),
                    scale,
                )
            )
        intercept = mean_y - mean_x @ coefficients if mean_x is not None else None
        if not np.isfinite(coefficients).all() or (
            intercept is not None and not np.isfinite(intercept).all()
        ):
            raise RuntimeError(
                "LSMRReadoutLayer produced non-finite coefficients; rescale the data"
            )
        coefs = torch.from_numpy(coefficients).to(dtype=states.dtype).to(device=states.device)
        bias = (
            torch.from_numpy(intercept).to(dtype=states.dtype).to(device=states.device)
            if intercept is not None
            else None
        )
        if not torch.isfinite(coefs).all() or (bias is not None and not torch.isfinite(bias).all()):
            raise RuntimeError("LSMRReadoutLayer coefficients exceed the input dtype range")
        self.last_solve = LSMRSolveReport(
            float(self.alpha),
            str(states.device),
            states.dtype,
            "cpu",
            torch.float64,
            x.shape[0],
            x.shape[1],
            self.chunk_size,
            bridge,
            tuple(reports),
        )
        return coefs, bias

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, alpha={self.alpha}, atol={self.atol}, "
            f"btol={self.btol}, max_iter={self.max_iter}, conlim={self.conlim}"
        )
