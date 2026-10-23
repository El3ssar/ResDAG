"""Forecast-accuracy metrics.

The headline metric is the **valid prediction time** (VPT): how many steps a
closed-loop forecast tracks the true trajectory before the normalized error
crosses a threshold. It is the standard way to compare predictive skill on
chaotic systems and is reported both in steps and in Lyapunov times.

Lyapunov conversions/constants come from :mod:`resdag.metrics`. Benchmark VPT
uses explicit, fixed training scales, unlike the public metric's historical
truth-window scaling. Every forecast is scored independently, with the first
threshold failure ending its valid prefix.
"""

from __future__ import annotations

import numpy as np

from resdag.metrics import steps_to_lyapunov
from resdag.utils.data import DATASET_DT, dataset_lambda_max

# The generator the benchmark drives everything with (see ``rc_bench.data``).
# Kept as a name so the dt / lambda lookups below generalize to any built-in
# chaotic generator resdag ships.
DATASET = "lorenz"

# Lorenz-63 largest Lyapunov exponent and the integrator step used in data.py,
# sourced from the single library-side tables rather than re-declared here.
_LAMBDA = dataset_lambda_max(DATASET)
assert _LAMBDA is not None, f"{DATASET!r} has no canonical Lyapunov exponent"
LORENZ_LAMBDA_MAX: float = _LAMBDA
LORENZ_DT: float = DATASET_DT[DATASET]


def valid_prediction_time(
    pred: np.ndarray,
    truth: np.ndarray,
    threshold: float = 0.3,
    *,
    scale: np.ndarray,
) -> int:
    """Contiguous valid prefix for one forecast using admitted training scales.

    Arrays have shape ``(time, features)`` and must match exactly. ``scale``
    is a positive per-feature vector from training, never from test truth.
    Already training-standardized inputs use ones, including constant channels.
    """
    pred, truth, scale = (np.asarray(value, dtype=np.float64) for value in (pred, truth, scale))
    if pred.ndim != 2 or pred.shape != truth.shape or pred.shape[1] < 1:
        raise ValueError("predictions and truth must have matching (time, features) shapes")
    if scale.shape != (pred.shape[1],) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("scale must contain one positive finite training scale per feature")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be positive and finite")
    if not np.isfinite(truth).all():
        raise ValueError("truth must be finite")
    # A model may diverge after a useful prefix. Its first non-finite error
    # ends that prefix instead of turning the whole evaluation into an API error.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        errors = np.hypot.reduce((pred - truth) / scale, axis=1) / np.sqrt(pred.shape[1])
    failures = np.flatnonzero(~np.isfinite(errors) | (errors >= threshold))
    return int(failures[0]) if len(failures) else len(pred)


def lyapunov_times(steps: int, dt: float = LORENZ_DT, lam: float = LORENZ_LAMBDA_MAX) -> float:
    """Convert a step count to Lyapunov times (Λ·t) via :mod:`resdag.metrics`."""
    return steps_to_lyapunov(steps, dt=dt, lambda_max=lam)
