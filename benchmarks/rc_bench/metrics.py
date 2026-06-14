"""Forecast-accuracy metrics.

The headline metric is the **valid prediction time** (VPT): how many steps a
closed-loop forecast tracks the true trajectory before the normalized error
crosses a threshold. It is the standard way to compare predictive skill on
chaotic systems and is reported both in steps and in Lyapunov times.
"""

from __future__ import annotations

import numpy as np

# Lorenz-63 largest Lyapunov exponent and the integrator step used in data.py.
LORENZ_LAMBDA_MAX = 0.9056
LORENZ_DT = 0.02


def stepwise_error(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-step RMS error across channels, in normalized (per-channel std) units.

    Both arrays are ``(steps, dim)``; the series is standardized so a value of
    1.0 means the forecast is one standard deviation off.
    """
    n = min(len(pred), len(truth))
    diff = pred[:n] - truth[:n]
    return np.sqrt(np.mean(diff**2, axis=1))


def valid_prediction_time(pred: np.ndarray, truth: np.ndarray, threshold: float = 0.5) -> int:
    """Number of steps before the normalized error first exceeds ``threshold``.

    Returns the full compared length if the threshold is never crossed.
    """
    err = stepwise_error(pred, truth)
    over = np.nonzero(err > threshold)[0]
    return int(over[0]) if over.size else int(len(err))


def lyapunov_times(steps: int, dt: float = LORENZ_DT, lam: float = LORENZ_LAMBDA_MAX) -> float:
    """Convert a step count to Lyapunov times (Λ·t)."""
    return steps * dt * lam
