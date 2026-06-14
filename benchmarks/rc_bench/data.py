"""Shared benchmark data.

A single Lorenz-63 trajectory is generated once (NumPy, float64) and handed to
every adapter, so all libraries train and forecast on byte-identical inputs.
The series is normalized to zero mean / unit std per channel — the usual
preprocessing for ESN forecasting and a fair common ground across libraries.
"""

from __future__ import annotations

import numpy as np


def _lorenz_rhs(state: np.ndarray, sigma: float, rho: float, beta: float) -> np.ndarray:
    x, y, z = state
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])


def lorenz(
    n_timesteps: int,
    dt: float = 0.02,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    discard: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """Generate a normalized Lorenz-63 trajectory.

    Parameters
    ----------
    n_timesteps : int
        Number of timesteps to return (after the discarded transient).
    dt : float
        Integration step (RK4).
    discard : int
        Leading transient steps to drop so the trajectory sits on the attractor.
    seed : int
        Seed for the (tiny) random perturbation of the initial condition.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_timesteps, 3)``, dtype float64, per-channel
        standardized.
    """
    rng = np.random.default_rng(seed)
    state = np.array([1.0, 1.0, 1.0]) + 0.01 * rng.standard_normal(3)
    total = n_timesteps + discard
    out = np.empty((total, 3), dtype=np.float64)
    for i in range(total):
        out[i] = state
        k1 = _lorenz_rhs(state, sigma, rho, beta)
        k2 = _lorenz_rhs(state + 0.5 * dt * k1, sigma, rho, beta)
        k3 = _lorenz_rhs(state + 0.5 * dt * k2, sigma, rho, beta)
        k4 = _lorenz_rhs(state + dt * k3, sigma, rho, beta)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    series = out[discard:]
    series = (series - series.mean(axis=0)) / series.std(axis=0)
    return np.ascontiguousarray(series, dtype=np.float64)


def make_series(min_length: int, seed: int = 0) -> np.ndarray:
    """Return a Lorenz series at least ``min_length`` steps long (plus headroom
    for warmup and a forecast ground-truth window)."""
    return lorenz(min_length, seed=seed)
