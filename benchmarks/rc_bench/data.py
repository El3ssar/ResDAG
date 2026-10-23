"""Shared benchmark data.

A single raw Lorenz-63 trajectory is generated once. Each benchmark point learns
normalization from its supervised training targets after washout only, then hands
byte-identical transformed arrays to every adapter.

The Lorenz integrator now lives in the public API
(:func:`resdag.utils.data.lorenz`); this module is a thin NumPy-returning
adapter over it, so the benchmark harness and the library share a single
source of truth.
"""

from __future__ import annotations

import numpy as np
import torch

from resdag.utils.data import lorenz as _lorenz_tensor

from .config import Point


def lorenz(
    n_timesteps: int,
    dt: float = 0.02,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    discard: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """Generate a raw Lorenz-63 trajectory as a 2D NumPy array.

    Thin wrapper over :func:`resdag.utils.data.lorenz` that drops the batch axis
    and returns NumPy, matching the interface the benchmark adapters expect.

    Parameters
    ----------
    n_timesteps : int
        Number of timesteps to return (after the discarded transient).
    dt : float
        Integration step (RK4).
    sigma, rho, beta : float
        Lorenz parameters.
    discard : int
        Leading transient steps to drop so the trajectory sits on the attractor.
    seed : int
        Seed for the (tiny) random perturbation of the initial condition.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_timesteps, 3)``, dtype float64, per-channel
        in the system's original units.
    """
    series = _lorenz_tensor(
        n_timesteps,
        dt=dt,
        sigma=sigma,
        rho=rho,
        beta=beta,
        discard=discard,
        normalize=None,
        seed=seed,
        dtype=torch.float64,
    )
    return np.ascontiguousarray(series.squeeze(0).numpy(), dtype=np.float64)


def prepare_point(series: np.ndarray, point: Point) -> tuple[np.ndarray, dict]:
    """Transform one shared point using only observations admitted to training.

    Targets at indices ``warmup + 1`` through ``train_len`` are admitted to the
    readout solve. Using these rows matches the common HPO score-scale policy.
    Future rows cannot influence the mean or scale.
    Returned bounds are half-open, zero-based indices in the raw trajectory.
    """
    from .adapters.base import Adapter

    Adapter._validate_series(series, point.series_len)
    scale_start = point.warmup + 1
    train_stop = point.train_len + 1
    observed = series[scale_start:train_stop]
    mean = observed.mean(axis=0)
    std = observed.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    transformed = np.ascontiguousarray((series - mean) / std, dtype=np.float64)
    return transformed, {
        "version": "heldout-v1",
        "bounds": "zero-based, half-open",
        "train_inputs": [0, point.train_len],
        "train_targets": [1, train_stop],
        "readout_input_rows": [point.warmup, point.train_len],
        "forecast_history": [point.history_start, point.forecast_start],
        "forecast_truth": [point.forecast_start, point.series_len],
        "normalization": {
            "fit_rows": [scale_start, train_stop],
            "mean": mean.tolist(),
            "scale": std.tolist(),
            "ddof": 0,
        },
    }


def make_series(min_length: int, seed: int = 0) -> np.ndarray:
    """Return a Lorenz series at least ``min_length`` steps long (plus headroom
    for warmup and a forecast ground-truth window)."""
    return lorenz(min_length, seed=seed)
