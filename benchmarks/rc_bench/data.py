"""Shared benchmark data.

A single Lorenz-63 trajectory is generated once and handed to every adapter, so
all libraries train and forecast on byte-identical inputs. The series is
normalized to zero mean / unit std per channel — the usual preprocessing for ESN
forecasting and a fair common ground across libraries.

The Lorenz integrator now lives in the public API
(:func:`resdag.utils.data.lorenz`); this module is a thin NumPy-returning
adapter over it, so the benchmark harness and the library share a single
source of truth.
"""

from __future__ import annotations

import numpy as np
import torch

from resdag.utils.data import lorenz as _lorenz_tensor


def lorenz(
    n_timesteps: int,
    dt: float = 0.02,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    discard: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """Generate a normalized Lorenz-63 trajectory as a 2D NumPy array.

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
        standardized.
    """
    series = _lorenz_tensor(
        n_timesteps,
        dt=dt,
        sigma=sigma,
        rho=rho,
        beta=beta,
        discard=discard,
        normalize="standard",
        seed=seed,
        dtype=torch.float64,
    )
    return np.ascontiguousarray(series.squeeze(0).numpy(), dtype=np.float64)


def make_series(min_length: int, seed: int = 0) -> np.ndarray:
    """Return a Lorenz series at least ``min_length`` steps long (plus headroom
    for warmup and a forecast ground-truth window)."""
    return lorenz(min_length, seed=seed)
