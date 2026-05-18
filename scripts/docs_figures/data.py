"""Synthetic data generators used by multiple documentation figures.

Keeping these in one place means tuning a single seed/parameter is reflected
in every figure that consumes them.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def sine_wave(n: int = 4000, periods: float = 32.0) -> torch.Tensor:
    """Pure sine of shape ``(1, n, 1)``."""
    t = torch.linspace(0, 2 * math.pi * periods, n)
    return torch.sin(t).view(1, -1, 1)


def lorenz63(n_steps: int = 30_000, dt: float = 0.02, seed: int = 42) -> torch.Tensor:
    """Lorenz-63 timeseries, normalised to unit variance, shape ``(1, n_steps, 3)``."""
    torch.manual_seed(seed)
    xyz = torch.zeros(n_steps, 3)
    xyz[0] = torch.tensor([1.0, 1.0, 1.0])
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    for t in range(1, n_steps):
        x, y, z = xyz[t - 1]
        d = torch.stack([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])
        xyz[t] = xyz[t - 1] + dt * d
    xyz = (xyz - xyz.mean(0)) / xyz.std(0)
    return xyz.unsqueeze(0)


def mackey_glass(
    n: int = 8000,
    tau: int = 17,
    beta: float = 0.2,
    gamma: float = 0.1,
    n_exp: int = 10,
    dt: float = 1.0,
    seed: int = 11,
) -> torch.Tensor:
    """Mackey–Glass series, normalised, shape ``(1, n, 1)``."""
    rng = np.random.default_rng(seed)
    history = rng.uniform(0.5, 1.5, size=max(tau, 1))
    out = np.zeros(n)
    out[: len(history)] = history
    for i in range(len(history), n):
        x_tau = out[i - tau]
        x = out[i - 1]
        dx = beta * x_tau / (1 + x_tau ** n_exp) - gamma * x
        out[i] = x + dt * dx
    out = (out - out.mean()) / out.std()
    return torch.tensor(out, dtype=torch.float32).view(1, -1, 1)
