"""Canonical reservoir-computing dataset generators.

One-line access to the field's standard benchmark tasks. Every generator
returns a ``(1, T, D)`` :class:`torch.Tensor` — ``(batch, time, features)``,
matching resdag's tensor convention — so the output drops straight into
:func:`resdag.utils.prepare_esn_data`, :class:`resdag.training.ESNTrainer`, or
the :class:`resdag.ESN` facade.

Shared conventions
------------------
Every generator accepts:

``n_timesteps`` : int
    Number of timesteps to return, *after* the discarded transient.
``discard`` : int
    Leading transient steps integrated then dropped, so the trajectory sits on
    the attractor (continuous systems) or past the warm-up of the recurrence
    (discrete maps / delay systems).
``normalize`` : {"standard", "minmax", None}
    Optional per-channel normalization applied to the returned window:
    ``"standard"`` (zero mean, unit std — the usual ESN preprocessing),
    ``"minmax"`` (scaled to ``[-1, 1]``), or ``None`` (raw values).
``seed`` : int, numpy.random.Generator, or None
    Threaded through :func:`resdag.utils.create_rng` for reproducibility.
``dtype`` / ``device`` : torch dtype / device
    Dtype and device of the returned tensor (integration is always done in
    NumPy float64 for numerical stability, then cast).

Continuous systems (:func:`lorenz`, :func:`rossler`) are integrated with a
fixed-step classic RK4 scheme.

Examples
--------
>>> from resdag.utils.data import lorenz
>>> series = lorenz(2000)
>>> series.shape
torch.Size([1, 2000, 3])

>>> from resdag import datasets
>>> x = datasets.mackey_glass(1500, normalize="minmax", seed=0)
>>> x.shape
torch.Size([1, 1500, 1])
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import torch

from ..general import create_rng

Normalize = Literal["standard", "minmax"]

__all__ = [
    "lorenz",
    "mackey_glass",
    "narma",
    "rossler",
    "henon",
    "sine",
]


def _normalize(series: np.ndarray, method: Normalize | None) -> np.ndarray:
    """Apply optional per-channel normalization to a ``(T, D)`` array.

    Parameters
    ----------
    series : numpy.ndarray
        Trajectory of shape ``(T, D)``.
    method : {"standard", "minmax"} or None
        ``"standard"`` rescales each channel to zero mean / unit std;
        ``"minmax"`` rescales each channel to ``[-1, 1]``; ``None`` returns the
        input unchanged. Channels with zero spread are left untouched (their
        scale is treated as ``1.0``) to avoid division by zero.

    Returns
    -------
    numpy.ndarray
        The normalized (or unchanged) array, ``float64``.
    """
    if method is None:
        return series
    if method == "standard":
        mean = series.mean(axis=0, keepdims=True)
        std = series.std(axis=0, keepdims=True)
        std = np.where(std == 0.0, 1.0, std)
        standardized: np.ndarray = (series - mean) / std
        return standardized
    if method == "minmax":
        lo = series.min(axis=0, keepdims=True)
        hi = series.max(axis=0, keepdims=True)
        rng = hi - lo
        rng = np.where(rng == 0.0, 1.0, rng)
        scaled: np.ndarray = 2.0 * (series - lo) / rng - 1.0
        return scaled
    raise ValueError(
        f"Unknown normalize method {method!r}. Expected 'standard', 'minmax', or None."
    )


def _to_tensor(
    series: np.ndarray,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> torch.Tensor:
    """Turn a ``(T, D)`` NumPy array into a ``(1, T, D)`` tensor.

    Parameters
    ----------
    series : numpy.ndarray
        Trajectory of shape ``(T, D)``.
    dtype : torch.dtype
        Dtype of the returned tensor.
    device : torch.device, str, or None
        Device of the returned tensor.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(1, T, D)`` — a leading singleton batch axis.
    """
    tensor = torch.as_tensor(np.ascontiguousarray(series), dtype=dtype, device=device)
    return tensor.unsqueeze(0)


def _integrate_rk4(
    rhs: Callable[[np.ndarray], np.ndarray],
    state: np.ndarray,
    n_steps: int,
    dt: float,
) -> np.ndarray:
    """Integrate an autonomous ODE with classic fixed-step RK4.

    Parameters
    ----------
    rhs : callable
        Right-hand side ``f(state) -> dstate/dt``, operating on a ``(D,)`` array.
    state : numpy.ndarray
        Initial condition of shape ``(D,)``.
    n_steps : int
        Number of states to record (including the initial condition).
    dt : float
        Integration step.

    Returns
    -------
    numpy.ndarray
        Trajectory of shape ``(n_steps, D)``, ``float64``.
    """
    out = np.empty((n_steps, state.shape[0]), dtype=np.float64)
    for i in range(n_steps):
        out[i] = state
        k1 = rhs(state)
        k2 = rhs(state + 0.5 * dt * k1)
        k3 = rhs(state + 0.5 * dt * k2)
        k4 = rhs(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return out


def lorenz(
    n_timesteps: int,
    *,
    dt: float = 0.02,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    x0: tuple[float, float, float] = (1.0, 1.0, 1.0),
    perturb: float = 0.01,
    discard: int = 1000,
    normalize: Normalize | None = "standard",
    seed: int | np.random.Generator | None = 0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Generate a Lorenz-63 trajectory (classic RK4 integration).

    The Lorenz-63 system is the canonical chaotic benchmark for reservoir
    forecasting::

        dx/dt = sigma * (y - x)
        dy/dt = x * (rho - z) - y
        dz/dt = x * y - beta * z

    Parameters
    ----------
    n_timesteps : int
        Number of timesteps to return (after the discarded transient).
    dt : float, optional
        Integration step (RK4). Default ``0.02``.
    sigma, rho, beta : float, optional
        Lorenz parameters. Defaults ``10.0``, ``28.0``, ``8/3`` (the classic
        chaotic regime).
    x0 : tuple of float, optional
        Base initial condition ``(x, y, z)``. Default ``(1.0, 1.0, 1.0)``.
    perturb : float, optional
        Std of the Gaussian perturbation added to ``x0`` (drawn from the seeded
        RNG). Set to ``0`` for a fully deterministic initial condition.
    discard : int, optional
        Leading transient steps to drop so the trajectory sits on the
        attractor. Default ``1000``.
    normalize : {"standard", "minmax"} or None, optional
        Per-channel normalization of the returned window. Default ``"standard"``.
    seed : int, numpy.random.Generator, or None, optional
        Seed for the initial-condition perturbation (via
        :func:`resdag.utils.create_rng`). Default ``0``.
    dtype : torch.dtype, optional
        Dtype of the returned tensor. Default ``torch.float32`` (matches
        torch's default, so the series drops straight into a model).
    device : torch.device, str, or None, optional
        Device of the returned tensor. Default ``None`` (CPU).

    Returns
    -------
    torch.Tensor
        Trajectory of shape ``(1, n_timesteps, 3)``.

    See Also
    --------
    rossler : Another canonical chaotic attractor.
    """
    rng = create_rng(seed)
    state = np.asarray(x0, dtype=np.float64)
    if perturb:
        state = state + perturb * rng.standard_normal(3)

    def rhs(s: np.ndarray) -> np.ndarray:
        x, y, z = s
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

    traj = _integrate_rk4(rhs, state, n_timesteps + discard, dt)
    series = _normalize(traj[discard:], normalize)
    return _to_tensor(series, dtype, device)


def rossler(
    n_timesteps: int,
    *,
    dt: float = 0.05,
    a: float = 0.2,
    b: float = 0.2,
    c: float = 5.7,
    x0: tuple[float, float, float] = (1.0, 1.0, 1.0),
    perturb: float = 0.01,
    discard: int = 1000,
    normalize: Normalize | None = "standard",
    seed: int | np.random.Generator | None = 0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Generate a Rössler trajectory (classic RK4 integration).

    The Rössler system is a three-dimensional chaotic attractor::

        dx/dt = -y - z
        dy/dt = x + a * y
        dz/dt = b + z * (x - c)

    Parameters
    ----------
    n_timesteps : int
        Number of timesteps to return (after the discarded transient).
    dt : float, optional
        Integration step (RK4). Default ``0.05``.
    a, b, c : float, optional
        Rössler parameters. Defaults ``0.2``, ``0.2``, ``5.7`` (the standard
        chaotic regime).
    x0 : tuple of float, optional
        Base initial condition ``(x, y, z)``. Default ``(1.0, 1.0, 1.0)``.
    perturb : float, optional
        Std of the Gaussian perturbation added to ``x0``. Set to ``0`` for a
        deterministic initial condition.
    discard : int, optional
        Leading transient steps to drop. Default ``1000``.
    normalize : {"standard", "minmax"} or None, optional
        Per-channel normalization of the returned window. Default ``"standard"``.
    seed : int, numpy.random.Generator, or None, optional
        Seed for the initial-condition perturbation. Default ``0``.
    dtype : torch.dtype, optional
        Dtype of the returned tensor. Default ``torch.float32`` (matches
        torch's default, so the series drops straight into a model).
    device : torch.device, str, or None, optional
        Device of the returned tensor. Default ``None`` (CPU).

    Returns
    -------
    torch.Tensor
        Trajectory of shape ``(1, n_timesteps, 3)``.

    See Also
    --------
    lorenz : The canonical chaotic forecasting benchmark.
    """
    rng = create_rng(seed)
    state = np.asarray(x0, dtype=np.float64)
    if perturb:
        state = state + perturb * rng.standard_normal(3)

    def rhs(s: np.ndarray) -> np.ndarray:
        x, y, z = s
        return np.array([-y - z, x + a * y, b + z * (x - c)])

    traj = _integrate_rk4(rhs, state, n_timesteps + discard, dt)
    series = _normalize(traj[discard:], normalize)
    return _to_tensor(series, dtype, device)


def henon(
    n_timesteps: int,
    *,
    a: float = 1.4,
    b: float = 0.3,
    x0: tuple[float, float] = (0.0, 0.0),
    perturb: float = 0.0,
    discard: int = 1000,
    normalize: Normalize | None = "standard",
    seed: int | np.random.Generator | None = 0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Generate a Hénon-map trajectory.

    The Hénon map is a two-dimensional discrete-time chaotic system::

        x_{n+1} = 1 - a * x_n^2 + y_n
        y_{n+1} = b * x_n

    Parameters
    ----------
    n_timesteps : int
        Number of timesteps to return (after the discarded transient).
    a, b : float, optional
        Hénon parameters. Defaults ``1.4``, ``0.3`` (the classic chaotic
        regime).
    x0 : tuple of float, optional
        Base initial condition ``(x, y)``. Default ``(0.0, 0.0)``.
    perturb : float, optional
        Std of the Gaussian perturbation added to ``x0``. Default ``0.0`` (the
        map is robustly chaotic, so a fixed start suffices).
    discard : int, optional
        Leading transient steps to drop. Default ``1000``.
    normalize : {"standard", "minmax"} or None, optional
        Per-channel normalization of the returned window. Default ``"standard"``.
    seed : int, numpy.random.Generator, or None, optional
        Seed for the initial-condition perturbation. Default ``0``.
    dtype : torch.dtype, optional
        Dtype of the returned tensor. Default ``torch.float32`` (matches
        torch's default, so the series drops straight into a model).
    device : torch.device, str, or None, optional
        Device of the returned tensor. Default ``None`` (CPU).

    Returns
    -------
    torch.Tensor
        Trajectory of shape ``(1, n_timesteps, 2)``.
    """
    rng = create_rng(seed)
    state = np.asarray(x0, dtype=np.float64)
    if perturb:
        state = state + perturb * rng.standard_normal(2)

    total = n_timesteps + discard
    out = np.empty((total, 2), dtype=np.float64)
    x, y = float(state[0]), float(state[1])
    for i in range(total):
        out[i] = (x, y)
        x, y = 1.0 - a * x * x + y, b * x

    series = _normalize(out[discard:], normalize)
    return _to_tensor(series, dtype, device)


def mackey_glass(
    n_timesteps: int,
    *,
    tau: int = 17,
    beta: float = 0.2,
    gamma: float = 0.1,
    n: float = 10.0,
    dt: float = 1.0,
    x0: float = 1.2,
    discard: int = 1000,
    normalize: Normalize | None = "standard",
    seed: int | np.random.Generator | None = 0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Generate a Mackey-Glass delay-differential-equation trajectory.

    The Mackey-Glass equation is a scalar delay system whose dynamics are
    chaotic for ``tau > ~16.8``::

        dx/dt = beta * x(t - tau) / (1 + x(t - tau)^n) - gamma * x(t)

    Integrated with the Euler method on the history buffer (the standard scheme
    for this benchmark). The delay ``tau`` and step ``dt`` give an integer lag
    of ``round(tau / dt)`` samples.

    Parameters
    ----------
    n_timesteps : int
        Number of timesteps to return (after the discarded transient).
    tau : int, optional
        Delay. Default ``17`` (the canonical chaotic value).
    beta, gamma, n : float, optional
        Equation parameters. Defaults ``0.2``, ``0.1``, ``10.0``.
    dt : float, optional
        Integration step. Default ``1.0``.
    x0 : float, optional
        Constant value of the history buffer for ``t <= 0``. Default ``1.2``.
    discard : int, optional
        Leading transient steps to drop. Default ``1000``.
    normalize : {"standard", "minmax"} or None, optional
        Normalization of the returned window. Default ``"standard"``.
    seed : int, numpy.random.Generator, or None, optional
        Seed (accepted for API symmetry; the default trajectory is
        deterministic). Default ``0``.
    dtype : torch.dtype, optional
        Dtype of the returned tensor. Default ``torch.float32`` (matches
        torch's default, so the series drops straight into a model).
    device : torch.device, str, or None, optional
        Device of the returned tensor. Default ``None`` (CPU).

    Returns
    -------
    torch.Tensor
        Trajectory of shape ``(1, n_timesteps, 1)``.
    """
    create_rng(seed)  # accepted for API symmetry / reproducibility contract
    lag = max(1, round(tau / dt))
    total = n_timesteps + discard
    history = np.full(lag + 1, float(x0), dtype=np.float64)
    out = np.empty(total, dtype=np.float64)

    x_t = float(x0)
    for i in range(total):
        out[i] = x_t
        x_lag = history[0]
        x_next = x_t + dt * (beta * x_lag / (1.0 + x_lag**n) - gamma * x_t)
        history = np.roll(history, -1)
        history[-1] = x_t
        x_t = x_next

    series = _normalize(out[discard:].reshape(-1, 1), normalize)
    return _to_tensor(series, dtype, device)


def narma(
    n_timesteps: int,
    *,
    order: int = 10,
    alpha: float = 0.3,
    beta: float = 0.05,
    gamma: float = 1.5,
    delta: float = 0.1,
    input_low: float = 0.0,
    input_high: float = 0.5,
    discard: int = 200,
    normalize: Normalize | None = None,
    seed: int | np.random.Generator | None = 0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Generate a NARMA (Nonlinear Auto-Regressive Moving Average) series.

    NARMA-``order`` is a standard nonlinear system-identification benchmark: a
    random uniform input ``u`` drives a recurrence whose output ``y`` depends on
    its own ``order`` past values::

        y_{t+1} = alpha * y_t
                  + beta * y_t * sum_{i=0}^{order-1} y_{t-i}
                  + gamma * u_t * u_{t-order+1}
                  + delta

    Two channels are returned per timestep: the input ``u`` and the output
    ``y`` (so a readout can be trained to map ``u -> y``).

    Parameters
    ----------
    n_timesteps : int
        Number of timesteps to return (after the discarded transient).
    order : int, optional
        Memory order of the recurrence. Default ``10`` (NARMA-10).
    alpha, beta, gamma, delta : float, optional
        Recurrence coefficients. Defaults ``0.3``, ``0.05``, ``1.5``, ``0.1``.
    input_low, input_high : float, optional
        Bounds of the uniform input ``u``. Defaults ``0.0``, ``0.5``.
    discard : int, optional
        Leading transient steps to drop. Default ``200``.
    normalize : {"standard", "minmax"} or None, optional
        Per-channel normalization of the returned window. Default ``None``
        (NARMA is conventionally used on its native scale).
    seed : int, numpy.random.Generator, or None, optional
        Seed for the random input (via :func:`resdag.utils.create_rng`).
        Default ``0``.
    dtype : torch.dtype, optional
        Dtype of the returned tensor. Default ``torch.float32`` (matches
        torch's default, so the series drops straight into a model).
    device : torch.device, str, or None, optional
        Device of the returned tensor. Default ``None`` (CPU).

    Returns
    -------
    torch.Tensor
        Trajectory of shape ``(1, n_timesteps, 2)`` — columns ``[u, y]``.
    """
    rng = create_rng(seed)
    total = n_timesteps + discard
    u = rng.uniform(input_low, input_high, size=total)
    y = np.zeros(total, dtype=np.float64)

    for t in range(order, total - 1):
        y[t + 1] = (
            alpha * y[t]
            + beta * y[t] * np.sum(y[t - order + 1 : t + 1])
            + gamma * u[t] * u[t - order + 1]
            + delta
        )

    series = np.stack([u, y], axis=1)
    series = _normalize(series[discard:], normalize)
    return _to_tensor(series, dtype, device)


def sine(
    n_timesteps: int,
    *,
    periods: float = 20.0,
    amplitude: float = 1.0,
    phase: float = 0.0,
    offset: float = 0.0,
    noise: float = 0.0,
    discard: int = 0,
    normalize: Normalize | None = None,
    seed: int | np.random.Generator | None = 0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Generate a (optionally noisy) sine wave.

    A simple, smooth, periodic signal — useful as a sanity-check task and for
    illustrating the API::

        y_t = offset + amplitude * sin(2*pi * periods * t / n_timesteps + phase)

    Parameters
    ----------
    n_timesteps : int
        Number of timesteps to return (after the discarded transient).
    periods : float, optional
        Number of full oscillations over the returned window. Default ``20.0``.
    amplitude : float, optional
        Peak amplitude. Default ``1.0``.
    phase : float, optional
        Phase offset in radians. Default ``0.0``.
    offset : float, optional
        Additive vertical offset. Default ``0.0``.
    noise : float, optional
        Std of additive Gaussian noise (drawn from the seeded RNG). Default
        ``0.0``.
    discard : int, optional
        Leading transient steps to drop. Default ``0`` (a pure sine has no
        transient).
    normalize : {"standard", "minmax"} or None, optional
        Normalization of the returned window. Default ``None``.
    seed : int, numpy.random.Generator, or None, optional
        Seed for the additive noise (via :func:`resdag.utils.create_rng`).
        Default ``0``.
    dtype : torch.dtype, optional
        Dtype of the returned tensor. Default ``torch.float32`` (matches
        torch's default, so the series drops straight into a model).
    device : torch.device, str, or None, optional
        Device of the returned tensor. Default ``None`` (CPU).

    Returns
    -------
    torch.Tensor
        Trajectory of shape ``(1, n_timesteps, 1)``.
    """
    rng = create_rng(seed)
    total = n_timesteps + discard
    t = np.arange(total, dtype=np.float64)
    omega = 2.0 * np.pi * periods / max(1, n_timesteps)
    y = offset + amplitude * np.sin(omega * t + phase)
    if noise:
        y = y + noise * rng.standard_normal(total)

    series = _normalize(y[discard:].reshape(-1, 1), normalize)
    return _to_tensor(series, dtype, device)
