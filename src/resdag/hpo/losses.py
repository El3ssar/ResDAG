"""Loss functions for hyperparameter optimization.

This module provides specialized loss functions for evaluating multi-step
forecasts in reservoir computing applications. All functions operate on
batched predictions with shape (B, T, D) where B is batch size, T is time
steps, and D is dimensions.

Available Losses
----------------
- ``"efh"`` : Expected Forecast Horizon (default, recommended for chaotic systems)
- ``"horizon"`` : Forecast Horizon Loss (contiguous valid steps)
- ``"lyap"`` : Lyapunov-weighted Loss (exponential decay for chaotic systems)
- ``"standard"`` : Standard Loss (mean geometric mean error)
- ``"discounted"`` : Discounted RMSE (half-life weighted)

Example
-------
>>> from resdag.hpo import LOSSES, get_loss
>>> loss_fn = get_loss("efh")
>>> loss = loss_fn(y_true, y_pred, threshold=0.2)
"""

from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit
from scipy.stats import gmean

__all__ = [
    "LossProtocol",
    "LOSSES",
    "get_loss",
    "expected_forecast_horizon",
    "forecast_horizon",
    "lyapunov_weighted",
    "standard_loss",
    "discounted_rmse",
]

MetricType = Literal["rmse", "mse", "mae", "nrmse"]


@runtime_checkable
class LossProtocol(Protocol):
    """Protocol for HPO loss functions.

    All loss functions must accept y_true and y_pred arrays of shape (B, T, D)
    and return a single float value to minimize.
    """

    def __call__(
        self,
        y_true: NDArray[np.floating],
        y_pred: NDArray[np.floating],
        /,
        **kwargs,
    ) -> float: ...


def _compute_errors(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    metric: MetricType = "rmse",
) -> NDArray[np.floating]:
    """Compute per-timestep errors with shape (B, T).

    Parameters
    ----------
    y_true : ndarray
        True values of shape (B, T, D).
    y_pred : ndarray
        Predicted values of shape (B, T, D).
    metric : {"rmse", "mse", "mae", "nrmse"}
        Error metric to compute.

    Returns
    -------
    ndarray
        Per-timestep errors of shape (B, T).
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")

    diff = y_pred - y_true

    if metric == "mse":
        return np.mean(diff**2, axis=2)
    if metric == "rmse":
        return np.sqrt(np.mean(diff**2, axis=2))
    if metric == "mae":
        return np.mean(np.abs(diff), axis=2)
    if metric == "nrmse":
        scale = np.std(y_true, axis=(0, 1), keepdims=True)
        scale = np.where(scale == 0, 1.0, scale)
        diff_n = diff / scale
        return np.sqrt(np.mean(diff_n**2, axis=2))

    raise ValueError(f"Unknown metric: '{metric}'. Use 'rmse', 'mse', 'mae', or 'nrmse'.")


def expected_forecast_horizon(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    /,
    metric: MetricType = "nrmse",
    threshold: float = 0.2,
    softness: float = 0.02,
) -> float:
    """Differentiable proxy for forecast horizon length.

    This is the recommended loss for chaotic systems. It provides a smooth,
    differentiable approximation of the forecast horizon by using a soft
    threshold. The loss rewards models that keep errors below the threshold
    for as many consecutive steps as possible.

    Parameters
    ----------
    y_true : ndarray
        True values of shape (B, T, D).
    y_pred : ndarray
        Predicted values of shape (B, T, D).
    metric : {"rmse", "mse", "mae", "nrmse"}, default="nrmse"
        Error metric to compute.
    threshold : float, default=0.2
        Error threshold below which predictions are considered "good".
    softness : float, default=0.02
        Controls the width of the soft threshold boundary. Smaller values
        create a harder threshold. Good default is ~10% of threshold.

    Returns
    -------
    float
        Negative expected forecast horizon. Lower (more negative) is better.
    """
    errors = _compute_errors(y_true, y_pred, metric)  # (B, T)
    e_t = np.median(errors, axis=0)  # Robust across batch

    # Soft indicator of "good prediction" at each step
    good_t = expit((threshold - e_t) / softness)  # ∈ (0, 1)

    # Survival probability: product of all good indicators up to t
    log_g = np.log(np.clip(good_t, 1e-12, 1.0))
    surv_t = np.exp(np.cumsum(log_g))

    # Expected horizon length
    expected_horizon = np.sum(surv_t)

    return -float(expected_horizon)  # Minimize → maximize horizon


def forecast_horizon(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    /,
    metric: MetricType = "rmse",
    threshold: float = 0.2,
) -> float:
    """Negative log of the contiguous valid forecast horizon.

    Counts the number of consecutive time steps where the error stays below
    the threshold, starting from the beginning of the forecast.

    Parameters
    ----------
    y_true : ndarray
        True values of shape (B, T, D).
    y_pred : ndarray
        Predicted values of shape (B, T, D).
    metric : {"rmse", "mse", "mae", "nrmse"}, default="rmse"
        Error metric to compute.
    threshold : float, default=0.2
        Error threshold below which predictions are considered valid.

    Returns
    -------
    float
        Negative log of the valid horizon length. Lower is better.
    """
    errors = _compute_errors(y_true, y_pred, metric)  # (B, T)
    e_t = np.median(errors, axis=0)  # Robust across batch

    below = e_t < threshold
    if not below[0]:
        valid_len = 0
    else:
        valid_len = int(np.argmax(~below)) if (~below).any() else int(below.size)

    return -float(np.log(valid_len + 1e-9))


def lyapunov_weighted(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    /,
    metric: MetricType = "rmse",
    dt: float = 1.0,
    lle: float = 1.0,
) -> float:
    """Lyapunov-weighted multi-step geometric mean error.

    Applies exponential weighting based on the Lyapunov exponent, emphasizing
    short-term accuracy while accounting for exponential error growth in
    chaotic systems. Errors are weighted by exp(-lle * dt * t).

    Parameters
    ----------
    y_true : ndarray
        True values of shape (B, T, D).
    y_pred : ndarray
        Predicted values of shape (B, T, D).
    metric : {"rmse", "mse", "mae", "nrmse"}, default="rmse"
        Error metric to compute.
    dt : float, default=1.0
        Time step size.
    lle : float, default=1.0
        Largest Lyapunov exponent of the system.

    Returns
    -------
    float
        Weighted geometric mean error. Lower is better.
    """
    errors = _compute_errors(y_true, y_pred, metric)
    geom_mean = gmean(errors, axis=0)

    timesteps = np.arange(geom_mean.shape[0], dtype=float)
    weights = np.exp(-lle * dt * timesteps)
    weights /= np.sum(weights) + 1e-12

    return float(np.sum(weights * geom_mean))


def standard_loss(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    /,
    metric: MetricType = "nrmse",
) -> float:
    """Mean geometric mean error across all timesteps.

    Simple baseline loss suitable for both stable and unstable systems.

    Parameters
    ----------
    y_true : ndarray
        True values of shape (B, T, D).
    y_pred : ndarray
        Predicted values of shape (B, T, D).
    metric : {"rmse", "mse", "mae", "nrmse"}, default="nrmse"
        Error metric to compute.

    Returns
    -------
    float
        Mean of geometric mean errors. Lower is better.
    """
    errors = _compute_errors(y_true, y_pred, metric)
    geom_mean = gmean(errors, axis=0)
    return float(np.mean(geom_mean))


def discounted_rmse(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    /,
    metric: MetricType = "rmse",
    half_life: int = 64,
) -> float:
    """Time-discounted error with exponential half-life.

    Applies exponential discounting to errors, giving more weight to early
    time steps. The discount factor decays to 0.5 after half_life steps.

    Parameters
    ----------
    y_true : ndarray
        True values of shape (B, T, D).
    y_pred : ndarray
        Predicted values of shape (B, T, D).
    metric : {"rmse", "mse", "mae", "nrmse"}, default="rmse"
        Error metric to compute.
    half_life : int, default=64
        Half-life in time steps.

    Returns
    -------
    float
        Weighted average error. Lower is better.
    """
    errors = _compute_errors(y_true, y_pred, metric)  # (B, T)
    e_t = np.mean(errors, axis=0)  # (T,)

    gamma = 0.5 ** (1.0 / max(half_life, 1))
    weights = gamma ** np.arange(1, e_t.shape[0] + 1)

    return float(np.sum(weights * e_t) / np.sum(weights))


# Loss function registry
LOSSES: dict[str, LossProtocol] = {
    "efh": expected_forecast_horizon,
    "horizon": forecast_horizon,
    "lyap": lyapunov_weighted,
    "standard": standard_loss,
    "discounted": discounted_rmse,
}


def get_loss(key_or_callable: str | LossProtocol) -> LossProtocol:
    """Get a loss function by key or return the callable directly.

    Parameters
    ----------
    key_or_callable : str or callable
        Either a string key from LOSSES (e.g., "efh") or a custom callable
        following the LossProtocol interface.

    Returns
    -------
    LossProtocol
        The loss function callable.

    Raises
    ------
    KeyError
        If the string key is not found in LOSSES.
    TypeError
        If the provided callable doesn't match LossProtocol.

    Example
    -------
    >>> loss_fn = get_loss("efh")
    >>> loss_fn = get_loss(my_custom_loss)
    """
    if isinstance(key_or_callable, str):
        if key_or_callable not in LOSSES:
            available = ", ".join(LOSSES.keys())
            raise KeyError(f"Unknown loss '{key_or_callable}'. Available: {available}")
        return LOSSES[key_or_callable]

    if not callable(key_or_callable):
        raise TypeError("Loss must be a string key or a callable.")

    return key_or_callable
