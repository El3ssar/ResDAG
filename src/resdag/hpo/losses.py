"""Loss functions for hyperparameter optimization.

This module provides specialized loss functions for evaluating multi-step
forecasts in reservoir computing applications.  All functions operate on
batched predictions with shape ``(B, T, D)`` where *B* is batch size, *T* is
time steps, and *D* is the number of dimensions.

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

See Also
--------
resdag.hpo.objective : Objective builder that wraps these losses for Optuna.
resdag.hpo.run : High-level HPO orchestrator.
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
]

MetricType = Literal["rmse", "mse", "mae", "nrmse"]


@runtime_checkable
class LossProtocol(Protocol):
    """Protocol for HPO loss functions.

    All loss functions must accept *y_true* and *y_pred* arrays of shape
    ``(B, T, D)`` as positional-only arguments and return a single ``float``
    value to minimize.

    Examples
    --------
    Implementing a custom loss:

    >>> def my_loss(y_true, y_pred, /, **kwargs):
    ...     return float(np.mean((y_true - y_pred) ** 2))
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
    softness: float = 0.04,
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
    # log_g = np.log(np.clip(good_t, 1e-12, 1.0))
    surv_t = np.cumprod(good_t)

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
    """Contiguous valid forecast horizon length.

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

    return -float(valid_len)


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


def lyapunov_weighted(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    /,
    metric: MetricType = "rmse",
    lyapunov_t: int = 64,
) -> float:
    """Lyapunov-compensated time-weighted error.

    Computes a time-weighted error where weights decay exponentially with
    characteristic time equal to the Lyapunov time. This compensates for
    expected exponential error growth in chaotic systems by applying a
    weight of exp(-t / lyapunov_t) at time step t.

    At t = lyapunov_t, the weight equals exp(-1).

    Parameters
    ----------
    y_true : ndarray
        True values of shape (B, T, D).
    y_pred : ndarray
        Predicted values of shape (B, T, D).
    metric : {"rmse", "mse", "mae", "nrmse"}, default="rmse"
        Error metric to compute per time step.
    lyapunov_t : int, default=64
        Characteristic Lyapunov time (in time steps) controlling the
        exponential decay rate of the weights.

    Returns
    -------
    float
        Exponentially time-weighted average error. Lower is better.
    """
    errors = _compute_errors(y_true, y_pred, metric)  # (B, T)
    e_t = gmean(errors, axis=0)  # (T,)

    t = np.arange(e_t.shape[0])
    weights = np.exp(-t / lyapunov_t)

    return float(np.sum(weights * e_t) / np.sum(weights))


def soft_valid_horizon(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    /,
    metric: MetricType = "rmse",
    threshold: float = 0.3,
    n: int = 6,
) -> float:
    """Soft forecast horizon via cumulative survival probability.

    At each timestep a soft indicator measures how "good" the prediction is
    using a Hill-function gate:

    .. math::

        g_t = \\frac{1}{1 + (e_t / \\theta)^n}

    where :math:`e_t` is the median error at step *t*, :math:`\\theta` is the
    threshold, and *n* controls the sharpness.  A cumulative product of these
    indicators gives a survival probability that drops to zero once any step
    fails, mimicking a contiguous forecast horizon:

    .. math::

        H = \\sum_t \\prod_{i \\leq t} g_i

    The function is numerically safe: error ratios are clipped before
    exponentiation to prevent overflow.

    Parameters
    ----------
    y_true : ndarray
        True values of shape (B, T, D).
    y_pred : ndarray
        Predicted values of shape (B, T, D).
    metric : {"rmse", "mse", "mae", "nrmse"}, default="rmse"
        Error metric to compute per timestep.
    threshold : float, default=0.3
        Error threshold below which predictions are considered "good".
    n : int, default=6
        Hill exponent controlling gate sharpness.  Lower values (4-8) give
        smoother landscapes suitable for TPE optimisation; higher values
        (>12) approach a hard step function with poor gradient signal.

    Returns
    -------
    float
        Negative expected horizon.  Lower (more negative) is better.

    Notes
    -----
    Compared to :func:`expected_forecast_horizon` (which uses a sigmoid gate
    and batch-median), this loss uses a Hill-function gate and batch-median.
    The Hill gate has a sharper but still tunable transition and is symmetric
    around the threshold on a log-error scale.

    See Also
    --------
    expected_forecast_horizon : Sigmoid-gated variant with ``cumprod`` survival.
    """
    errors = _compute_errors(y_true, y_pred, metric)  # (B, T)
    e_t = np.median(errors, axis=0)  # (T,) — robust across batch

    # Numerically safe Hill gate: clip the ratio to avoid overflow in x**n
    ratio = np.clip(e_t / threshold, 0.0, 1e4)
    good_t = 1.0 / (1.0 + ratio**n)

    # Survival (cumulative product) — once one step fails, all later are ~0
    surv_t = np.cumprod(good_t)

    horizon = np.sum(surv_t)
    return -float(horizon)


# Registry mapping loss names to loss functions.
LOSSES: dict[str, LossProtocol] = {
    "efh": expected_forecast_horizon,
    "forecast_horizon": forecast_horizon,
    "lyapunov": lyapunov_weighted,
    "standard": standard_loss,
    "soft_horizon": soft_valid_horizon,
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
