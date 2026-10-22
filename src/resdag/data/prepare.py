"""Data preparation utilities for ESN training and forecasting.

Provides functions for splitting time series data into:
- Warmup data (for reservoir state synchronization)
- Training data and targets
- Forecast warmup data
- Validation data
"""

import operator
import warnings
from pathlib import Path
from typing import Any, Literal

import torch

from resdag.core._forecast import ForecastAlignment, validate_alignment

# Type aliases
NormMethod = Literal["minmax", "standard", "noncentered", "meanpreserving"]
ESNDataSplits = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
ESNDataSplitsWithStats = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor] | None,
]


def _validate_step_count(name: str, value: int, *, allow_zero: bool = False) -> int:
    """Accept integer indices without silently truncating floats or booleans."""
    requirement = "non-negative" if allow_zero else "a positive integer"
    try:
        if isinstance(value, bool):
            raise TypeError
        count = operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be {requirement}, got {value!r}") from None
    if count < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be {requirement}, got {value!r}")
    return count


def _validate_series_shape(data: torch.Tensor) -> None:
    """Require the batch/time/feature convention before slicing or reducing."""
    if data.ndim != 3 or data.shape[0] == 0 or data.shape[-1] == 0:
        raise ValueError(
            "data must have shape (batch, time, features) with non-empty batch and "
            f"feature axes, got {tuple(data.shape)}"
        )


def normalize_data(
    data: torch.Tensor,
    method: NormMethod = "minmax",
    stats: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Normalize time series data globally.

    Computes statistics across all batches and timesteps, applying the same
    normalization to the entire dataset. This is the correct approach when
    batches contain trajectories from the same dynamical system.

    Parameters
    ----------
    data : torch.Tensor
        Data tensor of shape (B, T, D).
    method : {"minmax", "standard", "noncentered", "meanpreserving"}
        Normalization method:
        - "minmax": Scale to [-1, 1] range
        - "standard": Zero mean, unit variance
        - "noncentered": Scale by max absolute value (preserves zero)
        - "meanpreserving": Scale deviations to [-1, 1], then restore mean
    stats : dict, optional
        Pre-computed statistics for normalization. If provided, these are used
        instead of computing from data. They are applied on the input device
        and dtype without modifying the supplied statistics.

    Returns
    -------
    normalized : torch.Tensor
        Normalized data with same shape as input.
    stats : dict
        Statistics used for normalization (for applying to other data).

    Examples
    --------
    >>> data = torch.randn(1, 100, 3)
    >>> normalized, stats = normalize_data(data, method="minmax")
    >>> # Apply same normalization to new data
    >>> new_normalized, _ = normalize_data(new_data, method="minmax", stats=stats)
    """
    _validate_series_shape(data)
    if stats is None:
        stats = _compute_stats(data, method)

    normalized = _apply_norm(data, method, stats)
    return normalized, stats


def denormalize_data(
    data: torch.Tensor,
    method: NormMethod,
    stats: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Invert :func:`normalize_data`, mapping data back to its original scale.

    Applies the exact inverse of the transform that :func:`normalize_data` (and
    ``prepare_esn_data(normalize=True, ...)``) applied, using the fitted
    statistics. This closes the normalize -> forecast -> report loop: feed model
    predictions through ``denormalize_data`` to recover physical units, compute
    errors in the original scale, or plot against the raw series.

    Parameters
    ----------
    data : torch.Tensor
        Normalized data of shape (B, T, D) to map back to the original scale.
    method : {"minmax", "standard", "noncentered", "meanpreserving"}
        Normalization method that produced ``data``. Must match the method used
        for the forward transform. The inverse of each method is:

        - "minmax": ``(data + 1) / 2 * range + min``
        - "standard": ``data * std + mean``
        - "noncentered": ``data * scale``
        - "meanpreserving": ``(data - mean) * maxdev + mean``
    stats : dict
        Fitted statistics returned by :func:`normalize_data` (or by
        :func:`prepare_esn_data` with ``return_stats=True``). The required keys
        depend on ``method``: ``{"min", "range"}`` for "minmax",
        ``{"mean", "std"}`` for "standard", ``{"scale"}`` for "noncentered",
        and ``{"mean", "maxdev"}`` for "meanpreserving".

    Returns
    -------
    torch.Tensor
        Data restored to the original scale, with the same shape as ``data``.

    Raises
    ------
    ValueError
        If ``method`` is not a recognized normalization method.

    See Also
    --------
    normalize_data : The forward transform this function inverts.
    prepare_esn_data : Returns the fitted ``stats`` when ``return_stats=True``.

    Examples
    --------
    >>> data = torch.randn(1, 100, 3)
    >>> normalized, stats = normalize_data(data, method="standard")
    >>> restored = denormalize_data(normalized, method="standard", stats=stats)
    >>> torch.allclose(restored, data, atol=1e-6)
    True
    """
    return _apply_denorm(data, method, stats)


def _compute_stats(data: torch.Tensor, method: NormMethod) -> dict[str, torch.Tensor]:
    """Compute feature-wise normalization statistics from data."""
    if data.shape[1] == 0:
        raise ValueError("Cannot fit normalization statistics on an empty time axis")
    reduce_dims = (0, 1)  # batch and time

    if method == "minmax":
        data_min = data.amin(dim=reduce_dims, keepdim=True)
        data_max = data.amax(dim=reduce_dims, keepdim=True)
        data_range = data_max - data_min
        data_range = torch.where(data_range == 0, torch.ones_like(data_range), data_range)
        return {"min": data_min, "range": data_range}

    elif method == "standard":
        mean = data.mean(dim=reduce_dims, keepdim=True)
        # Keep the public sample-std convention; one observation has no spread.
        correction = 1 if data.shape[0] * data.shape[1] > 1 else 0
        std = data.std(dim=reduce_dims, keepdim=True, correction=correction)
        std = torch.where(std == 0, torch.ones_like(std), std)
        return {"mean": mean, "std": std}

    elif method == "noncentered":
        scale = data.abs().amax(dim=reduce_dims, keepdim=True)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        return {"scale": scale}

    elif method == "meanpreserving":
        mean = data.mean(dim=reduce_dims, keepdim=True)
        centered = data - mean
        maxdev = centered.abs().amax(dim=reduce_dims, keepdim=True)
        maxdev = torch.where(maxdev == 0, torch.ones_like(maxdev), maxdev)
        return {"mean": mean, "maxdev": maxdev}

    else:
        raise ValueError(f"Unknown normalization method: '{method}'")


def _apply_norm(
    data: torch.Tensor, method: NormMethod, stats: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Apply fitted statistics on the input device/dtype without mutating them."""
    stats = {
        key: value.to(
            device=data.device, dtype=data.dtype if data.is_floating_point() else value.dtype
        )
        for key, value in stats.items()
    }
    if method == "minmax":
        return 2 * ((data - stats["min"]) / stats["range"]) - 1
    elif method == "standard":
        return (data - stats["mean"]) / stats["std"]
    elif method == "noncentered":
        return data / stats["scale"]
    elif method == "meanpreserving":
        return (data - stats["mean"]) / stats["maxdev"] + stats["mean"]
    else:
        raise ValueError(f"Unknown normalization method: '{method}'")


def _apply_denorm(
    data: torch.Tensor, method: NormMethod, stats: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Apply the inverse of :func:`_apply_norm` to data (exact round-trip)."""
    stats = {
        key: value.to(
            device=data.device, dtype=data.dtype if data.is_floating_point() else value.dtype
        )
        for key, value in stats.items()
    }
    if method == "minmax":
        return (data + 1) / 2 * stats["range"] + stats["min"]
    elif method == "standard":
        return data * stats["std"] + stats["mean"]
    elif method == "noncentered":
        return data * stats["scale"]
    elif method == "meanpreserving":
        return (data - stats["mean"]) * stats["maxdev"] + stats["mean"]
    else:
        raise ValueError(f"Unknown normalization method: '{method}'")


def prepare_esn_data(
    data: torch.Tensor,
    warmup_steps: int,
    train_steps: int,
    val_steps: int | None = None,
    discard_steps: int = 0,
    normalize: bool = False,
    norm_method: NormMethod = "minmax",
    return_stats: bool = False,
    alignment: ForecastAlignment = "next",
) -> ESNDataSplits | ESNDataSplitsWithStats:
    """Prepare time series data for ESN training and forecasting.

    Splits data into segments appropriate for ESN workflows:
    1. Warmup: Initial steps for reservoir state synchronization
    2. Train: Training input data
    3. Target: Training targets (train data shifted by 1 step)
    4. Forecast warmup: Last warmup_steps of all observed data, including the
       final training target, for forecast initialization
    5. Validation: Held-out data starting after that final training target.

    Data layout (``train_end = warmup_steps + train_steps``)::

        [discard][warmup][------- train -------][s][------ val ------]

    ``s`` is ``data[train_end]``: the final training target, which is already
    observed during fitting. The forecast warmup includes it. Its readout
    predicts the first held-out value, ``data[train_end + 1]``, so
    ``forecast(forecast_warmup, horizon=val_steps)`` lines up element-for-element
    with ``val`` (no manual shift).


    Parameters
    ----------
    data : torch.Tensor
        Input time series of shape (B, T, D).
    warmup_steps : int
        Number of steps for reservoir warmup/synchronization.
    train_steps : int
        Number of training steps (after warmup).
    val_steps : int, optional
        Number of validation steps. If None, uses all remaining data.
    discard_steps : int, default=0
        Number of initial steps to discard (e.g., initial transients).
    normalize : bool, default=False
        Whether to normalize data. If True, statistics are computed from
        training data and applied to all splits globally.
    norm_method : str, default="minmax"
        Normalization method if normalize=True.
    return_stats : bool, default=False
        If True, append the fitted normalization statistics as a 6th return
        element so the normalize -> forecast -> report loop can be closed via
        :func:`denormalize_data`. The element is the ``stats`` dict when
        ``normalize=True`` and ``None`` when ``normalize=False`` (nothing was
        fitted). When False (the default), the legacy 5-tuple is returned
        unchanged.
    alignment : {"next", "legacy"}, default="next"
        ``"next"`` includes the final training target in the forecast history.
        ``"legacy"`` ends that history one sample earlier; pair this with
        ``model.forecast(..., alignment="legacy")`` to reproduce old runs.
        Training and validation windows are identical in both modes.

    Returns
    -------
    warmup : torch.Tensor
        Warmup data, shape (B, warmup_steps, D).
    train : torch.Tensor
        Training input, shape (B, train_steps, D).
    target : torch.Tensor
        Training target (shifted by 1), shape (B, train_steps, D).
    forecast_warmup : torch.Tensor
        Last warmup_steps of the observed prefix for forecast init, shape
        (B, warmup_steps, D). Includes earlier warmup history if train_steps
        is shorter than warmup_steps.
    val : torch.Tensor
        Validation data, shape (B, val_steps, D). Starts at ``data[train_end + 1]``
        so it aligns directly with an autoregressive ``forecast`` of the same
        horizon. The final training target belongs to the observed history.
    stats : dict or None
        Only returned when ``return_stats=True``. The fitted normalization
        statistics (as returned by :func:`normalize_data`) when ``normalize=True``,
        otherwise ``None``. Pass to :func:`denormalize_data` to map forecasts
        back to the original scale.

    Raises
    ------
    ValueError
        If ``warmup_steps`` or ``train_steps`` is not strictly positive, if
        ``discard_steps`` is negative, if ``val_steps`` is negative, or if the
        data is too short for the requested splits.

    Warns
    -----
    UserWarning
        If the resulting validation window is empty (``val_steps=0`` or the data
        is exhausted by warmup, training inputs, and their final target), so no held-out validation data
        remains.

    See Also
    --------
    denormalize_data : Inverts the normalization using the returned ``stats``.

    Examples
    --------
    >>> data = torch.randn(1, 1000, 3)  # (batch=1, time=1000, features=3)
    >>> warmup, train, target, f_warmup, val = prepare_esn_data(
    ...     data, warmup_steps=100, train_steps=500, val_steps=200
    ... )
    >>> print(warmup.shape)   # (1, 100, 3)
    >>> print(train.shape)    # (1, 500, 3)
    >>> print(target.shape)   # (1, 500, 3)
    >>> print(f_warmup.shape) # (1, 100, 3)
    >>> print(val.shape)      # (1, 200, 3)

    Close the round-trip by keeping the fitted stats and inverting later:

    >>> warmup, train, target, f_warmup, val, stats = prepare_esn_data(
    ...     data, warmup_steps=100, train_steps=500, val_steps=200,
    ...     normalize=True, norm_method="minmax", return_stats=True,
    ... )
    >>> val_physical = denormalize_data(val, "minmax", stats)  # original units
    """
    _validate_series_shape(data)
    validate_alignment(alignment)
    _, timesteps, _ = data.shape
    warmup_steps = _validate_step_count("warmup_steps", warmup_steps)
    train_steps = _validate_step_count("train_steps", train_steps)
    discard_steps = _validate_step_count("discard_steps", discard_steps, allow_zero=True)
    if val_steps is not None:
        val_steps = _validate_step_count("val_steps", val_steps, allow_zero=True)

    # Validate discard_steps
    if discard_steps >= timesteps:
        raise ValueError(
            f"discard_steps ({discard_steps}) must be less than data length ({timesteps})"
        )

    # Trim initial steps
    data = data[:, discard_steps:, :]
    timesteps = data.shape[1]

    # Calculate required length
    train_end = warmup_steps + train_steps

    if train_end >= timesteps:
        raise ValueError(
            f"warmup_steps + train_steps ({train_end}) exceeds "
            f"available data length ({timesteps}) after discarding {discard_steps} steps"
        )

    # The final supervised target data[train_end] is already observed. Include
    # it in canonical forecast history, then hold out data[train_end + 1] onward.
    # Legacy preparation omits that observation from history and relies on the
    # old forecast's additional predicted-seed update to reach the same val index.
    val_start = train_end + 1
    if val_steps is None:
        val_steps = timesteps - val_start
        if val_steps == 0:
            warnings.warn(
                "Validation window is empty: warmup + training inputs + their final "
                f"target consume all {timesteps} available steps (after discarding "
                f"{discard_steps}), leaving no held-out data for val. Reduce "
                "warmup_steps/train_steps or supply more data if you need a val split.",
                UserWarning,
                stacklevel=2,
            )
    else:
        required = val_start + val_steps
        if required > timesteps:
            raise ValueError(
                f"Required data ({required} = warmup + train + 1 final target + val) "
                f"exceeds available length ({timesteps}). The +1 supplies the final "
                "supervised target data[train_end], which is already observed "
                "and must precede held-out validation."
            )
        if val_steps == 0:
            warnings.warn(
                "val_steps=0 produces an empty validation split. Pass a positive "
                "val_steps (or None to use all remaining data) if you need a val "
                "window.",
                UserWarning,
                stacklevel=2,
            )

    # Split data
    warmup = data[:, :warmup_steps, :]
    train = data[:, warmup_steps:train_end, :]
    target = data[:, warmup_steps + 1 : train_end + 1, :]
    # Include earlier observed history when train_steps < warmup_steps.
    forecast_end = train_end + (alignment == "next")
    forecast_warmup = data[:, forecast_end - warmup_steps : forecast_end, :]
    val = data[:, val_start : val_start + val_steps, :]

    # Optional normalization (compute stats from training data only)
    stats: dict[str, torch.Tensor] | None = None
    if normalize:
        stats = _compute_stats(train, norm_method)
        warmup = _apply_norm(warmup, norm_method, stats)
        train = _apply_norm(train, norm_method, stats)
        target = _apply_norm(target, norm_method, stats)
        forecast_warmup = _apply_norm(forecast_warmup, norm_method, stats)
        val = _apply_norm(val, norm_method, stats)

    if return_stats:
        return warmup, train, target, forecast_warmup, val, stats
    return warmup, train, target, forecast_warmup, val


def load_and_prepare(
    paths: str | Path | list[str | Path],
    warmup_steps: int,
    train_steps: int,
    val_steps: int | None = None,
    discard_steps: int = 0,
    normalize: bool = False,
    norm_method: NormMethod = "minmax",
    return_stats: bool = False,
    dtype: torch.dtype | None = None,
    alignment: ForecastAlignment = "next",
    **load_kwargs: Any,
) -> ESNDataSplits | ESNDataSplitsWithStats:
    """Load data from file(s) and prepare for ESN training.

    Convenience function that combines loading and preparation.
    If multiple paths are provided, data is concatenated along the batch dimension.

    Parameters
    ----------
    paths : str, pathlib.Path, or list of paths
        Path(s) to data file(s). Supports .csv, .npy, .npz, .nc
    warmup_steps : int
        Number of warmup steps.
    train_steps : int
        Number of training steps.
    val_steps : int, optional
        Number of validation steps.
    discard_steps : int, default=0
        Initial steps to discard.
    normalize : bool, default=False
        Whether to normalize data.
    norm_method : str, default="minmax"
        Normalization method.
    return_stats : bool, default=False
        If True, append the fitted normalization statistics as a 6th return
        element (``None`` when ``normalize=False``). Forwarded to
        :func:`prepare_esn_data`; see its docstring for details.
    dtype : torch.dtype, optional
        Desired tensor dtype. If None, uses ``torch.get_default_dtype()``.
    alignment : {"next", "legacy"}, default="next"
        Forecast timing convention forwarded to :func:`prepare_esn_data`.
    **load_kwargs
        Additional arguments passed to load_file (e.g., key for .npz).

    Returns
    -------
    ESNDataSplits or ESNDataSplitsWithStats
        Tuple of (warmup, train, target, forecast_warmup, val) tensors, with a
        trailing ``stats`` element when ``return_stats=True``.

    See Also
    --------
    denormalize_data : Inverts the normalization using the returned ``stats``.

    Examples
    --------
    >>> splits = load_and_prepare(
    ...     "timeseries.csv",
    ...     warmup_steps=100,
    ...     train_steps=500,
    ...     val_steps=200,
    ...     normalize=True,
    ... )
    >>> warmup, train, target, f_warmup, val = splits
    """
    from .io import load_file

    # Handle single path or list of paths
    if isinstance(paths, (str, Path, type(None))):
        paths = [paths] if paths else []

    if not paths:
        raise ValueError("At least one data path must be provided")

    # Load and concatenate along batch dimension
    tensors = [load_file(p, dtype=dtype, **load_kwargs) for p in paths]
    data = torch.cat(tensors, dim=0) if len(tensors) > 1 else tensors[0]

    return prepare_esn_data(
        data,
        warmup_steps=warmup_steps,
        train_steps=train_steps,
        val_steps=val_steps,
        discard_steps=discard_steps,
        normalize=normalize,
        norm_method=norm_method,
        return_stats=return_stats,
        alignment=alignment,
    )
