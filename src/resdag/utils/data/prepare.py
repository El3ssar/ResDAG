"""Data preparation utilities for ESN training and forecasting.

Provides functions for splitting time series data into:
- Warmup data (for reservoir state synchronization)
- Training data and targets
- Forecast warmup data
- Validation data
"""

from typing import Literal

import torch

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
        instead of computing from data.

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
    reduce_dims = (0, 1)  # batch and time

    if method == "minmax":
        data_min = data.amin(dim=reduce_dims, keepdim=True)
        data_max = data.amax(dim=reduce_dims, keepdim=True)
        data_range = data_max - data_min
        data_range[data_range == 0] = 1.0
        return {"min": data_min, "range": data_range}

    elif method == "standard":
        mean = data.mean(dim=reduce_dims, keepdim=True)
        std = data.std(dim=reduce_dims, keepdim=True)
        std[std == 0] = 1.0
        return {"mean": mean, "std": std}

    elif method == "noncentered":
        scale = data.abs().amax(dim=reduce_dims, keepdim=True)
        scale[scale == 0] = 1.0
        return {"scale": scale}

    elif method == "meanpreserving":
        mean = data.mean(dim=reduce_dims, keepdim=True)
        centered = data - mean
        maxdev = centered.abs().amax(dim=reduce_dims, keepdim=True)
        maxdev[maxdev == 0] = 1.0
        return {"mean": mean, "maxdev": maxdev}

    else:
        raise ValueError(f"Unknown normalization method: '{method}'")


def _apply_norm(
    data: torch.Tensor, method: NormMethod, stats: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Apply normalization statistics to data."""
    if method == "minmax":
        return 2 * (data - stats["min"]) / stats["range"] - 1
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
) -> ESNDataSplits | ESNDataSplitsWithStats:
    """Prepare time series data for ESN training and forecasting.

    Splits data into segments appropriate for ESN workflows:
    1. Warmup: Initial steps for reservoir state synchronization
    2. Train: Training input data
    3. Target: Training targets (train data shifted by 1 step)
    4. Forecast warmup: Last warmup_steps of training for forecast initialization
    5. Validation: Held-out data for testing, starting one step after the train
       window so it aligns with the (purely autoregressive) forecast.

    Data layout (``train_end = warmup_steps + train_steps``)::

        [discard][warmup][------- train -------][s][------ val ------]

    ``s`` is ``data[train_end]``: the autoregressive seam — the final training
    target and the value the forecast warmup's last output predicts (the seed
    the forecast consumes). It is intentionally excluded from ``val``, so
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

    Returns
    -------
    warmup : torch.Tensor
        Warmup data, shape (B, warmup_steps, D).
    train : torch.Tensor
        Training input, shape (B, train_steps, D).
    target : torch.Tensor
        Training target (shifted by 1), shape (B, train_steps, D).
    forecast_warmup : torch.Tensor
        Last warmup_steps of training for forecast init, shape (B, warmup_steps, D).
    val : torch.Tensor
        Validation data, shape (B, val_steps, D). Starts at ``data[train_end + 1]``
        so it aligns directly with an autoregressive ``forecast`` of the same
        horizon (the ``data[train_end]`` seam is the forecast's seed, not a target).
    stats : dict or None
        Only returned when ``return_stats=True``. The fitted normalization
        statistics (as returned by :func:`normalize_data`) when ``normalize=True``,
        otherwise ``None``. Pass to :func:`denormalize_data` to map forecasts
        back to the original scale.

    Raises
    ------
    ValueError
        If data is too short for the requested splits.

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
    _, timesteps, _ = data.shape

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

    # Determine validation length. The autoregressive forecast consumes
    # ``data[train_end]`` as its bootstrap seed (the value the forecast warmup's
    # final output predicts) and only emits predictions from ``data[train_end + 1]``
    # onward, so the validation window starts one step after ``train_end``.
    val_start = train_end + 1
    if val_steps is None:
        val_steps = timesteps - val_start
    else:
        required = val_start + val_steps
        if required > timesteps:
            raise ValueError(
                f"Required data ({required} = warmup + train + 1 seam + val) "
                f"exceeds available length ({timesteps}). The +1 is the "
                f"autoregressive seam step (data[train_end]) the forecast "
                f"consumes as its seed before producing val_steps predictions."
            )

    # Split data
    warmup = data[:, :warmup_steps, :]
    train = data[:, warmup_steps:train_end, :]
    target = data[:, warmup_steps + 1 : train_end + 1, :]
    forecast_warmup = train[:, -warmup_steps:, :]
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
    paths: str | list[str],
    warmup_steps: int,
    train_steps: int,
    val_steps: int | None = None,
    discard_steps: int = 0,
    normalize: bool = False,
    norm_method: NormMethod = "minmax",
    return_stats: bool = False,
    dtype: torch.dtype | None = None,
    **load_kwargs,
) -> ESNDataSplits | ESNDataSplitsWithStats:
    """Load data from file(s) and prepare for ESN training.

    Convenience function that combines loading and preparation.
    If multiple paths are provided, data is concatenated along the batch dimension.

    Parameters
    ----------
    paths : str or list of str
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
    if isinstance(paths, (str, type(None))):
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
    )
