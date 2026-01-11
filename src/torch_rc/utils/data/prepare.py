"""Data preparation utilities for ESN training and forecasting.

Provides functions for splitting time series data into:
- Warmup data (for reservoir state synchronization)
- Training data and targets
- Forecast warmup data
- Validation data
"""

from typing import Dict, List, Literal, Tuple, Union

import torch

# Type aliases
NormMethod = Literal["minmax", "standard", "noncentered", "meanpreserving"]
ESNDataSplits = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def normalize_data(
    data: torch.Tensor,
    method: NormMethod = "minmax",
    stats: Dict[str, torch.Tensor] | None = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Normalize time series data.

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
        instead of computing from data. Keys depend on method.

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

    if method == "minmax":
        # Scale to [-1, 1]
        data_min, data_range = stats["min"], stats["range"]
        normalized = 2 * (data - data_min) / data_range - 1

    elif method == "standard":
        # Zero mean, unit variance
        mean, std = stats["mean"], stats["std"]
        normalized = (data - mean) / std

    elif method == "noncentered":
        # Scale by max absolute value
        scale = stats["scale"]
        normalized = data / scale

    elif method == "meanpreserving":
        # Scale deviations to [-1, 1], restore mean
        mean, maxdev = stats["mean"], stats["maxdev"]
        normalized = (data - mean) / maxdev + mean

    else:
        raise ValueError(
            f"Unknown normalization method: '{method}'. "
            "Supported: 'minmax', 'standard', 'noncentered', 'meanpreserving'"
        )

    return normalized, stats


def _compute_stats(data: torch.Tensor, method: NormMethod) -> Dict[str, torch.Tensor]:
    """Compute normalization statistics from data."""
    # Compute along time axis, keep batch and feature dims
    if method == "minmax":
        data_min = data.min(dim=1, keepdim=True).values
        data_max = data.max(dim=1, keepdim=True).values
        data_range = data_max - data_min
        # Prevent division by zero
        data_range = torch.where(data_range == 0, torch.ones_like(data_range), data_range)
        return {"min": data_min, "range": data_range}

    elif method == "standard":
        mean = data.mean(dim=1, keepdim=True)
        std = data.std(dim=1, keepdim=True)
        std = torch.where(std == 0, torch.ones_like(std), std)
        return {"mean": mean, "std": std}

    elif method == "noncentered":
        scale = data.abs().max(dim=1, keepdim=True).values
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        return {"scale": scale}

    elif method == "meanpreserving":
        mean = data.mean(dim=1, keepdim=True)
        centered = data - mean
        maxdev = centered.abs().max(dim=1, keepdim=True).values
        maxdev = torch.where(maxdev == 0, torch.ones_like(maxdev), maxdev)
        return {"mean": mean, "maxdev": maxdev}

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
) -> ESNDataSplits:
    """Prepare time series data for ESN training and forecasting.

    Splits data into segments appropriate for ESN workflows:
    1. Warmup: Initial steps for reservoir state synchronization
    2. Train: Training input data
    3. Target: Training targets (train data shifted by 1 step)
    4. Forecast warmup: Last warmup_steps of training for forecast initialization
    5. Validation: Held-out data for testing

    Data layout:
    ```
    [discard][warmup][--------train-------][---val---]
                     ^target starts here+1
    ```

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
        training data and applied to all splits.
    norm_method : str, default="minmax"
        Normalization method if normalize=True.

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
        Validation data, shape (B, val_steps, D).

    Raises
    ------
    ValueError
        If data is too short for the requested splits.

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
    """
    _, timesteps, _ = data.shape

    # Validate discard_steps
    if discard_steps >= timesteps:
        raise ValueError(f"discard_steps ({discard_steps}) must be less than data length ({timesteps})")

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

    # Determine validation length
    if val_steps is None:
        # Use all remaining data (need +1 for target shift)
        val_steps = timesteps - train_end - 1
    else:
        required = train_end + val_steps + 1  # +1 for target shift
        if required > timesteps:
            raise ValueError(
                f"Required data ({required} = warmup + train + val + 1) "
                f"exceeds available length ({timesteps})"
            )

    # Split data
    warmup = data[:, :warmup_steps, :]
    train = data[:, warmup_steps:train_end, :]
    target = data[:, warmup_steps + 1 : train_end + 1, :]
    forecast_warmup = train[:, -warmup_steps:, :]
    val = data[:, train_end : train_end + val_steps, :]

    # Optional normalization (compute stats from training data only)
    if normalize:
        _, stats = normalize_data(train, method=norm_method)
        warmup, _ = normalize_data(warmup, method=norm_method, stats=stats)
        train, _ = normalize_data(train, method=norm_method, stats=stats)
        target, _ = normalize_data(target, method=norm_method, stats=stats)
        forecast_warmup, _ = normalize_data(forecast_warmup, method=norm_method, stats=stats)
        val, _ = normalize_data(val, method=norm_method, stats=stats)

    return warmup, train, target, forecast_warmup, val


def load_and_prepare(
    paths: Union[str, List[str]],
    warmup_steps: int,
    train_steps: int,
    val_steps: int | None = None,
    discard_steps: int = 0,
    normalize: bool = False,
    norm_method: NormMethod = "minmax",
    **load_kwargs,
) -> ESNDataSplits:
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
    **load_kwargs
        Additional arguments passed to load_file (e.g., key for .npz).

    Returns
    -------
    ESNDataSplits
        Tuple of (warmup, train, target, forecast_warmup, val) tensors.

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
    tensors = [load_file(p, **load_kwargs) for p in paths]
    data = torch.cat(tensors, dim=0) if len(tensors) > 1 else tensors[0]

    return prepare_esn_data(
        data,
        warmup_steps=warmup_steps,
        train_steps=train_steps,
        val_steps=val_steps,
        discard_steps=discard_steps,
        normalize=normalize,
        norm_method=norm_method,
    )
