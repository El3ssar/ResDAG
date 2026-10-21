"""Device-local implementations of the built-in horizon and error losses.

The public NumPy functions remain the reference implementation. This adapter
recognizes those exact functions (including keyword-configured partials), so a
custom callable is never replaced based only on its name.
"""

from functools import partial
from inspect import signature
from typing import Any

import torch

from .losses import (
    _GMEAN_FLOOR,
    LossProtocol,
    _warn_if_scale_unsafe,
    expected_forecast_horizon,
    forecast_horizon,
    lyapunov_weighted,
    soft_valid_horizon,
    standard_loss,
)

_BUILTINS = (
    expected_forecast_horizon,
    forecast_horizon,
    lyapunov_weighted,
    soft_valid_horizon,
    standard_loss,
)


def score_builtin_tensor_loss(
    fn: LossProtocol,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    **kwargs: Any,
) -> torch.Tensor | None:
    """Return an on-device scalar, or ``None`` for an unrecognized loss."""
    base = fn
    if isinstance(fn, partial):
        if fn.args:
            return None
        base = fn.func
        kwargs = {**(fn.keywords or {}), **kwargs}
    if not any(base is builtin for builtin in _BUILTINS):
        return None
    # Keep defaults and invalid-argument errors identical to the public API.
    args = signature(base).bind(y_true, y_pred, **kwargs)
    args.apply_defaults()
    options = args.arguments
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {tuple(y_true.shape)} vs {tuple(y_pred.shape)}")

    diff = y_pred - y_true
    metric = options["metric"]
    if metric == "nrmse":
        scale = y_true.std(dim=(0, 1), correction=0, keepdim=True)
        diff = diff / torch.where(scale == 0, 1.0, scale)
        errors = diff.square().mean(dim=2).sqrt()
    elif metric == "rmse":
        errors = diff.square().mean(dim=2).sqrt()
    elif metric == "mse":
        errors = diff.square().mean(dim=2)
    elif metric == "mae":
        errors = diff.abs().mean(dim=2)
    else:
        raise ValueError(f"Unknown metric: '{metric}'. Use 'rmse', 'mse', 'mae', or 'nrmse'.")

    if base is standard_loss or base is lyapunov_weighted:
        e_t = errors.clamp_min(_GMEAN_FLOOR).log().mean(dim=0).exp()
        if base is standard_loss:
            return e_t.mean()
        steps = torch.arange(e_t.shape[0], device=e_t.device, dtype=e_t.dtype)
        weights = (-steps / float(options["lyapunov_t"])).exp()
        return (weights * e_t).sum() / weights.sum()

    threshold = float(options["threshold"])
    _warn_if_scale_unsafe(metric, getattr(base, "__name__", "loss"), threshold)
    # torch.median uses the lower middle item for even batch sizes; NumPy
    # averages both middle items. Sorting also supports half precision, unlike
    # torch.quantile. Propagate NaNs just as np.median does.
    ordered = errors.sort(dim=0).values
    mid = errors.shape[0] // 2
    e_t = ordered[mid] if errors.shape[0] % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    e_t = e_t.masked_fill(errors.isnan().any(dim=0), float("nan"))
    if base is forecast_horizon:
        return -(e_t < threshold).to(errors.dtype).cumprod(dim=0).sum()
    if base is expected_forecast_horizon:
        good = ((threshold - e_t) / float(options["softness"])).sigmoid()
    else:
        ratio = (e_t / threshold).clamp(0.0, 1e4)
        good = 1.0 / (1.0 + ratio ** float(options["n"]))
    return -good.cumprod(dim=0).sum()
