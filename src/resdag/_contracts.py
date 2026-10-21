"""Shared tensor contracts at public model and fitting boundaries.

Validation does not move, cast or detach data. A caller can therefore fix a
device/dtype mismatch explicitly without silently losing precision or gradients.
The recurrent inner loops deliberately stay outside these checks.
"""

from collections.abc import Sequence
from itertools import chain

import torch
from torch import nn


def module_reference(module: nn.Module) -> torch.Tensor | None:
    """Find a floating parameter/buffer describing a module's working placement."""
    return next(
        (
            tensor
            for tensor in chain(module.parameters(), module.buffers())
            if tensor.is_floating_point()
        ),
        None,
    )


def validate_sequence_inputs(
    inputs: Sequence[torch.Tensor],
    *,
    feature_sizes: Sequence[int],
    reference: torch.Tensor | None = None,
    label: str = "inputs",
    allow_empty: bool = True,
) -> None:
    """Check aligned ``(batch, time, features)`` input streams once per call."""
    if len(inputs) != len(feature_sizes):
        raise ValueError(
            f"{label} has {len(inputs)} tensors but the model expects "
            f"{len(feature_sizes)} input tensors. Must match."
        )
    if not inputs:
        raise ValueError(f"{label} must contain at least one input tensor")
    for index, (tensor, width) in enumerate(zip(inputs, feature_sizes)):
        name = f"{label}[{index}]"
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.ndim != 3:
            raise ValueError(
                f"{name} must be 3-D (batch, time, features); got {tuple(tensor.shape)}"
            )
        if tensor.shape[0] == 0:
            raise ValueError(f"{name} must have a non-empty batch axis")
        if tensor.shape[-1] != width:
            raise ValueError(
                f"{name} has {tensor.shape[-1]} features but the model expects {width}"
            )
        if not allow_empty and tensor.shape[1] == 0:
            raise ValueError(f"{name} must contain at least one timestep")
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must use a floating dtype, got {tensor.dtype}")
        if tensor.shape[:2] != inputs[0].shape[:2]:
            raise ValueError(f"{label} tensors must share batch size and timesteps")
        ref = reference if reference is not None else inputs[0]
        if tensor.device != ref.device:
            raise ValueError(
                f"{name} is on device={tensor.device}, expected device={ref.device}. "
                "Move the model and all inputs to the same device."
            )
        if tensor.dtype != ref.dtype and not torch.is_autocast_enabled(tensor.device.type):
            raise ValueError(
                f"{name} has dtype={tensor.dtype}, expected dtype={ref.dtype}. "
                "Use tensor.to(dtype=...) or model.to(dtype=...) to match them."
            )


def prepare_fit_inputs(
    states: torch.Tensor,
    targets: torch.Tensor,
    *,
    in_features: int,
    out_features: int,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and flatten algebraic readout data without changing its values."""
    for name, tensor in (("states", states), ("targets", targets)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{label}: {name} must be a torch.Tensor")
        if tensor.ndim not in (2, 3):
            raise ValueError(
                f"{label}: {name} must be 2D (samples, features) or "
                f"3D (batch, time, features), got shape {tuple(tensor.shape)}."
            )
        if tensor.numel() == 0:
            raise ValueError(f"{label}: {name} must contain at least one sample.")
        if tensor.is_complex():
            raise TypeError(f"{label}: {name} must be real-valued")
    if not states.is_floating_point():
        raise TypeError(f"{label}: states must use a floating dtype, got {states.dtype}")
    if states.ndim == targets.ndim == 3 and states.shape[:2] != targets.shape[:2]:
        raise ValueError(
            f"{label}: states and targets must share batch and time axes; "
            f"got {tuple(states.shape[:2])} and {tuple(targets.shape[:2])}."
        )
    if states.device != targets.device:
        raise ValueError(f"{label}: states and targets must be on the same device")
    states = states.reshape(-1, states.shape[-1])
    targets = targets.reshape(-1, targets.shape[-1])
    if states.shape[0] != targets.shape[0]:
        raise ValueError(
            f"{label}: sample count mismatch. States have {states.shape[0]} samples "
            f"after flattening, targets have {targets.shape[0]}."
        )
    if states.shape[-1] != in_features:
        raise ValueError(
            f"{label}: state feature dimension ({states.shape[-1]}) "
            f"does not match readout in_features ({in_features})."
        )
    if targets.shape[-1] != out_features:
        raise ValueError(
            f"{label}: target feature dimension ({targets.shape[-1]}) "
            f"does not match readout out_features ({out_features})."
        )
    for name, tensor in (("states", states), ("targets", targets)):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{label}: {name} contain non-finite values (NaN or Inf).")
    return states, targets
