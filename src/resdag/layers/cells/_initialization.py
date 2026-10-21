"""Shared vector and initial-state hooks for reservoir cells."""

from typing import Any

import torch
from torch import nn

from resdag.init.utils import InitializerSpec, resolve_initializer


def device_matches_seed(seed: object, device: torch.device | str | None) -> bool:
    """Keep a supplied generator when its device matches allocations."""
    return not isinstance(seed, torch.Generator) or seed.device == torch.device(device or "cpu")


def initialize_vector(weight: torch.Tensor, spec: InitializerSpec, *, seed: int | None) -> None:
    """Apply the rectangular initializer grammar to a vector viewed as ``(1, N)``."""
    initializer = resolve_initializer(spec, seed=seed)
    if initializer is None:
        return
    with torch.no_grad():
        initializer.initialize(weight.view(1, -1))
    if not torch.isfinite(weight).all():
        raise ValueError("Bias initializer must produce finite values")


def _restore_initial_state(
    module: nn.Module,
    state_dict: dict[str, Any],
    prefix: str,
    local_metadata: dict[str, Any],
    strict: bool,
    missing_keys: list[str],
    unexpected_keys: list[str],
    error_msgs: list[str],
) -> None:
    """Reconcile optional templates when loading historical or configured weights."""
    reference = next(module.parameters())
    key = prefix + "_initial_state"
    width = int(getattr(module, "state_size"))
    module._buffers["_initial_state"] = reference.new_empty(1, width) if key in state_dict else None


def install_initial_state(
    module: nn.Module,
    spec: InitializerSpec,
    *,
    seed: int | None,
    state_size: int,
    reference: torch.Tensor,
) -> None:
    """Sample one persistent reset template, independent of future batch size."""
    initial = None
    initializer = resolve_initializer(spec, seed=seed)
    if initializer is not None:
        initial = reference.new_empty(1, state_size)
        with torch.no_grad():
            initializer.initialize(initial)
        if not torch.isfinite(initial).all():
            raise ValueError("Initial-state initializer must produce finite values")
    module.register_buffer("_initial_state", initial)
    module.register_load_state_dict_pre_hook(_restore_initial_state)


def restore_initial_state_support(module: nn.Module) -> None:
    """Upgrade historical full-model pickles before subsequent weight restores."""
    module._buffers.setdefault("_initial_state", None)
    if not any(
        getattr(hook, "hook", None) is _restore_initial_state
        for hook in module._load_state_dict_pre_hooks.values()
    ):
        module.register_load_state_dict_pre_hook(_restore_initial_state)


def initialized_state(
    module: nn.Module, batch_size: int, state_size: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Expand the reset template into independent per-example state storage."""
    template = module._buffers.get("_initial_state")
    if template is None:
        return torch.zeros(batch_size, state_size, device=device, dtype=dtype)
    return template.to(device=device, dtype=dtype).expand(batch_size, -1).clone()
