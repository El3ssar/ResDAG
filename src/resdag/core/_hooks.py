"""Hook-aware execution through PyTorch's native module dispatcher."""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import torch
from torch import nn
from torch.nn.modules import module as module_runtime

from resdag.layers.reservoirs import BaseReservoirLayer

_HOOKS = ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks")
_OWNER: ContextVar[nn.Module | None] = ContextVar("resdag_forecast_owner", default=None)


def has_execution_hooks(module: nn.Module) -> bool:
    """Inspect registrations, including global hooks, without executing them."""
    if any(getattr(module_runtime, "_global" + name, None) for name in _HOOKS):
        return True
    return any(getattr(child, name, None) for child in module.modules() for name in _HOOKS)


def eager_with_hooks(compile: bool) -> None:
    if compile:
        warnings.warn(
            "compile=True uses eager execution while PyTorch hooks are registered, "
            "so ordinary module hook behavior is preserved.",
            RuntimeWarning,
            stacklevel=3,
        )


@contextmanager
def forecast_as(owner: nn.Module) -> Iterator[None]:
    token = _OWNER.set(owner)
    try:
        yield
    finally:
        _OWNER.reset(token)


def forecast_owner(workflow: nn.Module) -> nn.Module:
    owner = _OWNER.get()
    return owner if owner is not None and getattr(owner, "workflow", None) is workflow else workflow


def state_paths(module: nn.Module, prefix: str = "") -> dict[str, str]:
    paths = {
        name.removeprefix(prefix): name
        for name, child in module.named_modules()
        if isinstance(child, BaseReservoirLayer)
    }
    if prefix and any(not path.startswith(prefix) for path in paths.values()):
        raise ValueError(
            "Explicit-state execution and forecasting cannot thread reservoirs outside "
            "the declared model.graph. Put recurrent operations in visible functional "
            "graph nodes; ordinary model(x) remains available."
        )
    return paths


def call_with_states(
    module: nn.Module,
    args: tuple[Any, ...],
    states: Mapping[str, torch.Tensor] | None,
    paths: Mapping[str, str],
    *,
    compile: bool = False,
) -> tuple[Any, dict[str, torch.Tensor]]:
    """Call the real module with temporary explicit reservoir buffers.

    functional_call owns PyTorch dispatch and restores the original buffers,
    including on exceptions. Only recurrent buffers are substituted: parameters,
    preprocessing, nested modules and all hook registrations stay owned by the
    original module. Returned states keep their autograd connections.
    """
    eager_with_hooks(compile)
    if states is not None and set(states) != set(paths):
        raise ValueError("Reservoir state keys must match the model")
    modules = dict(module.named_modules())
    if {name for name, child in modules.items() if isinstance(child, BaseReservoirLayer)} != set(
        paths.values()
    ):
        raise ValueError(
            "Hook-compatible explicit execution cannot thread reservoirs outside the "
            "declared graph state. Expose recurrent operations as visible graph nodes."
        )
    overrides: dict[str, Any] = {}
    flags = []
    for name, path in paths.items():
        reservoir = modules[path]
        assert isinstance(reservoir, BaseReservoirLayer)
        value = None if states is None else states[name]
        if states is not None:
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"State for '{name}' must be a torch.Tensor")
            reservoir.cell.validate_state(value)
            value = value.clone()
        overrides[path + ".state" if path else "state"] = value
        flags.append((reservoir, reservoir.detach_state_between_calls, reservoir._state_user_set))
    try:
        for reservoir, _, _ in flags:
            reservoir.detach_state_between_calls = False
            reservoir._state_user_set = states is not None
        output = torch.func.functional_call(module, overrides, args, tie_weights=False)
        final = {
            name: overrides[path + ".state" if path else "state"] for name, path in paths.items()
        }
        if any(not isinstance(value, torch.Tensor) for value in final.values()):
            raise RuntimeError("A hook-compatible forward did not initialize every reservoir state")
        return output, final
    finally:
        for reservoir, detach, user_set in flags:
            reservoir.detach_state_between_calls = detach
            reservoir._state_user_set = user_set


def original_outputs(
    values: tuple[torch.Tensor, ...], names: tuple[str, ...], two_dimensional: bool
) -> torch.Tensor | dict[str, torch.Tensor]:
    restored = tuple(value.squeeze(0) if two_dimensional else value for value in values)
    return restored[0] if len(restored) == 1 else dict(zip(names, restored))
