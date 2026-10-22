"""Fixed-schema reversible preprocessing with staged, state-only fitting."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import nn

from resdag.core._forecast_capabilities import _framewise
from resdag.core._modes import evaluating
from resdag.layers.reservoirs import BaseReservoirLayer


def _tensors(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        **dict(module.named_parameters(remove_duplicate=False)),
        **dict(module.named_buffers(remove_duplicate=False)),
    }


def _validate_tensor_layout(value: torch.Tensor) -> None:
    """Prove state can be copied without writing overlapping storage locations."""
    if value.layout != torch.strided or value.device.type == "meta":
        raise ValueError("Custom transform state requires concrete writable strided tensors")
    if not value.numel():
        return
    # A sorted-stride span proves nonoverlap for ordinary contiguous, transposed
    # and sliced tensors without writing user state or synchronizing a device.
    # Unusual layouts without this proof require an independent clone instead.
    span = 1
    for stride, size in sorted(
        (stride, size) for stride, size in zip(value.stride(), value.shape) if size > 1
    ):
        if stride < span:
            raise ValueError(
                "Custom transform state requires writable nonoverlapping tensor storage"
            )
        span += (size - 1) * stride


def transform_schema(module: nn.Module, width: int) -> dict[str, Any]:
    """Portable structure only; callers reconstruct matching Python configuration."""
    return {
        "width": width,
        "modules": tuple(
            (name, f"{type(child).__module__}.{type(child).__qualname__}")
            for name, child in module.named_modules()
        ),
        "parameters": tuple(
            (name, tuple(value.shape)) for name, value in module.named_parameters()
        ),
        "buffers": tuple(
            (
                name,
                tuple(value.shape),
                value.is_floating_point(),
                str(value.dtype) if not value.is_floating_point() else None,
            )
            for name, value in module.named_buffers()
        ),
        "state_keys": tuple(module.state_dict()),
    }


def _contains_tensor(value: Any, seen: set[int]) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if id(value) in seen:
        return False
    seen.add(id(value))
    if isinstance(value, Mapping):
        return any(_contains_tensor(item, seen) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(item, seen) for item in value)
    return False


def validate_custom_transform(module: nn.Module, width: int) -> None:
    """Validate public methods, explicit capability and serializable state."""
    if not isinstance(module, nn.Module):
        raise TypeError("Transform overrides must be torch.nn.Module instances")
    registrations = tuple(module.named_modules(remove_duplicate=False))
    if len({id(child) for _, child in registrations}) != len(registrations):
        raise ValueError("Custom transform submodules require unique registrations")
    if type(module) is not nn.Identity and (
        not callable(getattr(module, "fit", None)) or not callable(getattr(module, "inverse", None))
    ):
        raise TypeError(
            "Custom transforms require public fit(training), forward and inverse methods"
        )
    if not _framewise(module):
        raise ValueError(
            "Custom preprocessing must declare supports_framewise=True and preserve signals; "
            "put temporal or feature-adding operations in the functional graph"
        )
    for child in module.modules():
        if isinstance(child, (BaseReservoirLayer, nn.RNNBase, nn.RNNCellBase)):
            raise ValueError("Custom preprocessing cannot contain hidden recurrent state")
        if any(
            _contains_tensor(value, set())
            for name, value in vars(child).items()
            if name not in ("_parameters", "_buffers", "_modules")
        ):
            raise ValueError(
                "All custom transform tensor state must be registered as buffers or parameters"
            )
        for parameter in child.parameters(recurse=False):
            if parameter.is_inference():
                raise ValueError(
                    "Custom transform Parameters must be constructed outside inference_mode; "
                    "fitting an ordinary transform inside inference_mode is supported"
                )
        with torch.inference_mode(False), torch.no_grad():
            for name, value in child.named_buffers(recurse=False):
                if value.is_inference():
                    setattr(child, name, value.detach().clone())
    for value in module.state_dict().values():
        if not isinstance(value, torch.Tensor):
            raise ValueError("Custom transform learned state must use registered Torch tensors")
        _validate_tensor_layout(value)
        if value.device.type == "meta" or not bool(torch.isfinite(value).all()):
            raise ValueError("Custom transform state must contain finite concrete tensors")
    if width < 1:
        raise ValueError("Custom transform feature width must be positive")


def validate_transform_ownership(graph: nn.Module, modules: tuple[nn.Module, ...]) -> None:
    """Reject ambiguous fitting/serialization when ports or graph share state."""
    seen_modules = {id(child) for child in graph.modules()}
    seen_tensors = {id(value) for value in _tensors(graph).values()}

    def storage(value: torch.Tensor) -> tuple[torch.device, int] | None:
        if value.numel() and value.layout == torch.strided and value.device.type != "meta":
            return value.device, value.untyped_storage().data_ptr()
        return None

    seen_storage = {
        key for value in _tensors(graph).values() if (key := storage(value)) is not None
    }
    for module in modules:
        children = {id(child) for child in module.modules()}
        tensors = {id(value) for value in _tensors(module).values()}
        if children & seen_modules or tensors & seen_tensors:
            raise ValueError(
                "Each transform port requires a distinct module and independent registered state"
            )
        for value in _tensors(module).values():
            _validate_tensor_layout(value)
            key = storage(value)
            if key is not None:
                if key in seen_storage:
                    raise ValueError(
                        "Transform registered tensors require independent storage and unique registrations"
                    )
                seen_storage.add(key)
        seen_modules.update(children)
        seen_tensors.update(tensors)


def apply_transform(
    module: nn.Module, value: torch.Tensor, *, inverse: bool = False
) -> torch.Tensor:
    """Check structural contracts without device-synchronizing inference scans."""
    output = (
        cast(Any, module).inverse(value)
        if inverse and type(module) is not nn.Identity
        else module(value)
    )
    if (
        not isinstance(output, torch.Tensor)
        or output.shape != value.shape
        or output.dtype != value.dtype
        or output.device != value.device
    ):
        raise ValueError(
            "Preprocessing forward/inverse must preserve exact shape, dtype and device"
        )
    return cast(torch.Tensor, output)


def transform_sample(training: torch.Tensor) -> torch.Tensor:
    """Three actual observations give bounded, checkpointable validation data."""
    flat = training.detach().reshape(-1, training.shape[-1])
    with torch.inference_mode(False), torch.no_grad():
        return torch.stack((flat[0], flat[len(flat) // 2], flat[-1])).unsqueeze(0).clone()


def _close(left: torch.Tensor, right: torch.Tensor) -> bool:
    # Validation occurs only at fit/load. Scale absolute tolerance to the data
    # and dtype, allowing a handful of rounding errors in inverse arithmetic.
    tolerance = 16 * torch.finfo(left.dtype).eps
    scale = right.abs().clamp_min(1)
    normalized = right / scale
    return bool(((left / scale - normalized).abs() <= tolerance * (1 + normalized.abs())).all())


def validate_transform_sample(module: nn.Module, sample: torch.Tensor) -> torch.Tensor:
    """Check inverse, frame splitting and pure eval behavior on a bounded sample."""
    if sample.ndim != 3 or sample.shape[:2] != (1, 3) or not sample.is_floating_point():
        raise ValueError("Invalid custom transform validation sample")
    if sample.device.type == "meta" or not bool(torch.isfinite(sample).all()):
        raise ValueError("Custom transform validation sample must be finite")
    before = {name: value.detach().clone() for name, value in _tensors(module).items()}

    def checked(value: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
        original = value.detach().clone()
        output = apply_transform(module, value, inverse=inverse)
        if not torch.equal(value, original):
            raise ValueError("Custom preprocessing must not modify input tensors")
        return output

    with evaluating(module), torch.no_grad():
        transformed = checked(sample).clone()
        restored = checked(transformed, inverse=True)
        repeated = checked(sample)
        frames = torch.cat([checked(frame) for frame in sample.split(1, dim=1)], 1)
    if (
        not bool(torch.isfinite(transformed).all())
        or not bool(torch.isfinite(restored).all())
        or not _close(restored, sample)
        or not _close(repeated, transformed)
        or not _close(frames, transformed)
    ):
        raise ValueError(
            "Custom preprocessing must be finite, reversible, framewise and pure in eval mode"
        )
    after = _tensors(module)
    if before.keys() != after.keys() or any(
        not torch.equal(value, after[name]) for name, value in before.items()
    ):
        raise ValueError("Custom preprocessing must not mutate registered state in eval mode")
    return transformed


def _state(module: nn.Module) -> dict[str, torch.Tensor]:
    with torch.inference_mode(False), torch.no_grad():
        return {name: value.detach().clone() for name, value in module.state_dict().items()}


def stage_transform(module: nn.Module, training: torch.Tensor) -> tuple[nn.Module, torch.Tensor]:
    """Fit a copy, then prove its learned behavior survives state-only loading."""
    schema = transform_schema(module, training.shape[-1])
    with torch.inference_mode(False), torch.enable_grad():
        candidate = copy.deepcopy(module)
        returned = (
            candidate
            if type(candidate) is nn.Identity
            else cast(Any, candidate).fit(training.detach().clone())
        )
        if returned is not candidate:
            raise ValueError("Custom transform fit(training) must return self")
        validate_custom_transform(candidate, training.shape[-1])
        if transform_schema(candidate, training.shape[-1]) != schema:
            raise ValueError(
                "Custom transform fit must preserve pre-registered state names and fixed shapes"
            )
        sample = transform_sample(training)
        expected = validate_transform_sample(candidate, sample)
        probe = copy.deepcopy(module)
        probe.load_state_dict(_state(candidate))
        try:
            actual = validate_transform_sample(probe, sample)
        except Exception as error:
            raise ValueError(
                "Custom fitted behavior must survive loading registered tensor state alone"
            ) from error
        if not _close(actual, expected):
            raise ValueError("Custom fitted behavior must depend only on registered tensor state")
    return probe, sample


def commit_transform_state(destination: nn.Module, source: nn.Module) -> None:
    """Copy prevalidated fixed tensor state without replacing user Parameters."""
    state = source.state_dict()
    destinations = _tensors(destination)
    with torch.inference_mode(False), torch.no_grad():
        for name, value in state.items():
            destinations[name].copy_(value)


def checkpoint_transform(
    module: nn.Module,
    state: dict[str, Any],
    sample: Any,
    width: int,
    *,
    fitted: bool,
    assign: bool,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Preflight complete custom state and its actual destination precision."""
    expected = module.state_dict()
    if state.keys() != expected.keys():
        raise ValueError("Custom transform checkpoint keys differ from the registered schema")
    if (
        not isinstance(sample, torch.Tensor)
        or sample.shape != (1, 3, width)
        or not sample.is_floating_point()
        or sample.device.type == "meta"
        or not bool(torch.isfinite(sample).all())
    ):
        raise ValueError("Invalid custom transform checkpoint validation sample")
    with torch.inference_mode(False), torch.no_grad():
        clean = {}
        for name, destination in expected.items():
            value = state[name]
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != destination.shape
                or value.is_floating_point() != destination.is_floating_point()
                or (not value.is_floating_point() and value.dtype != destination.dtype)
                or value.device.type == "meta"
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError("Invalid custom transform checkpoint tensor schema or values")
            clean[name] = (
                value.detach().clone() if assign else value.to(destination, copy=True).detach()
            )
            if not bool(torch.isfinite(clean[name]).all()):
                raise ValueError("Custom transform state overflows the destination dtype")
        probe = copy.deepcopy(module)
        probe.load_state_dict(clean, assign=assign)
        reference = next(
            (value for value in _tensors(probe).values() if value.is_floating_point()), sample
        )
        converted = sample.detach().clone() if assign else sample.to(reference, copy=True).detach()
        if fitted:
            validate_transform_sample(probe, converted)
    return clean, converted
