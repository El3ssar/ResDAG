"""Optional planning estimates and staged screening for model tuning.

These helpers never inspect or change process limits, allocator settings or
model precision. Byte estimates guide admission and concurrency; they are not
operating-system, resident-memory or accelerator-allocation limits.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, cast

import torch

from resdag._contracts import module_reference
from resdag.layers.readouts import ReadoutLayer
from resdag.layers.reservoirs import BaseReservoirLayer

if TYPE_CHECKING:
    from resdag.core.workflow import Model

    from ._tuning_data import TuningData

__all__ = [
    "Screening",
    "MemoryBudget",
    "MemoryEstimate",
    "MemoryContext",
    "memory_context",
    "raw_memory_context",
    "estimate_memory",
]


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        adjective = "positive" if minimum else "nonnegative"
        raise ValueError(f"{name} must be a {adjective} integer")
    return int(value)


class _FrozenDict(dict[str, Any]):
    """A small JSON/pickle-compatible immutable snapshot of metadata."""

    def _immutable(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("Resource metadata is immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _immutable
    __ior__ = _immutable

    def __reduce__(self) -> tuple[Any, tuple[dict[str, Any]]]:
        return type(self), (dict(self),)

    def __deepcopy__(self, memo: dict[int, Any]) -> _FrozenDict:
        return self


def _metadata(value: Any) -> Any:
    """Keep only immutable values, never caller tensors, arrays or modules."""
    if value is None or isinstance(value, (str, bool, int, float, torch.dtype, torch.device)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Resource metadata mapping keys must be strings")
        return _FrozenDict({key: _metadata(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_metadata(item) for item in value)
    if isinstance(value, torch.Tensor):
        return ("tensor", tuple(value.shape), str(value.dtype), str(value.device))
    if callable(value):
        return (
            "callable",
            getattr(value, "__module__", type(value).__module__),
            getattr(value, "__qualname__", type(value).__qualname__),
        )
    return ("object", type(value).__module__, type(value).__qualname__)


@dataclass(frozen=True)
class Screening:
    """Evaluate a cohort cheaply, then reevaluate its best fraction in full.

    Screening is opt-in. The default ``None`` uses all configured seeds and
    origins for every candidate. With balanced tuning, this default screen
    checks all 24 candidates on one seed/origin, then promotes the best eight.
    ``cohort_size=None`` uses the entire new-trial budget as one cohort.
    """

    seed_count: int = 1
    origin_count: int = 1
    keep_fraction: float = 1 / 3
    cohort_size: int | None = None

    def __post_init__(self) -> None:
        for name in ("seed_count", "origin_count"):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum=1))
        if (
            isinstance(self.keep_fraction, bool)
            or not isinstance(self.keep_fraction, Real)
            or not math.isfinite(self.keep_fraction)
            or not 0 < self.keep_fraction <= 1
        ):
            raise ValueError("keep_fraction must be finite and greater than zero, at most one")
        object.__setattr__(self, "keep_fraction", float(self.keep_fraction))
        if self.cohort_size is not None:
            object.__setattr__(
                self, "cohort_size", _integer(self.cohort_size, "cohort_size", minimum=1)
            )

    def validate(self, full_seed_count: int, full_origin_count: int) -> None:
        """Reject a screen that asks for more repetitions than full evaluation."""
        seeds = _integer(full_seed_count, "full_seed_count", minimum=1)
        origins = _integer(full_origin_count, "full_origin_count", minimum=1)
        if self.seed_count > seeds or self.origin_count > origins:
            raise ValueError("Screening seed/origin counts cannot exceed full evaluation counts")

    def promoted_count(self, n: int) -> int:
        """Round promotion upward; a nonempty cohort always retains a candidate."""
        n = _integer(n, "cohort size")
        numerator, denominator = Decimal(str(self.keep_fraction)).as_integer_ratio()
        return min(n, (n * numerator + denominator - 1) // denominator)


_UNITS = {"B": 1}
_UNITS.update({f"{prefix}B": 1000**index for index, prefix in enumerate("KMGT", 1)})
_UNITS.update({f"{prefix}iB": 1024**index for index, prefix in enumerate("KMGT", 1)})
_BYTES = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(B|KB|MB|GB|TB|KiB|MiB|GiB|TiB)")


def _bytes(value: int | str, name: str, *, zero: bool = False) -> int:
    if isinstance(value, str):
        match = _BYTES.fullmatch(value.strip())
        if match is None:
            raise ValueError(f"{name} must be integer bytes or a size such as '8GB' or '8GiB'")
        numerator, denominator = Decimal(match[1]).as_integer_ratio()
        count, remainder = divmod(numerator * _UNITS[match[2]], denominator)
        if remainder:
            raise ValueError(f"{name} must resolve to a whole number of bytes")
        value = count
    return _integer(value, name, minimum=0 if zero else 1)


@dataclass(frozen=True)
class MemoryEstimate:
    """Estimated peak bytes for one candidate, including its final full refit.

    ``host_bytes`` and ``device_bytes`` are separate planning accounts. A custom
    estimator owns their accuracy. Details are detached immutable metadata.
    """

    host_bytes: int
    device_bytes: int = 0
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("host_bytes", "device_bytes"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if not isinstance(self.details, Mapping):
            raise TypeError("MemoryEstimate.details must be a mapping")
        object.__setattr__(self, "details", _metadata(self.details))

    @property
    def total_bytes(self) -> int:
        """Host plus device planning bytes, before per-worker overhead."""
        return self.host_bytes + self.device_bytes


@dataclass(frozen=True)
class MemoryBudget:
    """Admission budget for estimated host/device peaks and worker overhead.

    ``limit`` bounds their sum. ``device_limit`` adds a device-only bound.
    Decimal GB and binary GiB are distinct. Values are parsed to integer bytes
    once. This never sets an OS, RSS, process or accelerator allocator limit.
    """

    limit: int | str
    device_limit: int | str | None = None
    worker_overhead: int | str = "256MiB"

    def __post_init__(self) -> None:
        object.__setattr__(self, "limit", _bytes(self.limit, "limit"))
        if self.device_limit is not None:
            object.__setattr__(self, "device_limit", _bytes(self.device_limit, "device_limit"))
        object.__setattr__(
            self, "worker_overhead", _bytes(self.worker_overhead, "worker_overhead", zero=True)
        )

    def fits(self, estimate: MemoryEstimate, current: Sequence[MemoryEstimate] = ()) -> bool:
        """Whether adding a candidate fits alongside all active estimates.

        Each candidate, including the first, adds ``worker_overhead`` host
        bytes. Call with an empty ``current`` to reject an oversized candidate
        before construction; otherwise wait/reduce concurrency until it fits.
        """
        estimates = (*current, estimate)
        if any(not isinstance(item, MemoryEstimate) for item in estimates):
            raise TypeError("fits expects MemoryEstimate instances")
        host = sum(item.host_bytes for item in estimates)
        device = sum(item.device_bytes for item in estimates)
        total = host + device + len(estimates) * cast(int, self.worker_overhead)
        return total <= cast(int, self.limit) and (
            self.device_limit is None or device <= cast(int, self.device_limit)
        )

    @property
    def limit_bytes(self) -> int:
        """The parsed combined limit as integer bytes."""
        return cast(int, self.limit)

    @property
    def device_limit_bytes(self) -> int | None:
        """The optional parsed accelerator-only limit."""
        return cast(int | None, self.device_limit)

    @property
    def worker_overhead_bytes(self) -> int:
        """Host overhead reserved per admitted candidate."""
        return cast(int, self.worker_overhead)


@dataclass(frozen=True)
class MemoryContext:
    """Immutable shapes/storage facts for resource estimation; no live objects.

    Sample counts cover complete observation prefixes, including washout. The
    final refit is therefore included even when validation fits are shorter.
    Storage counts deduplicate aliases within the dataset and within the model.
    """

    input_shapes: Mapping[str, tuple[int, ...]]
    target_shapes: Mapping[str, tuple[int, ...]]
    dtype: torch.dtype
    device: torch.device
    train_samples: int
    refit_samples: int
    dataset_host_bytes: int
    dataset_device_bytes: int
    model_host_bytes: int
    model_device_bytes: int
    readout_features: tuple[int, ...]
    readout_outputs: tuple[int, ...]
    activation_width: int
    max_horizon: int
    preset_options: Mapping[str, Any] = field(default_factory=dict)
    retained_forecast_bytes: int = 0
    replay_history_bytes: int = 0
    forecast_samples: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.dtype, torch.dtype) or not self.dtype.is_floating_point:
            raise ValueError("MemoryContext.dtype must be a real floating Torch dtype")
        object.__setattr__(self, "device", torch.device(self.device))
        for name in ("input_shapes", "target_shapes"):
            mapping = getattr(self, name)
            if not isinstance(mapping, Mapping) or not mapping:
                raise ValueError(f"{name} must be a nonempty named shape mapping")
            shapes = {}
            for key, shape in mapping.items():
                if len(shape) not in (2, 3):
                    raise ValueError(
                        "Memory shapes must be (time, features) or (batch, time, features)"
                    )
                shapes[key] = tuple(_integer(n, "shape dimension", minimum=1) for n in shape)
            object.__setattr__(self, name, _metadata(shapes))
        for name in ("train_samples", "refit_samples", "activation_width", "max_horizon"):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum=1))
        for name in (
            "dataset_host_bytes",
            "dataset_device_bytes",
            "model_host_bytes",
            "model_device_bytes",
            "retained_forecast_bytes",
            "replay_history_bytes",
            "forecast_samples",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        for name in ("readout_features", "readout_outputs"):
            object.__setattr__(
                self, name, tuple(_integer(n, name, minimum=1) for n in getattr(self, name))
            )
        if len(self.readout_features) != len(self.readout_outputs):
            raise ValueError("Readout feature/output widths must be paired")
        if not isinstance(self.preset_options, Mapping):
            raise TypeError("preset_options must be a mapping")
        object.__setattr__(self, "preset_options", _metadata(self.preset_options))

    @property
    def batch_size(self) -> int:
        shape = next(iter(self.input_shapes.values()))
        return shape[0] if len(shape) == 3 else 1

    @property
    def input_features(self) -> int:
        return sum(shape[-1] for shape in self.input_shapes.values())

    @property
    def output_features(self) -> int:
        return sum(shape[-1] for shape in self.target_shapes.values())


def _storage_parts(value: torch.Tensor) -> tuple[torch.Tensor, ...]:
    if value.layout == torch.sparse_coo:
        return value._indices(), value._values()
    if value.layout in (torch.sparse_csr, torch.sparse_bsr):
        return value.crow_indices(), value.col_indices(), value.values()
    if value.layout in (torch.sparse_csc, torch.sparse_bsc):
        return value.ccol_indices(), value.row_indices(), value.values()
    return (value,)


def _storage_bytes(values: Sequence[torch.Tensor]) -> tuple[int, int]:
    seen = set()
    host = device = 0
    for value in values:
        for part in _storage_parts(value):
            storage = part.untyped_storage()
            key = (str(part.device), storage._cdata)
            if key in seen:
                continue
            seen.add(key)
            if part.device.type == "cpu":
                host += storage.nbytes()
            else:
                device += storage.nbytes()
    return host, device


def memory_context(
    workflow: Model, data: TuningData, *, preset_options: Mapping[str, Any] | None = None
) -> MemoryContext:
    """Read allocation metadata without fitting, cloning or executing the model.

    CPU and accelerator buffers count, including nonpersistent recurrent state.
    Dataset histories/targets/views sharing backing storage count once. Python
    objects, backend workspaces and custom-operation scratch are not measured.
    """
    reference = module_reference(workflow.graph)
    first = next(iter(data.refit.inputs.values()))
    dtype, device = (
        (first.dtype, first.device) if reference is None else (reference.dtype, reference.device)
    )
    values = [
        *data.train.inputs.values(),
        *data.train.targets.values(),
        *data.refit.inputs.values(),
        *data.refit.targets.values(),
        *data.scales.values(),
    ]
    for origin in data.origins:
        values.extend((*origin.history.values(), *origin.truth.values(), origin.times))
        if origin.future_inputs is not None:
            values.extend(origin.future_inputs.values())
    dataset_host, dataset_device = _storage_bytes(values)
    model_host, model_device = _storage_bytes([*workflow.parameters(), *workflow.buffers()])
    heads = [module for module in workflow.graph.modules() if isinstance(module, ReadoutLayer)]
    widths = sum(workflow._input_widths) + sum(
        head.in_features + head.out_features for head in heads
    )
    for module in workflow.graph.modules():
        if isinstance(module, BaseReservoirLayer):
            widths += module.cell.output_size + module.cell.state_size
        elif isinstance(module, torch.nn.Linear) and not isinstance(module, ReadoutLayer):
            widths += module.out_features
    return MemoryContext(
        input_shapes={name: tuple(value.shape) for name, value in data.refit.inputs.items()},
        target_shapes={name: tuple(value.shape) for name, value in data.refit.targets.items()},
        dtype=dtype,
        device=device,
        train_samples=max(math.prod(value.shape[:-1]) for value in data.train.inputs.values()),
        refit_samples=max(math.prod(value.shape[:-1]) for value in data.refit.inputs.values()),
        dataset_host_bytes=dataset_host,
        dataset_device_bytes=dataset_device,
        model_host_bytes=model_host,
        model_device_bytes=model_device,
        readout_features=tuple(head.in_features for head in heads),
        readout_outputs=tuple(head.out_features for head in heads),
        activation_width=max(1, widths),
        max_horizon=max(origin.horizon for origin in data.origins),
        forecast_samples=max(
            math.prod(value.shape[:-1])
            for origin in data.origins
            for value in origin.history.values()
        ),
        preset_options=preset_options or {},
    )


def raw_memory_context(
    prepared: Mapping[str, Any],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    horizon: int | None = None,
) -> MemoryContext:
    """Describe prepared raw-runner tensors before calling an arbitrary factory.

    Readout/model sizes are unknown and require a custom estimator. Complete
    prediction prefixes needed by arbitrary raw losses are reserved explicitly.
    Storage includes every tensor entry and known device-transfer copies; no
    copies, allocations on an accelerator or model construction occur here.
    """
    tensors = {name: value for name, value in prepared.items() if isinstance(value, torch.Tensor)}
    for name in ("train", "target", "val"):
        value = tensors.get(name)
        if value is None or value.ndim != 3 or not value.is_floating_point():
            raise ValueError(f"Raw memory context requires floating (batch,time,features) {name}")
    train, target, validation = (tensors[name] for name in ("train", "target", "val"))
    destination = train.device if device is None else torch.device(device)
    working_dtype = train.dtype if dtype is None else dtype
    if not isinstance(working_dtype, torch.dtype) or not working_dtype.is_floating_point:
        raise ValueError("dtype must be a real floating Torch dtype")
    length = _integer(validation.shape[-2] if horizon is None else horizon, "horizon", minimum=1)
    shapes = {}
    for name, value in tensors.items():
        if name != "train" and not name.startswith("train_"):
            continue
        if value.ndim != 3:
            raise ValueError("Raw training/driver tensors must have three dimensions")
        suffix = name.removeprefix("train")
        warmup = tensors.get("warmup" + suffix)
        count = value.shape[-2] + (0 if warmup is None else warmup.shape[-2])
        shapes[name] = (value.shape[0], count, value.shape[-1])
    host, accelerator = _storage_bytes(list(tensors.values()))
    transfers = sum(
        value.numel() * value.element_size()
        for value in tensors.values()
        if value.device != destination
    )
    if destination.type == "cpu":
        host += transfers
    else:
        accelerator += transfers
    samples = max(math.prod(shape[:-1]) for shape in shapes.values())
    prefix = train.shape[0] * length * validation.shape[-1] * (torch.finfo(working_dtype).bits // 8)
    return MemoryContext(
        input_shapes=shapes,
        target_shapes={"target": tuple(target.shape)},
        dtype=working_dtype,
        device=destination,
        train_samples=samples,
        refit_samples=samples,
        dataset_host_bytes=host,
        dataset_device_bytes=accelerator,
        model_host_bytes=0,
        model_device_bytes=0,
        readout_features=(),
        readout_outputs=(),
        activation_width=sum(shape[-1] for shape in shapes.values()) + validation.shape[-1],
        max_horizon=length,
        retained_forecast_bytes=prefix,
    )


def estimate_memory(
    params: Mapping[str, Any],
    context: MemoryContext,
    *,
    custom_factory: bool = False,
    estimator: Callable[[dict[str, Any], MemoryContext], MemoryEstimate | int] | None = None,
) -> MemoryEstimate:
    """Estimate peak fit/forecast/refit bytes, before candidate construction.

    Automatic planning reserves dense preset weights, retained/staged model
    copies, full-refit feature traces and 8-byte QR/SVD/Gram workspaces. On an
    accelerator it also reserves a CPU solve bridge. This deliberately favors
    overestimation; custom layers and factories should provide their own
    estimator. A custom result is a complete estimate, not an added surcharge;
    it cannot remove known resident dataset/model or retained-prefix storage.
    """
    if not isinstance(params, Mapping) or not isinstance(context, MemoryContext):
        raise TypeError("estimate_memory expects parameter mapping and MemoryContext")
    if estimator is not None:
        if not callable(estimator):
            raise TypeError("memory estimator must be callable")
        result = estimator(copy.deepcopy(dict(params)), context)
        if not isinstance(result, MemoryEstimate):
            result = MemoryEstimate(_integer(result, "custom memory estimate"))
        host = context.dataset_host_bytes + context.model_host_bytes
        device = context.dataset_device_bytes + context.model_device_bytes
        retained = context.retained_forecast_bytes + context.replay_history_bytes
        if context.device.type == "cpu":
            host += retained
        else:
            device += retained
        return MemoryEstimate(
            max(result.host_bytes, host),
            max(result.device_bytes, device),
            {
                **result.details,
                "requested_host_bytes": result.host_bytes,
                "requested_device_bytes": result.device_bytes,
                "minimum_host_bytes": host,
                "minimum_device_bytes": device,
                "retained_forecast_bytes": context.retained_forecast_bytes,
                "replay_history_bytes": context.replay_history_bytes,
                "floor_applied": result.host_bytes < host or result.device_bytes < device,
            },
        )
    if custom_factory:
        raise ValueError(
            "A custom factory with a memory budget requires memory_estimator(params, context)"
        )
    if not context.readout_features:
        raise ValueError("Unknown readout widths require a custom memory estimator")
    element = torch.finfo(context.dtype).bits // 8
    n = max(context.train_samples, context.refit_samples)
    features, outputs = context.readout_features, context.readout_outputs
    model = context.model_host_bytes + context.model_device_bytes
    activation_width = context.activation_width
    if context.preset_options:
        options = {**context.preset_options, **params}
        width = _integer(options.get("reservoir_size", 500), "reservoir_size", minimum=1)
        inputs, output = context.input_features, context.output_features
        features, outputs = (width + inputs,), (output,)
        # Dense recurrence, feedback/input projection, vector bias/leak/state
        # and readout; no reservoir, initializer or layer is constructed here.
        model = max(
            model, element * (width * width + width * (inputs + 3) + output * (width + inputs + 1))
        )
        activation_width = width * 3 + inputs + output
    resident_host = context.dataset_host_bytes + 3 * context.model_host_bytes
    resident_device = context.dataset_device_bytes + 3 * context.model_device_bytes
    model_copies = 3 * model
    fit_data = 4 * n * (context.input_features + context.output_features) * element
    trace_samples = max(n, context.forecast_samples)
    traces = 4 * trace_samples * activation_width * element
    forecasts = 3 * max(
        context.retained_forecast_bytes,
        context.batch_size * context.output_features * element,
    )
    # QR augmentation and compact SVD, plus centered/cast inputs, inverse-norm
    # guards and factorization work arrays. Sum heads to cover retained features.
    workspace = sum(
        8 * (6 * n * (f + out) + 12 * f * f + 6 * min(n, f) ** 2 + 4 * f * out)
        for f, out in zip(features, outputs)
    )
    working = (
        model_copies + fit_data + traces + forecasts + context.replay_history_bytes + workspace
    )
    host, device = resident_host, resident_device
    bridge = 0
    if context.device.type == "cpu":
        host += working
    else:
        device += working
        bridge = workspace
        host += bridge
    return MemoryEstimate(
        host,
        device,
        {
            "kind": "planning estimate; not a hard cap",
            "samples": n,
            "element_bytes": element,
            "model_copies": model_copies,
            "fit_data": fit_data,
            "feature_traces": traces,
            "trace_samples": trace_samples,
            "forecast_outputs": forecasts,
            "replay_history": context.replay_history_bytes,
            "solve_workspace": workspace,
            "cpu_bridge": bridge,
        },
    )
