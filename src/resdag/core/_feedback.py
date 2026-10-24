"""Named feedback clocks and bounded, device-local forecast queues."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch

from resdag.core._execution import ReservoirStates
from resdag.core._forecast import positive_step_count


@dataclass(frozen=True)
class Feedback:
    """Connect a named output to a named input after `delay` graph steps.

    An output produced at graph step `q` is consumed at step `q + delay`.
    This delay is independent of the output's `prediction_lead`: a head with
    lead two estimates the physical sample two steps after its input. Matching
    lead and delay gives naturally aligned self-feedback; other combinations
    deliberately model lagged coupling. Widths must match at both ports.
    """

    output: str
    input: str
    delay: int = 1

    def __post_init__(self) -> None:
        for field in ("output", "input"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"Feedback {field} must be a non-empty port name")
            object.__setattr__(self, field, str(value))
        object.__setattr__(self, "delay", positive_step_count(self.delay, "feedback delay"))


def prediction_leads(
    value: int | Mapping[str, int], output_names: tuple[str, ...]
) -> tuple[int, ...]:
    """Canonicalize an optional per-head override without losing integer checks."""
    if isinstance(value, Mapping):
        if any(not isinstance(name, str) for name in value):
            raise ValueError("prediction_lead keys must be public output names")
        unknown = set(value) - set(output_names)
        if unknown:
            raise ValueError(f"prediction_lead has unknown output names: {sorted(unknown)}")
        return tuple(
            positive_step_count(value.get(name, 1), f"prediction_lead[{name!r}]")
            for name in output_names
        )
    lead = positive_step_count(value, "prediction_lead")
    return (lead,) * len(output_names)


def feedback_specs(value: Sequence[Feedback] | None) -> tuple[Feedback, ...] | None:
    """Freeze the user's routes while preserving the automatic-routing request."""
    if value is None:
        return None
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise TypeError("feedback must be a sequence of Feedback connections or None")
    if not all(isinstance(route, Feedback) for route in value):
        raise TypeError("feedback must contain Feedback connections")
    return tuple(value)


@dataclass(frozen=True)
class FeedbackRoute:
    output_index: int
    input_index: int
    delay: int


@dataclass(frozen=True)
class FeedbackPlan:
    routes: tuple[FeedbackRoute, ...]
    leads: tuple[int, ...]
    driver_indices: tuple[int, ...]
    output_widths: tuple[int, ...]

    @property
    def required_history(self) -> int:
        return max((*self.leads, *(route.delay for route in self.routes)))

    def future_steps(self, horizon: int) -> int:
        return max(0, horizon - min(self.leads))


def feedback_plan(
    specs: tuple[Feedback, ...] | None,
    leads: tuple[int, ...],
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    input_widths: tuple[int, ...],
    output_widths: tuple[int, ...],
    *,
    validate_widths: bool = True,
) -> FeedbackPlan:
    """Resolve public names before any execution, never by readout/module names."""
    resolved = (Feedback(output_names[0], input_names[0], leads[0]),) if specs is None else specs
    routes = []
    destinations: set[int] = set()
    for route in resolved:
        if route.input not in input_names:
            raise ValueError(
                f"Unknown feedback input {route.input!r}; expected {list(input_names)}"
            )
        if route.output not in output_names:
            raise ValueError(
                f"Unknown feedback output {route.output!r}; expected {list(output_names)}"
            )
        source, destination = output_names.index(route.output), input_names.index(route.input)
        if destination in destinations:
            raise ValueError(f"Input {route.input!r} has more than one feedback connection")
        if validate_widths and output_widths[source] != input_widths[destination]:
            raise ValueError(
                f"Feedback output {route.output!r} has width {output_widths[source]}, "
                f"but input {route.input!r} has width {input_widths[destination]}"
            )
        destinations.add(destination)
        routes.append(FeedbackRoute(source, destination, route.delay))
    return FeedbackPlan(
        tuple(routes),
        leads,
        tuple(index for index in range(len(input_names)) if index not in destinations),
        output_widths,
    )


class SequenceExecutor(Protocol):
    def start(
        self, inputs: tuple[torch.Tensor, ...]
    ) -> tuple[tuple[torch.Tensor, ...], ReservoirStates]: ...

    def advance(
        self, inputs: tuple[torch.Tensor, ...]
    ) -> tuple[tuple[torch.Tensor, ...], ReservoirStates]: ...


@torch.no_grad()
def feedback_rollout(
    executor: SequenceExecutor,
    history: tuple[torch.Tensor, ...],
    drivers: tuple[torch.Tensor, ...],
    plan: FeedbackPlan,
    bridge: Callable[[FeedbackRoute, torch.Tensor], torch.Tensor],
    *,
    horizon: int,
    return_warmup: bool,
) -> tuple[tuple[torch.Tensor, ...], ReservoirStates]:
    """Return heads on one physical clock while routing graph-clock emissions.

    Only delayed emission tails are retained in queues. All data remain Torch
    tensors on the input device; the Python clock never reads tensor values.
    The caller commits the final reservoir states only after successful rollout.
    """
    warm, states = executor.start(history)
    batch, length = history[0].shape[:2]
    if len(warm) != len(plan.leads) or any(
        value.shape != (batch, length, width) for value, width in zip(warm, plan.output_widths)
    ):
        raise ValueError(
            "Forecast graph outputs must preserve history batch/time and output widths"
        )
    max_lead = max(plan.leads)
    prefix = length - max_lead if return_warmup else 0
    predictions = []
    queues: dict[int, deque[torch.Tensor]] = {}
    for index, (value, lead) in enumerate(zip(warm, plan.leads)):
        result = value.new_empty(batch, prefix + horizon, value.shape[-1])
        if prefix:
            result[:, :prefix] = value[:, max_lead - lead : length - lead]
        seeded = min(lead, horizon)
        result[:, prefix : prefix + seeded] = value[:, length - lead : length - lead + seeded]
        predictions.append(result)
        delay = max(
            (route.delay for route in plan.routes if route.output_index == index), default=0
        )
        if delay:
            tail = value[:, -delay:].clone()
            queues[index] = deque((tail[:, tick : tick + 1] for tick in range(delay)), maxlen=delay)
    for tick in range(plan.future_steps(horizon)):
        frames: list[torch.Tensor | None] = [None] * len(history)
        for index, value in zip(plan.driver_indices, drivers):
            frames[index] = value[:, tick : tick + 1]
        for route in plan.routes:
            frames[route.input_index] = bridge(route, queues[route.output_index][-route.delay])
        assert all(frame is not None for frame in frames)
        outputs, states = executor.advance(tuple(frame for frame in frames if frame is not None))
        if len(outputs) != len(plan.leads):
            raise ValueError("Forecast graph output count differs from the declared heads")
        for index, (output, lead) in enumerate(zip(outputs, plan.leads)):
            if output.shape != (batch, 1, plan.output_widths[index]):
                raise ValueError("Forecast graph outputs must preserve each input frame's layout")
            if index in queues:
                queues[index].append(output)
            position = tick + lead
            if position < horizon:
                predictions[index][:, prefix + position : prefix + position + 1] = output
    return tuple(predictions), states
