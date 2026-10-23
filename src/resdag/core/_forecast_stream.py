"""Private bounded forecast streams for tuning observers.

Closing a stream stops before the next chunk and leaves stored recurrent state
untouched. Only normal exhaustion commits evolved states. The ordinary public
forecast collector remains on its existing optimized path.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Generator, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch

from resdag.core._execution import ReservoirStates
from resdag.core._feedback import FeedbackPlan, FeedbackRoute
from resdag.core._flat_inference import rollout
from resdag.core._forecast import ForecastAlignment, positive_step_count, prepare_forecast_inputs
from resdag.core._forecast_capabilities import ForecastSequenceExecutor, validate_forecast_execution
from resdag.core._modes import evaluating
from resdag.core._workflow_transforms import apply_transform
from resdag.layers.reservoirs import BaseReservoirLayer

if TYPE_CHECKING:
    from resdag.core.model import ESNModel
    from resdag.core.workflow import Model, Streams

Outputs = tuple[torch.Tensor, ...]
Steps = Generator[tuple[int, Outputs, ReservoirStates], None, None]


def _boundaries(horizon: int, checkpoints: Sequence[int] | None) -> tuple[int, ...]:
    if checkpoints is None:
        return (horizon,)
    if isinstance(checkpoints, (str, bytes)) or not isinstance(checkpoints, Sequence):
        raise TypeError("checkpoints must be a sequence of increasing forecast sample counts")
    values = tuple(positive_step_count(value, "checkpoint") for value in checkpoints)
    if any(left >= right for left, right in zip((0, *values), values)) or any(
        value > horizon for value in values
    ):
        raise ValueError("checkpoints must increase strictly within the forecast horizon")
    return values if values and values[-1] == horizon else (*values, horizon)


def _ranges(
    horizon: int, chunk_size: int, checkpoints: tuple[int, ...]
) -> Iterator[tuple[int, int]]:
    start = 0
    for boundary in checkpoints:
        while start < boundary:
            stop = min(start + chunk_size, boundary, horizon)
            yield start, stop
            start = stop


@dataclass(frozen=True)
class ForecastChunk:
    """Aligned output samples; ``start`` includes any unscored origin preroll."""

    start: int
    outputs: Outputs
    names: tuple[str, ...] | None = None

    @property
    def stop(self) -> int:
        return self.start + self.outputs[0].shape[-2]

    @property
    def predictions(self) -> torch.Tensor | dict[str, torch.Tensor] | Outputs:
        if len(self.outputs) == 1:
            return self.outputs[0]
        return dict(zip(self.names, self.outputs)) if self.names is not None else self.outputs

    @property
    def named_predictions(self) -> dict[str, torch.Tensor]:
        names = self.names or tuple(
            "output" if len(self.outputs) == 1 else f"output_{index}"
            for index in range(len(self.outputs))
        )
        return dict(zip(names, self.outputs))


def _commit_states(graph: torch.nn.Module, states: ReservoirStates) -> None:
    layers = {
        name: module
        for name, module in graph.named_modules()
        if isinstance(module, BaseReservoirLayer)
    }
    if set(states) != set(layers):
        raise RuntimeError("Forecast stream returned an inconsistent reservoir state schema")
    staged = {}
    with torch.inference_mode(False), torch.no_grad():
        for name, layer in layers.items():
            layer.cell.validate_state(states[name])
            staged[name] = states[name].detach().clone()
        for name, layer in layers.items():
            layer.reset_state()
            layer.state = staged[name]


def _initial_states(model: ESNModel, inputs: Outputs, reset: bool) -> ReservoirStates | None:
    if reset:
        return None
    result = {}
    for name, layer in model.named_modules():
        if not isinstance(layer, BaseReservoirLayer):
            continue
        value = layer.state
        batch = inputs[0].shape[0]
        if value is None or value.shape[0] != batch:
            if value is not None and layer._state_user_set:
                raise RuntimeError("A state restored via set_state has a different batch size")
            value = layer.cell.init_state(batch, inputs[0].device, inputs[0].dtype)
        result[name] = value.clone()
    return result


def _flat_chunks(
    model: ESNModel,
    history: Outputs,
    drivers: Outputs,
    *,
    horizon: int,
    chunk_size: int,
    checkpoints: tuple[int, ...],
    compile: bool,
    initial_feedback: torch.Tensor | None = None,
    boundary: bool = True,
    reset: bool = True,
) -> Steps:
    # The flat plan preflight occurs before sequence hooks or any generation.
    flat = model._get_flat_step()
    warm, warm_states = model.forward_stateless(
        *history, states=_initial_states(model, history, reset)
    )
    outputs = warm if isinstance(warm, tuple) else (warm,)
    if outputs[0].shape[-1] != history[0].shape[-1]:
        raise ValueError("Forecast feedback output width must match the first input")
    names = {id(layer): name for name, layer in model.named_modules()}
    state_names = tuple(names[id(layer)] for layer in flat.reservoir_layers)
    states = [warm_states[name] for name in state_names]
    feedback = initial_feedback if initial_feedback is not None else outputs[0][:, -1:, :].clone()
    seeds = tuple(value[:, -1:, :].clone() for value in outputs) if boundary else ()
    templates = tuple(value.new_empty(value.shape[0], 0, value.shape[-1]) for value in outputs)
    del warm, outputs
    driver_offset = 0
    for start, stop in _ranges(horizon, chunk_size, checkpoints):
        size = stop - start
        seed = boundary and start == 0
        steps = size - int(seed)
        if steps:
            blocks, states = rollout(
                flat,
                feedback,
                tuple(value[:, driver_offset : driver_offset + steps] for value in drivers),
                states,
                steps,
                [value.shape[-1] for value in templates],
                no_grad=True,
                compile=compile,
                output_templates=templates,
            )
            feedback = blocks[0][:, -1:, :].clone()
            driver_offset += steps
        else:
            blocks = [value for value in templates]
        if seed:
            blocks = [torch.cat((value, block), dim=1) for value, block in zip(seeds, blocks)]
        yield start, tuple(blocks), dict(zip(state_names, states))


def _feedback_chunks(
    graph: torch.nn.Module,
    history: Outputs,
    drivers: Outputs,
    plan: FeedbackPlan,
    bridge: Any,
    *,
    horizon: int,
    chunk_size: int,
    checkpoints: tuple[int, ...],
    execution: Literal["fast", "history"],
    compile: bool,
    workflow: Any = None,
    owner: torch.nn.Module | None = None,
) -> Steps:
    executor = ForecastSequenceExecutor(
        graph, execution, compile=compile, workflow=workflow, owner=owner
    )
    warm, states = executor.start(history)
    batch, length = history[0].shape[:2]
    if len(warm) != len(plan.leads) or any(
        value.shape != (batch, length, width) for value, width in zip(warm, plan.output_widths)
    ):
        raise ValueError("Forecast graph outputs must preserve history layout and output widths")
    pending: list[deque[torch.Tensor]] = []
    queues: dict[int, deque[torch.Tensor]] = {}
    for index, (value, lead) in enumerate(zip(warm, plan.leads)):
        tail = value[:, length - lead : length - lead + min(lead, horizon)].clone()
        pending.append(deque(tail[:, tick : tick + 1] for tick in range(tail.shape[1])))
        delay = max(
            (route.delay for route in plan.routes if route.output_index == index), default=0
        )
        if delay:
            delayed = value[:, -delay:].clone()
            queues[index] = deque(
                (delayed[:, tick : tick + 1] for tick in range(delay)), maxlen=delay
            )
    del warm, value
    tick = 0
    for start, stop in _ranges(horizon, chunk_size, checkpoints):
        acc: list[list[torch.Tensor]] = [[] for _ in plan.leads]
        for _ in range(stop - start):
            while any(not values for values in pending):
                frames: list[torch.Tensor | None] = [None] * len(history)
                for index, value in zip(plan.driver_indices, drivers):
                    frames[index] = value[:, tick : tick + 1]
                for route in plan.routes:
                    frames[route.input_index] = bridge(
                        route, queues[route.output_index][-route.delay]
                    )
                assert all(value is not None for value in frames)
                outputs, states = executor.advance(
                    tuple(value for value in frames if value is not None)
                )
                for index, (output, lead) in enumerate(zip(outputs, plan.leads)):
                    if output.shape != (batch, 1, plan.output_widths[index]):
                        raise ValueError("Forecast graph outputs must preserve frame layout")
                    if index in queues:
                        queues[index].append(output.clone())
                    if tick + lead < horizon:
                        pending[index].append(output.clone())
                tick += 1
            for index, values in enumerate(pending):
                acc[index].append(values.popleft())
        yield start, tuple(torch.cat(values, dim=1) for values in acc), states


@contextmanager
def raw_forecast_stream(
    model: ESNModel,
    warmup_inputs: Outputs | torch.Tensor,
    forecast_inputs: Outputs | torch.Tensor | None = None,
    *,
    horizon: int,
    chunk_size: int = 64,
    checkpoints: Sequence[int] | None = None,
    compile: bool = False,
    initial_feedback: torch.Tensor | None = None,
    alignment: ForecastAlignment = "next",
    reset: bool = True,
) -> Iterator[Iterator[ForecastChunk]]:
    """Stream raw BTF outputs without allocating a full-horizon forecast."""
    chunk_size = positive_step_count(chunk_size, "chunk_size")
    history, drivers, horizon, boundary = prepare_forecast_inputs(
        warmup_inputs,
        forecast_inputs,
        num_inputs=len(model.inputs),
        horizon=horizon,
        initial_feedback=initial_feedback,
        alignment=alignment,
    )
    validate_forecast_execution(model, "fast", compile=compile)
    boundaries = _boundaries(horizon, checkpoints)

    def generate() -> Generator[ForecastChunk, None, None]:
        engine = _flat_chunks(
            model,
            history,
            drivers,
            horizon=horizon,
            chunk_size=chunk_size,
            checkpoints=boundaries,
            compile=compile,
            initial_feedback=initial_feedback,
            boundary=boundary,
            reset=reset,
        )
        final = None
        try:
            for start, outputs, states in engine:
                final = states
                yield ForecastChunk(start, outputs)
            if final is not None:
                _commit_states(model, final)
        finally:
            engine.close()

    iterator = generate()
    try:
        with evaluating(model), torch.no_grad():
            yield iterator
    finally:
        iterator.close()


@contextmanager
def forecast_stream(
    model: Model,
    history: Streams | None = None,
    *,
    horizon: int,
    future_inputs: Streams | None = None,
    chunk_size: int = 64,
    checkpoints: Sequence[int] | None = None,
    execution: Literal["fast", "history"] = "fast",
    compile: bool = False,
) -> Iterator[Iterator[ForecastChunk]]:
    """Stream prepared tensor ports in original units; clocks stay in the adapter."""
    from resdag.core.model import ESNModel

    horizon = positive_step_count(horizon, "horizon")
    chunk_size = positive_step_count(chunk_size, "chunk_size")
    boundaries = _boundaries(horizon, checkpoints)
    model._require_scaling()
    model._backend.validate_forecast(model.graph, compile=compile)
    validate_forecast_execution(model.graph, execution, compile=compile)
    from resdag.core._hooks import (
        eager_with_hooks,
        forecast_owner,
        has_execution_hooks,
        original_outputs,
    )

    owner = forecast_owner(model)
    hooked = has_execution_hooks(owner)
    if hooked:
        eager_with_hooks(compile)
        compile = False
    plan = model._feedback_plan()
    if any(not layer.is_fitted for _, layer in model._backend.fitted_readouts(model.graph)):
        raise RuntimeError("Forecast requires fitted terminal readouts; call fit(training) first")
    if history is None:
        if not model.is_fitted:
            raise RuntimeError("No cached history; provide history or call fit(training) first")
        history = {
            name: getattr(model, f"_history_{i}") for i, name in enumerate(model.input_names)
        }
    values, was_2d = model._streams(
        history, model.input_names, model._input_widths, "history", finite=True
    )
    if values[0].shape[1] < plan.required_history:
        raise ValueError(f"history needs at least {plan.required_history} samples")
    driver_names = tuple(model.input_names[index] for index in plan.driver_indices)
    driver_widths = tuple(model._input_widths[index] for index in plan.driver_indices)
    steps = plan.future_steps(horizon)
    if not driver_names:
        if future_inputs is not None:
            raise ValueError("This graph has no future driver inputs")
        drivers: Outputs = ()
    elif future_inputs is None:
        if steps:
            raise ValueError(f"future_inputs requires driver keys {list(driver_names)}")
        drivers = tuple(values[index][:, :0] for index in plan.driver_indices)
    else:
        drivers, _ = model._streams(
            future_inputs,
            driver_names,
            driver_widths,
            "future_inputs",
            finite=True,
            allow_empty=True,
        )
        if drivers[0].shape[0] != values[0].shape[0] or drivers[0].shape[1] < steps:
            raise ValueError("future_inputs must match history batch and required samples")

    def generate() -> Generator[ForecastChunk, None, None]:
        scaled = values if hooked else model._scaled_inputs(values)
        scaled_drivers = tuple(
            (
                apply_transform(model.input_scalers[index], value)
                if not hooked and steps and model._transform_active(output=False, index=index)
                else value
            )
            for index, value in zip(plan.driver_indices, drivers)
        )
        default_route = plan.routes == (FeedbackRoute(0, 0, 1),) and all(
            lead == 1 for lead in plan.leads
        )
        if hooked:
            engine = _feedback_chunks(
                model.graph,
                values,
                drivers,
                plan,
                lambda route, value: value,
                horizon=horizon,
                chunk_size=chunk_size,
                checkpoints=boundaries,
                execution=execution,
                compile=False,
                workflow=model,
                owner=owner,
            )
        elif (
            execution == "fast"
            and isinstance(model.graph, ESNModel)
            and default_route
            and model._feedback_scaling_matches()
        ):
            engine = _flat_chunks(
                model.graph,
                scaled,
                scaled_drivers,
                horizon=horizon,
                chunk_size=chunk_size,
                checkpoints=boundaries,
                compile=compile,
            )
        else:

            def bridge(route: FeedbackRoute, value: torch.Tensor) -> torch.Tensor:
                if model._transform_active(output=True, index=route.output_index):
                    value = apply_transform(
                        model.output_scalers[route.output_index], value, inverse=True
                    )
                if model._transform_active(output=False, index=route.input_index):
                    value = apply_transform(model.input_scalers[route.input_index], value)
                return value

            engine = _feedback_chunks(
                model.graph,
                scaled,
                scaled_drivers,
                plan,
                bridge,
                horizon=horizon,
                chunk_size=chunk_size,
                checkpoints=boundaries,
                execution=execution,
                compile=compile,
            )
        final = None
        try:
            for start, outputs, states in engine:
                final = states
                result = (
                    original_outputs(outputs, model.output_names, was_2d)
                    if hooked
                    else model._outputs(outputs if len(outputs) != 1 else outputs[0], was_2d)
                )
                restored = (
                    tuple(result[name] for name in model.output_names)
                    if isinstance(result, dict)
                    else (result,)
                )
                yield ForecastChunk(start, restored, model.output_names)
            if final is not None:
                model._commit_forecast_states(final)
        finally:
            engine.close()

    iterator = generate()
    try:
        with evaluating(owner), torch.no_grad():
            yield iterator
    finally:
        iterator.close()
