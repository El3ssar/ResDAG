"""One execution description shared by sequence and forecast inference.

Small adapters isolate third-party graph and generated-forward metadata. The
execution paths work with immutable operations and explicit value slots.
"""

from __future__ import annotations
import __future__

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from types import CodeType, MethodType
from typing import TYPE_CHECKING, Any, Protocol

import torch
import torch.nn as nn
from pytorch_symbolic.symbolic_model import SymbolicModel

from resdag._contracts import module_reference, validate_sequence_inputs
from resdag.layers.reservoirs import BaseReservoirLayer

if TYPE_CHECKING:
    from .model import ESNModel

ModelOutput = torch.Tensor | tuple[torch.Tensor, ...]
ReservoirStates = dict[str, torch.Tensor]


class ReservoirModelProtocol(Protocol):
    """Execution and state operations supplied by complete reservoir models."""

    def forward_stateless(
        self,
        *inputs: torch.Tensor,
        states: Mapping[str, torch.Tensor] | None = None,
        compile: bool = False,
    ) -> tuple[ModelOutput, ReservoirStates]: ...

    def reset_reservoirs(self) -> None: ...

    def get_state(self) -> ReservoirStates: ...

    def set_state(self, states: Mapping[str, torch.Tensor], strict: bool = True) -> None: ...


@dataclass(frozen=True)
class Operation:
    """A layer invocation with value slots and an optional recurrent-state slot."""

    layer: nn.Module
    name: str
    parents: tuple[int, ...]
    outputs: tuple[int, ...]
    reservoir_index: int | None = None


@dataclass(frozen=True)
class ExecutionPlan:
    """Immutable graph structure; parameter tensors remain owned by the model."""

    operations: tuple[Operation, ...]
    input_features: tuple[int, ...]
    output_slots: tuple[int, ...]
    reservoirs: tuple[BaseReservoirLayer, ...]
    reservoir_names: tuple[str, ...]
    n_values: int
    release_after: tuple[tuple[int, ...], ...]

    def run_stateful_sequence(self, inputs: tuple[torch.Tensor, ...]) -> ModelOutput:
        """Call each graph operation normally, releasing values at their last use.

        Unlike explicit-state inference, this path preserves reservoir,
        container and readout hooks and the modules' ordinary stored states.
        Batch fitting uses it under ``no_grad`` so discarded feature tensors
        do not remain reachable through an autograd graph.
        """
        self.validate_inputs(inputs)
        values: list[Any] = [None] * self.n_values
        values[: len(inputs)] = inputs
        for op, released in zip(self.operations, self.release_after):
            parents = tuple(values[index] for index in op.parents)
            if len(op.outputs) > 1:
                returned = op.layer(*parents[0])
                if len(returned) != len(op.outputs):
                    raise ValueError(
                        "A multi-output layer returned an unexpected number of outputs"
                    )
                for index, slot in enumerate(op.outputs):
                    values[slot] = returned[index]
            else:
                returned = op.layer(*parents)
                values[op.outputs[0]] = returned
            for slot in released:
                values[slot] = None
            del parents, returned
        outputs = tuple(values[index] for index in self.output_slots)
        return outputs[0] if len(outputs) == 1 else outputs

    def validate_reservoir_visibility(self) -> None:
        """Reject recurrent descendants added to ordinary nodes after plan caching."""
        for op in self.operations:
            if op.reservoir_index is not None:
                continue
            for child_name, child in op.layer.named_modules():
                if isinstance(child, BaseReservoirLayer):
                    reservoir_name = f"{op.name}.{child_name}"
                    raise ValueError(
                        f"Explicit-state execution cannot thread reservoir '{reservoir_name}' "
                        f"hidden inside graph operation '{op.name}' "
                        f"({type(op.layer).__name__}). Expose each reservoir as its own "
                        "functional graph node instead of placing it inside a container "
                        "or custom module. Ordinary stateful model execution remains available."
                    )

    def validate_inputs(self, inputs: tuple[torch.Tensor, ...]) -> None:
        """Check the model's sequence contract before graph execution."""
        reference = next(
            (ref for op in self.operations if (ref := module_reference(op.layer)) is not None),
            None,
        )
        validate_sequence_inputs(inputs, feature_sizes=self.input_features, reference=reference)

    def run_sequence(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: Mapping[str, torch.Tensor] | None = None,
        *,
        compile: bool = False,
    ) -> tuple[ModelOutput, ReservoirStates]:
        """Run a whole sequence while leaving stored reservoir states untouched."""
        self.validate_reservoir_visibility()
        self.validate_inputs(inputs)
        if states is not None:
            missing = set(self.reservoir_names) - set(states)
            extra = set(states) - set(self.reservoir_names)
            if missing or extra:
                raise ValueError(
                    f"Reservoir state keys must match the model; missing={sorted(missing)}, "
                    f"unexpected={sorted(extra)}"
                )
        values: list[Any] = [None] * self.n_values
        values[: len(inputs)] = inputs
        current: list[torch.Tensor | None] = [None] * len(self.reservoirs)
        if states is not None:
            current = [states[name] for name in self.reservoir_names]
        for op in self.operations:
            parents = tuple(values[i] for i in op.parents)
            if op.reservoir_index is not None:
                index = op.reservoir_index
                reservoir = self.reservoirs[index]
                feedback = parents[0]
                state = current[index]
                if state is None:
                    if states is not None:
                        raise TypeError(
                            f"State for '{self.reservoir_names[index]}' must be a torch.Tensor"
                        )
                    state = reservoir.cell.init_state(
                        feedback.shape[0], feedback.device, feedback.dtype
                    )
                else:
                    if not isinstance(state, torch.Tensor):
                        raise TypeError(
                            f"State for '{self.reservoir_names[index]}' must be a torch.Tensor"
                        )
                    reservoir.cell.validate_state(state)
                    if state.shape[0] != feedback.shape[0]:
                        raise ValueError(
                            f"State for '{self.reservoir_names[index]}' must match the input batch size"
                        )
                    if state.device != feedback.device or state.dtype != feedback.dtype:
                        raise ValueError(
                            f"State for '{self.reservoir_names[index]}' must match the input device "
                            "and dtype; move the explicit state together with the inputs"
                        )
                value, current[index] = reservoir.forward_stateless(
                    list(parents), state, compile=compile
                )
                values[op.outputs[0]] = value
            elif len(op.outputs) > 1:
                returned = op.layer(*parents[0])
                if len(returned) != len(op.outputs):
                    raise ValueError(
                        "A multi-output layer returned an unexpected number of outputs"
                    )
                for slot, value in zip(op.outputs, returned):
                    values[slot] = value
            else:
                values[op.outputs[0]] = op.layer(*parents)
        outputs = tuple(values[i] for i in self.output_slots)
        final: ReservoirStates = {}
        for name, state in zip(self.reservoir_names, current):
            assert state is not None
            final[name] = state
        result: ModelOutput = outputs[0] if len(outputs) == 1 else outputs
        return result, final


@contextmanager
def stateful_execution(model: nn.Module, plan: ExecutionPlan) -> Iterator[None]:
    """Use a fitting executor through the model's real ``nn.Module.__call__``.

    A temporary forward keeps model/global hooks in PyTorch's own dispatcher.
    Temporarily bypass ``Module.compile`` too: its cached callable must not
    execute an older forward instead of the fitting plan. Restore the exact
    prior instance attributes even when a pre-hook or fitting operation fails.
    Fitting the same mutable model recursively or concurrently is unsupported.
    """
    active = "_resdag_fitting_execution_active"
    if getattr(model, active, False):
        raise RuntimeError("Cannot recursively fit the same reservoir model")
    missing = object()
    previous = {
        name: model.__dict__.get(name, missing)
        for name in ("forward", "_compiled_call_impl", active)
    }

    def forward(module: nn.Module, *inputs: torch.Tensor) -> ModelOutput:
        return plan.run_stateful_sequence(inputs)

    model.forward = MethodType(forward, model)
    model.__dict__["_compiled_call_impl"] = None
    model.__dict__[active] = True
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is missing:
                model.__dict__.pop(name, None)
            else:
                model.__dict__[name] = value


@lru_cache(maxsize=128)
def _generated_forward_code(source: str) -> CodeType | None:
    """Inspect generated code without executing source or retaining a model."""
    flags = SymbolicModel.forward.__code__.co_flags & __future__.annotations.compiler_flag
    compiled = compile(source, "<string>", "exec", flags=flags, dont_inherit=True)
    return next(
        (
            value
            for value in compiled.co_consts
            if isinstance(value, CodeType) and value.co_name == "forward"
        ),
        None,
    )


def validate_fitting_forward(model: ESNModel) -> None:
    """Accept declared-DAG forwards, rejecting customized outer bodies early.

    Keep the symbolic library's generated-forward metadata adaptation here.
    Comparing code supports deepcopy and pickle reconstruction without relying
    on a method object's identity from the time a model was constructed.
    """
    forward = model.forward
    function = getattr(forward, "__func__", None)
    if getattr(forward, "__self__", None) is model:
        if function is SymbolicModel.forward:
            return
        if getattr(model, "_enable_forward_codegen", False):
            source = getattr(model, "_generated_forward_source", "")
            try:
                expected = _generated_forward_code(source)
            except (SyntaxError, ValueError, TypeError):
                expected = None
            if expected is not None and getattr(function, "__code__", None) == expected:
                return
    raise ValueError(
        "Single-pass fitting cannot bypass a custom top-level forward body. "
        "Put customization inside declared graph operations or ordinary module hooks, "
        "and keep the standard ESNModel forward."
    )


def _symbolic_operations(
    model: ESNModel,
) -> tuple[tuple[Any, nn.Module, tuple[Any, ...], tuple[Any, ...]], ...]:
    """Adapt the symbolic library's operation grouping in one place.

    Public ``parents`` and ``layer`` provide graph edges and layer objects.
    The library's private execution order and multi-output sibling grouping
    are necessary to preserve shared-layer invocation order and unpack nodes.
    Keep those compatibility details here rather than duplicating them in
    every backend.
    """
    operations = []
    for node in model._execution_order_nodes:
        layer = node.layer
        if not isinstance(layer, nn.Module):
            raise ValueError("A non-input symbolic graph node has no layer to execute")
        operations.append((node, layer, node.parents, tuple(node._layer_full_siblings)))
    return tuple(operations)


def _validate_unique_reservoir_invocations(model: ESNModel) -> None:
    """Keep adapter state ownership unambiguous without requiring explicit execution."""
    seen: set[int] = set()
    for _, layer, _, _ in _symbolic_operations(model):
        if isinstance(layer, BaseReservoirLayer):
            if id(layer) in seen:
                raise ValueError(
                    "The same reservoir layer instance is invoked more than once in the graph. "
                    "Sharing recurrent state this way is noncausal and makes results depend on "
                    "sequence chunk boundaries. Use distinct reservoir layer instances; "
                    "their parameter tensors may still share weights."
                )
            seen.add(id(layer))


def build_execution_plan(model: ESNModel) -> ExecutionPlan:
    """Describe an existing graph without running layers or changing model state."""
    _validate_unique_reservoir_invocations(model)
    input_nodes = tuple(model.inputs)
    slots = {id(node): index for index, node in enumerate(input_nodes)}
    names = {id(module): name for name, module in model.named_modules()}
    operations: list[Operation] = []
    reservoirs: list[BaseReservoirLayer] = []
    reservoir_names: list[str] = []
    reservoir_indices: dict[int, int] = {}
    for node, layer, parents, siblings in _symbolic_operations(model):
        output_nodes = siblings if len(siblings) > 1 else (node,)
        output_slots = []
        for output in output_nodes:
            slot = len(slots)
            slots[id(output)] = slot
            output_slots.append(slot)
        reservoir_index = None
        if isinstance(layer, BaseReservoirLayer):
            reservoir_indices[id(layer)] = len(reservoirs)
            reservoirs.append(layer)
            reservoir_names.append(names[id(layer)])
            reservoir_index = reservoir_indices[id(layer)]
        operations.append(
            Operation(
                layer=layer,
                name=names[id(layer)],
                parents=tuple(slots[id(parent)] for parent in parents),
                outputs=tuple(output_slots),
                reservoir_index=reservoir_index,
            )
        )
    model_output_slots = tuple(slots[id(node)] for node in model.outputs)
    retained = set(model_output_slots)
    last_use = {slot: index for index, op in enumerate(operations) for slot in op.parents}
    release_after: list[list[int]] = [[] for _ in operations]
    for slot, index in last_use.items():
        if slot not in retained:
            release_after[index].append(slot)
    for index, op in enumerate(operations):
        for slot in op.outputs:
            if slot not in last_use and slot not in retained:
                release_after[index].append(slot)
    return ExecutionPlan(
        operations=tuple(operations),
        input_features=tuple(int(node.shape[-1]) for node in input_nodes),
        output_slots=model_output_slots,
        reservoirs=tuple(reservoirs),
        reservoir_names=tuple(reservoir_names),
        n_values=len(slots),
        release_after=tuple(tuple(indices) for indices in release_after),
    )
