"""One execution description shared by sequence and forecast inference.

Only ``_symbolic_operations`` reads third-party private graph metadata. The
rest of ResDAG works with immutable operations and explicit reservoir states.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import torch
import torch.nn as nn

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


def build_execution_plan(model: ESNModel) -> ExecutionPlan:
    """Describe an existing graph without running layers or changing model state."""
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
            if id(layer) in reservoir_indices:
                raise ValueError(
                    "The same reservoir layer instance is invoked more than once in the graph. "
                    "Sharing recurrent state this way is noncausal and makes results depend on "
                    "sequence chunk boundaries. Use distinct reservoir layer instances; "
                    "their parameter tensors may still share weights."
                )
            reservoir_indices[id(layer)] = len(reservoirs)
            reservoirs.append(layer)
            reservoir_names.append(names[id(layer)])
            reservoir_index = reservoir_indices[id(layer)]
        operations.append(
            Operation(
                layer=layer,
                parents=tuple(slots[id(parent)] for parent in parents),
                outputs=tuple(output_slots),
                reservoir_index=reservoir_index,
            )
        )
    return ExecutionPlan(
        operations=tuple(operations),
        input_features=tuple(int(node.shape[-1]) for node in input_nodes),
        output_slots=tuple(slots[id(node)] for node in model.outputs),
        reservoirs=tuple(reservoirs),
        reservoir_names=tuple(reservoir_names),
        n_values=len(slots),
    )
