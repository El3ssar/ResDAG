"""Private, unregistered dispatch for the existing reservoir model families.

Adapters hold no graph references or tensors. The original module remains the
workflow's sole registered graph, including its parameters, hooks and states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn

from resdag._contracts import module_reference
from resdag.core._execution import ReservoirModelProtocol
from resdag.core.model import ESNModel
from resdag.layers.readouts import ReadoutLayer
from resdag.layers.reservoirs import BaseReservoirLayer

if TYPE_CHECKING:
    from resdag.ensemble.coupled import CoupledEnsembleESNModel
    from resdag.models.parallel_esn import ParallelESNModel


@dataclass(frozen=True)
class BackendInterface:
    input_widths: tuple[int, ...]
    output_widths: tuple[int, ...]
    schema: dict[str, Any]
    references: tuple[object, ...]


def _symbolic_interface(graph: ESNModel) -> BackendInterface:
    return BackendInterface(
        tuple(int(node.shape[-1]) for node in graph.inputs),
        tuple(int(node.shape[-1]) for node in graph.outputs),
        {"kind": "symbolic"},
        (*graph.inputs, *graph.outputs),
    )


def _terminal_names(graph: ESNModel) -> tuple[str, ...]:
    graph._get_execution_plan(require_explicit_state=False)
    terminals = [node.layer for node in graph.outputs]
    readouts = [
        (path, layer) for path, layer in graph.named_modules() if isinstance(layer, ReadoutLayer)
    ]
    if (
        not all(isinstance(layer, ReadoutLayer) for layer in terminals)
        or len({id(layer) for layer in terminals}) != len(terminals)
        or {id(layer) for layer in terminals} != {id(layer) for _, layer in readouts}
    ):
        raise ValueError(
            "Model.fit currently requires one distinct terminal readout per output "
            "and no internal fitted readouts; use ESNTrainer for an explicit staged fit"
        )
    keys = {id(layer): layer.name or path for path, layer in readouts}
    names = tuple(keys[id(layer)] for layer in terminals)
    if len(set(names)) != len(names):
        raise ValueError("Terminal readouts must have distinct internal names for fitting")
    return names


class WorkflowBackend:
    """Stateless dispatch selected explicitly for a supported model family."""

    def __init__(self, kind: str) -> None:
        self.kind = kind

    def checkpoint_schema(self, graph: nn.Module, interface: BackendInterface) -> dict[str, Any]:
        if self.kind == "symbolic":
            return dict(interface.schema)
        # Record portable construction structure once, without reading tensor
        # values or including nonpersistent recurrent states/compiled caches.
        parameters = tuple(
            (name, tuple(value.shape))
            for name, value in graph.named_parameters(remove_duplicate=False)
        )
        buffers = tuple(
            (f"{path}.{name}" if path else name, tuple(value.shape))
            for path, module in graph.named_modules(remove_duplicate=False)
            for name, value in module._buffers.items()
            if value is not None and name not in module._non_persistent_buffers_set
        )
        return {**interface.schema, "parameters": parameters, "buffers": buffers}

    def describe(self, graph: nn.Module) -> BackendInterface:
        if self.kind == "symbolic":
            return _symbolic_interface(cast(ESNModel, graph))
        if self.kind == "parallel":
            parallel = cast("ParallelESNModel", graph)
            d, w, halo = parallel.input_dim, parallel.core_width, parallel.halo
            if (
                d <= 0
                or w <= 0
                or halo < 0
                or d % w
                or parallel.n_patches != d // w
                or tuple(parallel.patch_index.shape) != (d // w, w + 2 * halo)
                or parallel.reservoir.feedback_size != w + 2 * halo
                or parallel.readout.out_features != w
                or parallel.readout.in_features != parallel.reservoir.reservoir_size
            ):
                raise ValueError("Parallel graph patch geometry and layer widths are inconsistent")
            return BackendInterface(
                (d,),
                (d,),
                {"kind": "parallel", "input_dim": d, "core_width": w, "halo": halo},
                (parallel.reservoir, parallel.readout),
            )
        coupled = cast("CoupledEnsembleESNModel", graph)
        members = list(coupled._iter_models())
        if not members or not all(isinstance(member, ESNModel) for member in members):
            raise ValueError("Coupled graph requires non-empty ESNModel members")
        coupled._validate_feedback_output_dims(members)
        coupled._validate_member_interfaces(members)
        interfaces = tuple(_symbolic_interface(member) for member in members)
        reservoirs = [
            layer
            for member in members
            for layer in member.modules()
            if isinstance(layer, BaseReservoirLayer)
        ]
        if len({id(layer) for layer in reservoirs}) != len(reservoirs):
            raise ValueError("Coupled graph members must not share reservoir instances")
        aggregator = coupled.aggregator_module
        if aggregator is None:
            aggregate = coupled._aggregator_str
            if aggregate not in ("mean", "median"):
                raise ValueError("Coupled graph aggregator must be mean, median or an nn.Module")
            aggregate_schema: str = aggregate
        else:
            aggregate_schema = f"{type(aggregator).__module__}.{type(aggregator).__qualname__}"
        return BackendInterface(
            interfaces[0].input_widths,
            interfaces[0].output_widths,
            {
                "kind": "coupled",
                "members": tuple((item.input_widths, item.output_widths) for item in interfaces),
                "aggregator": aggregate_schema,
            },
            (
                *members,
                *(reference for item in interfaces for reference in item.references),
                *(() if aggregator is None else (aggregator,)),
            ),
        )

    def validate_fit(self, graph: nn.Module) -> None:
        if self.kind == "symbolic":
            _terminal_names(cast(ESNModel, graph))
        elif self.kind == "coupled":
            for member in cast("CoupledEnsembleESNModel", graph)._iter_models():
                _terminal_names(member)

    def fitted_readouts(self, graph: nn.Module) -> tuple[tuple[str, ReadoutLayer], ...]:
        """Heads owned by the algebraic fitting workflow, excluding aggregators."""
        if self.kind == "parallel":
            return (("readout", cast("ParallelESNModel", graph).readout),)
        if self.kind == "coupled":
            return tuple(
                (f"models.{index}.{path}", layer)
                for index, member in enumerate(
                    cast("CoupledEnsembleESNModel", graph)._iter_models()
                )
                for path, layer in member.named_modules()
                if isinstance(layer, ReadoutLayer)
            )
        return tuple(
            (path, layer)
            for path, layer in graph.named_modules()
            if isinstance(layer, ReadoutLayer)
        )

    def synchronize_loaded_buffers(self, graph: nn.Module) -> None:
        if self.kind == "parallel":
            parallel = cast("ParallelESNModel", graph)
            reference = module_reference(parallel.reservoir)
            assert reference is not None
            # assign=True changes persistent tensor placement without visiting
            # nonpersistent derived indices. Keep indexing on the loaded device.
            with torch.inference_mode(False):
                parallel.patch_index = parallel.patch_index.to(
                    device=reference.device,
                    dtype=torch.long,
                    copy=parallel.patch_index.is_inference(),
                )

    def fit(
        self,
        graph: nn.Module,
        warmup: tuple[torch.Tensor, ...],
        training: tuple[torch.Tensor, ...],
        targets: tuple[torch.Tensor, ...],
    ) -> None:
        if self.kind == "parallel":
            cast("ParallelESNModel", graph).fit(warmup[0], training[0], targets[0])
            return
        from resdag.training import ESNTrainer

        members = (
            (cast(ESNModel, graph),)
            if self.kind == "symbolic"
            else tuple(cast("CoupledEnsembleESNModel", graph)._iter_models())
        )
        for member in members:
            ESNTrainer(member).fit(
                warmup_inputs=warmup,
                train_inputs=training,
                targets=dict(zip(_terminal_names(member), targets)),
            )

    def validate_forecast(self, graph: nn.Module, *, compile: bool) -> None:
        if self.kind != "symbolic" and compile:
            raise ValueError(
                f"Compiled forecasting is not supported by the {self.kind} backend; "
                "use compile=False"
            )
        if self.kind == "symbolic":
            cast(ESNModel, graph)._get_execution_plan()
        elif self.kind == "coupled":
            self.validate_stateless(graph)

    def validate_stateless(self, graph: nn.Module) -> None:
        if self.kind == "coupled":
            aggregator = cast("CoupledEnsembleESNModel", graph).aggregator_module
            if aggregator is not None and any(
                isinstance(layer, BaseReservoirLayer) for layer in aggregator.modules()
            ):
                raise ValueError(
                    "Explicit-state execution and forecasting cannot thread reservoirs "
                    "hidden inside a coupled aggregator; use a feed-forward aggregator. "
                    "Ordinary stateful forward remains available."
                )
            for member in cast("CoupledEnsembleESNModel", graph)._iter_models():
                member._get_execution_plan()

    def forecast(
        self,
        graph: nn.Module,
        history: tuple[torch.Tensor, ...],
        drivers: tuple[torch.Tensor, ...] | None,
        *,
        horizon: int,
        return_warmup: bool,
        compile: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if self.kind == "symbolic":
            return cast(ESNModel, graph).forecast(
                history,
                drivers,
                horizon=horizon,
                return_warmup=return_warmup,
                compile=compile,
            )
        if self.kind == "parallel":
            return cast("ParallelESNModel", graph).forecast(
                history[0], horizon=horizon, return_warmup=return_warmup
            )
        result = cast("CoupledEnsembleESNModel", graph).forecast(
            history, drivers, horizon=horizon, return_warmup=return_warmup
        )
        return cast(torch.Tensor, result)


def workflow_backend(graph: nn.Module) -> WorkflowBackend:
    """Import other families only when an existing graph is explicitly wrapped."""
    if isinstance(graph, ESNModel):
        return WorkflowBackend("symbolic")
    # These modules import core builders; importing them while core itself is
    # initializing would create a cycle and burden ordinary symbolic imports.
    from resdag.models.parallel_esn import ParallelESNModel

    if isinstance(graph, ParallelESNModel):
        return WorkflowBackend("parallel")
    from resdag.ensemble.coupled import CoupledEnsembleESNModel

    if isinstance(graph, CoupledEnsembleESNModel):
        return WorkflowBackend("coupled")
    raise TypeError(
        "Model currently supports ESNModel, ParallelESNModel and CoupledEnsembleESNModel graphs"
    )


def recurrent_graph(graph: nn.Module) -> ReservoirModelProtocol:
    """Known adapters share the existing explicit-state protocol."""
    return cast(ReservoirModelProtocol, graph)
