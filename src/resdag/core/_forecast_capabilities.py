"""Forecast capabilities and complete-prefix execution for custom sequence nodes.

Ordinary graph execution accepts arbitrary modules. Fast forecasting additionally
requires time-separable operations, or explicit ``BaseReservoirLayer`` state.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import CodeType, FunctionType
from typing import Any, Literal, cast

import torch
from pytorch_symbolic import useful_layers
from pytorch_symbolic.symbolic_data import SymbolicTensor
from torch import nn

from resdag._contracts import validate_sequence_inputs
from resdag.core._execution import ExecutionPlan, ReservoirModelProtocol, ReservoirStates
from resdag.layers.readouts import IncrementalRidgeReadoutLayer, ReadoutLayer
from resdag.layers.reservoirs import BaseReservoirLayer
from resdag.layers.transforms import (
    NLAT2,
    NLAT3,
    Concatenate,
    ExtendedSquare,
    FeaturePartitioner,
    Pad,
    PartialSquare,
    Power,
    SelectiveDropout,
    SelectiveExponentiation,
    Standardize,
)

ForecastExecution = Literal["fast", "history"]

_POINTWISE_TYPES = frozenset(
    (
        nn.Identity,
        nn.Linear,
        nn.Bilinear,
        nn.Tanh,
        nn.Sigmoid,
        nn.ReLU,
        nn.ReLU6,
        nn.LeakyReLU,
        nn.ELU,
        nn.CELU,
        nn.SELU,
        nn.GELU,
        nn.Hardsigmoid,
        nn.Hardswish,
        nn.Hardtanh,
        nn.SiLU,
        nn.Mish,
        nn.Softplus,
        nn.Softsign,
        nn.Tanhshrink,
        nn.LogSigmoid,
        nn.Threshold,
        nn.Hardshrink,
        nn.Softshrink,
        nn.Dropout,
        nn.AlphaDropout,
        nn.FeatureAlphaDropout,
        Concatenate,
        ExtendedSquare,
        FeaturePartitioner,
        NLAT2,
        NLAT3,
        Pad,
        PartialSquare,
        Power,
        SelectiveDropout,
        SelectiveExponentiation,
        Standardize,
        useful_layers.AddOpLayer,
        useful_layers.SubOpLayer,
        useful_layers.MulOpLayer,
        useful_layers.ModOpLayer,
        useful_layers.UnpackLayer,
    )
)

# The symbolic package implements ordinary scalar arithmetic with generic lambda
# modules. Recognize only the code objects created by its verified elementwise
# operators, never the LambdaOpLayer class or arbitrary user functions wholesale.
_SCALAR_OPERATORS = (
    "__abs__",
    "__neg__",
    "__add__",
    "__mul__",
    "__mod__",
    "__rmod__",
    "__pow__",
    "__rpow__",
    "__sub__",
    "__rsub__",
    "__truediv__",
    "__rtruediv__",
)
_SCALAR_CODES = frozenset(
    constant
    for name in _SCALAR_OPERATORS
    for constant in getattr(SymbolicTensor, name).__code__.co_consts
    if isinstance(constant, CodeType)
)
_SCALAR_GLOBALS = SymbolicTensor.__add__.__globals__


def _declared_framewise(layer: nn.Module) -> bool | None:
    if "supports_framewise" in vars(layer):
        return vars(layer)["supports_framewise"] is True
    for cls in type(layer).__mro__:
        if "supports_framewise" in vars(cls):
            # An overridden forward must renew the declaration; inheriting the
            # unchanged implementation also inherits its capability contract.
            return (
                vars(cls)["supports_framewise"] is True
                and "forward" not in vars(layer)
                and type(layer).forward is getattr(cls, "forward", None)
            )
    return None


def _feature_constant(value: Any) -> bool:
    if type(value) in (bool, int, float, complex):
        return True
    # Tensor/scalar subclasses may override arithmetic with temporal behavior.
    return type(value) in (torch.Tensor, nn.Parameter) and (
        value.ndim <= 1 or (value.ndim <= 3 and value.shape[-2] == 1)
    )


def _scalar_lambda(layer: nn.Module) -> bool:
    function = layer.forward
    known = (
        type(layer) is useful_layers.LambdaOpLayer
        and isinstance(function, FunctionType)
        and function.__code__ in _SCALAR_CODES
        and function.__globals__ is _SCALAR_GLOBALS
        and not function.__defaults__
        and not function.__kwdefaults__
    )
    if not known:
        return False
    assert isinstance(function, FunctionType)
    try:
        return all(_feature_constant(cell.cell_contents) for cell in function.__closure__ or ())
    except ValueError:
        return False  # An empty/mutated closure cannot certify a broadcast constant.


def _feature_slice(index: Any) -> bool:
    indices = index if isinstance(index, tuple) else (index,)
    if sum(item is Ellipsis for item in indices) > 1:
        return False
    expanded: list[Any] = []
    for item in indices:
        if item is Ellipsis:
            expanded.extend([slice(None)] * (4 - len(indices)))
        else:
            expanded.append(item)
    expanded.extend([slice(None)] * (3 - len(expanded)))
    if len(expanded) != 3 or any(
        not isinstance(item, slice) or item != slice(None) for item in expanded[:2]
    ):
        return False
    feature = expanded[2]
    return (
        isinstance(feature, slice)
        or (isinstance(feature, (tuple, list)) and all(isinstance(item, int) for item in feature))
        or (
            isinstance(feature, torch.Tensor)
            and feature.ndim == 1
            and feature.dtype in (torch.long, torch.int, torch.bool)
        )
    )


def _framewise(layer: nn.Module) -> bool:
    declaration = _declared_framewise(layer)
    if declaration is not None:
        return declaration
    if type(layer) is nn.Sequential and "forward" not in vars(layer):
        return all(_framewise(child) for child in layer)
    if _scalar_lambda(layer):
        return True
    if "forward" in vars(layer):
        return False
    if type(layer) in _POINTWISE_TYPES:
        return True
    if isinstance(layer, ReadoutLayer) and type(layer).forward in (
        ReadoutLayer.forward,
        IncrementalRidgeReadoutLayer.forward,
    ):
        return True
    if type(layer) in (nn.Softmax, nn.LogSoftmax):
        return cast(nn.Softmax | nn.LogSoftmax, layer).dim == -1
    if type(layer) is nn.LayerNorm:
        return len(layer.normalized_shape) == 1
    if type(layer) is nn.PReLU:
        return layer.num_parameters == 1
    if type(layer) is useful_layers.ConcatLayer:
        return layer.dim in (-1, 2)
    if type(layer) is useful_layers.SliceLayer:
        return _feature_slice(layer.idx)
    return False


def _require_framewise(layer: nn.Module, name: str) -> None:
    if not _framewise(layer):
        raise ValueError(
            f"Fast forecasting cannot safely execute operation '{name}' "
            f"({type(layer).__name__}) one frame at a time. Declare supports_framewise=True "
            "only for a time-separable module, use a BaseReservoirLayer streaming adapter, "
            "or wrap the graph with rd.Model(graph=...).forecast(execution='history') "
            "to replay its complete input prefix."
        )


def validate_fast_plan(plan: ExecutionPlan) -> None:
    """Preflight an existing plan, including mutable containers, without execution."""
    plan.validate_reservoir_visibility()
    for operation in plan.operations:
        if operation.reservoir_index is None:
            _require_framewise(operation.layer, operation.name)


def validate_forecast_execution(
    graph: nn.Module, execution: ForecastExecution = "fast", *, compile: bool = False
) -> None:
    """Validate a whole supported family before hooks, modes or recurrent state."""
    if execution not in ("fast", "history"):
        raise ValueError("execution must be 'fast' or 'history'")
    if execution == "history" and compile:
        raise ValueError("History forecasting requires compile=False")
    # Imports stay lazy: model.py uses the plan validator while core initializes.
    from resdag.core.model import ESNModel

    if isinstance(graph, ESNModel):
        plan = graph._get_execution_plan()
        if execution == "fast":
            validate_fast_plan(plan)
        return
    from resdag.models.parallel_esn import ParallelESNModel

    if isinstance(graph, ParallelESNModel):
        if not isinstance(graph.reservoir, BaseReservoirLayer):
            raise TypeError("Parallel forecasting requires a BaseReservoirLayer reservoir")
        if execution == "fast":
            _require_framewise(graph.readout, "readout")
        return
    from resdag.ensemble.coupled import CoupledEnsembleESNModel

    if isinstance(graph, CoupledEnsembleESNModel):
        for member in graph._iter_models():
            validate_forecast_execution(member, execution, compile=compile)
        aggregator = graph.aggregator_module
        if aggregator is not None:
            if any(isinstance(child, BaseReservoirLayer) for child in aggregator.modules()):
                raise ValueError(
                    "Forecasting cannot thread reservoirs hidden inside a coupled aggregator; "
                    "expose reservoir state in member graph nodes instead."
                )
            if execution == "fast":
                _require_framewise(aggregator, "aggregator")
        return
    raise TypeError("Forecast execution requires an ESN, parallel or coupled graph")


class ForecastSequenceExecutor:
    """Ephemeral explicit-state execution for the common named forecast planner.

    History mode retains every supplied input frame. Each advance recomputes
    the complete prefix from the original explicit state (configured reset when
    omitted), so stateless temporal functions retain their sequence semantics.
    Custom modules must be pure across calls in evaluation mode. Only visible
    BaseReservoirLayer nodes supply mutable temporal state.
    """

    def __init__(
        self,
        graph: nn.Module,
        execution: ForecastExecution = "fast",
        *,
        compile: bool = False,
        workflow: Any = None,
        owner: nn.Module | None = None,
    ) -> None:
        validate_forecast_execution(graph, execution, compile=compile)
        from resdag.core._workflow_backends import workflow_backend

        interface = workflow_backend(graph).describe(graph)
        self._input_features = interface.input_widths
        self._output_features = interface.output_widths
        self._graph = cast(ReservoirModelProtocol, graph)
        self._execution = execution
        self._compile = compile
        self._workflow = workflow
        self._owner: Any = owner
        self._history: tuple[torch.Tensor, ...] | None = None
        self._initial_states: ReservoirStates | None = None
        self._states: ReservoirStates | None = None
        self._layout: tuple[tuple[int, int, torch.device, torch.dtype], ...] | None = None

    def _validate_inputs(self, inputs: tuple[torch.Tensor, ...]) -> bool:
        from resdag.core._hooks import has_execution_hooks

        native = self._owner is not None or has_execution_hooks(cast(nn.Module, self._graph))
        if native and len(inputs) == len(self._input_features):
            # Native prehooks may reconcile feedback placement with other
            # ports or parameters. Validate structure before dispatch, leaving
            # those conversions to the same hooks as ordinary module calls.
            for value, width in zip(inputs, self._input_features):
                validate_sequence_inputs((value,), feature_sizes=(width,), allow_empty=False)
            if any(value.shape[:2] != inputs[0].shape[:2] for value in inputs):
                raise ValueError("Forecast inputs must share batch size and timesteps")
        else:
            validate_sequence_inputs(inputs, feature_sizes=self._input_features, allow_empty=False)
        return native

    def _run(
        self, inputs: tuple[torch.Tensor, ...], states: Mapping[str, torch.Tensor] | None
    ) -> tuple[tuple[torch.Tensor, ...], ReservoirStates]:
        if self._owner is None:
            output, final = self._graph.forward_stateless(
                *inputs, states=states, compile=self._compile
            )
        else:
            supplied = (
                inputs[0] if len(inputs) == 1 else dict(zip(self._workflow.input_names, inputs))
            )
            output, final = self._owner.forward_stateless(supplied, states=states)
            if isinstance(output, dict):
                output = tuple(output[name] for name in self._workflow.output_names)
        outputs = output if isinstance(output, tuple) else (output,)
        if len(outputs) != len(self._output_features) or any(
            not isinstance(value, torch.Tensor)
            or value.ndim != 3
            or value.shape[:2] != inputs[0].shape[:2]
            or value.shape[-1] != width
            for value, width in zip(outputs, self._output_features)
        ):
            raise ValueError("Forecast operations must preserve the input batch and time axes")
        return tuple(value.clone() for value in outputs), final

    def start(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: Mapping[str, torch.Tensor] | None = None,
    ) -> tuple[tuple[torch.Tensor, ...], ReservoirStates]:
        """Evaluate full observed history without changing stored graph state."""
        self._validate_inputs(inputs)
        initial = None if states is None else {key: value.clone() for key, value in states.items()}
        outputs, final = self._run(inputs, initial)
        self._initial_states, self._states = initial, final
        self._history = (
            tuple(value.clone() for value in inputs) if self._execution == "history" else ()
        )
        self._layout = tuple(
            (value.shape[0], value.shape[-1], value.device, value.dtype) for value in inputs
        )
        return outputs, dict(final)

    def advance(
        self, inputs: tuple[torch.Tensor, ...]
    ) -> tuple[tuple[torch.Tensor, ...], ReservoirStates]:
        """Consume one input frame and return its complete-prefix prediction."""
        if self._history is None or self._layout is None:
            raise RuntimeError("Call start(history) before advancing a forecast executor")
        native = self._validate_inputs(inputs)
        layout = tuple(
            (value.shape[0], value.shape[-1], value.device, value.dtype) for value in inputs
        )
        same_layout = (
            tuple(value[:2] for value in layout) == tuple(value[:2] for value in self._layout)
            if native
            else layout == self._layout
        )
        if not same_layout or any(value.shape[1] != 1 for value in inputs):
            raise ValueError("Forecast advance requires one frame per original input stream")
        if self._execution == "history":
            prefix = tuple(torch.cat((old, new), 1) for old, new in zip(self._history, inputs))
            outputs, final = self._run(prefix, self._initial_states)
            self._history = prefix
        else:
            outputs, final = self._run(inputs, self._states)
        self._states = final
        return tuple(value[:, -1:].clone() for value in outputs), dict(final)
