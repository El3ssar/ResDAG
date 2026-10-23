"""Configuration adapters for functional, local-patch and coupled model families."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from ._registry import ArtifactError, ComponentCodec, _register


def _graph_config(graph: Any) -> dict[str, Any]:
    from resdag.core._execution import _symbolic_operations

    values = {id(node): i for i, node in enumerate(graph.inputs)}
    calls = []
    for node, layer, parents, siblings in _symbolic_operations(graph):
        outputs = siblings if len(siblings) > 1 else (node,)
        slots = []
        for output in outputs:
            slot = len(values)
            values[id(output)] = slot
            slots.append(slot)
        calls.append(
            {
                "module": layer,
                "name": graph._node_to_layer_name[node],
                "parents": tuple(values[id(parent)] for parent in parents),
                "outputs": tuple(slots),
            }
        )
    return {
        "inputs": tuple(
            {
                "shape": tuple(node.shape),
                "dtype": node.v.dtype,
                "device": node.v.device,
                "known_batch": node.batch_size_known,
            }
            for node in graph.inputs
        ),
        "calls": calls,
        "outputs": tuple(values[id(node)] for node in graph.outputs),
        "codegen": graph._enable_forward_codegen,
    }


def _graph_decode(config: Mapping[str, Any]) -> nn.Module:
    from pytorch_symbolic.symbolic_data import SymbolicTensor
    from pytorch_symbolic.useful_layers import UnpackLayer

    from resdag.core.model import ESNModel
    from resdag.layers.reservoirs import BaseReservoirLayer

    values: list[Any] = []
    for port in config["inputs"]:
        shape = port["shape"]
        if not shape or any(type(size) is not int or size < 0 for size in shape):
            raise ArtifactError("Invalid symbolic input shape")
        # Stored placeholders contain no user observations. Device is resolved
        # from restored operations rather than a stale saved device string.
        from resdag._contracts import module_reference

        reference = next(
            (
                ref
                for call in config["calls"]
                if (ref := module_reference(call["module"])) is not None
            ),
            None,
        )
        device = torch.device("cpu") if reference is None else reference.device
        dtype = port["dtype"]
        if reference is not None and dtype.is_floating_point:
            dtype = reference.dtype
        values.append(
            SymbolicTensor(
                value=torch.zeros(shape, dtype=dtype, device=device),
                batch_size_known=port["known_batch"],
            )
        )
    inputs = tuple(values)
    # Tracing is real execution in the functional backend. Use evaluation mode,
    # then restore all state/modes through the normal artifact state restoration.
    modes = [
        (child, child.training) for call in config["calls"] for child in call["module"].modules()
    ]
    try:
        for child, _ in modes:
            child.training = False
        for call in config["calls"]:
            parents = call["parents"]
            if not parents or any(type(i) is not int or not 0 <= i < len(values) for i in parents):
                raise ArtifactError("Invalid or cyclic symbolic value reference")
            if not isinstance(call["name"], str) or not call["name"].isidentifier():
                raise ArtifactError("Symbolic module names must be valid Python identifiers")
            module = call["module"]
            args = tuple(values[i] for i in parents)
            if type(module) is UnpackLayer:
                result = tuple(args[0])
                for node in result:
                    node.layer = module
                result[0]._custom_provided_name = call["name"]
            else:
                node = args[0].apply_module(module, *args[1:], custom_name=call["name"])
                result = (node,)
            slots = call["outputs"]
            if tuple(slots) != tuple(range(len(values), len(values) + len(result))):
                raise ArtifactError("Symbolic output count or slot order differs from the artifact")
            values.extend(result)
        outputs = config["outputs"]
        if not outputs or any(type(i) is not int or not 0 <= i < len(values) for i in outputs):
            raise ArtifactError("Invalid symbolic output reference")
        return ESNModel(
            inputs, tuple(values[i] for i in outputs), enable_forward_codegen=config["codegen"]
        )
    finally:
        for child, mode in modes:
            child.training = mode
        for child, _ in modes:
            if isinstance(child, BaseReservoirLayer):
                child.reset_state()


def _workflow_config(value: Any) -> dict[str, Any]:
    return {
        "graph": value.graph,
        "input_names": value.input_names,
        "output_names": value.output_names,
        "washout": value.washout,
        "preprocessing": value.preprocessing,
        "prediction_lead": value.prediction_leads,
        "feedback": value._feedback,
        "resample_dt": value.resample_dt,
        "resample_method": value.resample_method,
        "input_transforms": {
            name: value.input_scalers[i]
            for i, name in enumerate(value.input_names)
            if value._custom_inputs[i]
        },
        "output_transforms": {
            name: value.output_scalers[i]
            for i, name in enumerate(value.output_names)
            if value._custom_outputs[i]
        },
    }


def _preset_config(value: Any) -> dict[str, Any]:
    options = dict(value._factory_options)
    options["alpha"] = options.pop("readout_alpha")
    options.update(value._workflow_options)
    workflow = value.workflow
    if workflow is not None:
        options.update(
            {
                k: v
                for k, v in _workflow_config(workflow).items()
                if k not in ("graph", "input_names", "output_names")
            }
        )
    return {
        "options": options,
        "workflow": workflow,
        "input_size": value.input_size,
        "reference": value._reference,
        "device_pinned": value._device_pinned,
        "dtype_pinned": value._dtype_pinned,
    }


def _preset_decode(config: Mapping[str, Any]) -> nn.Module:
    from resdag.facade import ESN

    reference = config["reference"]
    result = ESN(**config["options"], device=reference.device, dtype=reference.dtype)
    workflow = config["workflow"]
    if workflow is not None:
        result.workflow = workflow
        result._input_size = config["input_size"]
        result._pending_input_transforms = None
        result._pending_output_transforms = None
    result._device_pinned = config["device_pinned"]
    result._dtype_pinned = config["dtype_pinned"]
    return result


def initialize_model_codecs() -> None:
    from resdag.core import (
        ESNModel,
        Feedback,
        Model,
        ReservoirFeatureExtractor,
        ReservoirModelModule,
    )
    from resdag.ensemble.coupled import CoupledEnsembleESNModel
    from resdag.facade import ESN
    from resdag.models.parallel_esn import ParallelESNModel

    _register(ComponentCodec("resdag.ESNModel", ESNModel, _graph_config, _graph_decode))
    _register(ComponentCodec("resdag.Model", Model, _workflow_config, lambda c: Model(**c)))
    _register(ComponentCodec("resdag.ESN", ESN, _preset_config, _preset_decode))
    _register(
        ComponentCodec(
            "resdag.Feedback",
            Feedback,
            lambda v: {"output": v.output, "input": v.input, "delay": v.delay},
            lambda c: Feedback(**c),
        )
    )
    _register(
        ComponentCodec(
            "resdag.ParallelESNModel",
            ParallelESNModel,
            lambda v: {
                "reservoir": v.reservoir,
                "readout": v.readout,
                "input_dim": v.input_dim,
                "core_width": v.core_width,
                "halo": v.halo,
            },
            lambda c: ParallelESNModel(**c),
        )
    )
    _register(
        ComponentCodec(
            "resdag.CoupledEnsembleESNModel",
            CoupledEnsembleESNModel,
            lambda v: {
                "models": list(v.models),
                "aggregator": (
                    v.aggregator_module if v.aggregator_module is not None else v._aggregator_str
                ),
            },
            lambda c: CoupledEnsembleESNModel(**c),
        )
    )
    _register(
        ComponentCodec(
            "resdag.ReservoirModelModule",
            ReservoirModelModule,
            lambda v: {"model": v.model, "stateful": v.stateful},
            lambda c: ReservoirModelModule(**c),
        )
    )
    _register(
        ComponentCodec(
            "resdag.ReservoirFeatureExtractor",
            ReservoirFeatureExtractor,
            lambda v: {"layers": list(v.reservoirs), "stateful": v.stateful},
            lambda c: ReservoirFeatureExtractor(**c),
        )
    )
