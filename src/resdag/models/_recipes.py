"""Conservative reconstruction recipes for the built-in functional presets.

Recipes describe construction, not serialized executable Python. A snapshot of
the actual feature graph prevents tuning from silently replacing user edits.
This module deliberately has no dependency on the optional tuning package.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
from collections.abc import Callable
from functools import wraps
from typing import Any, cast

from torch import nn

_RUNTIME = {
    "_builtin_recipe",
    "_factory_kwargs",
    "_compiled_call_impl",
    "_compiled_scan_cache",
    "_state_user_set",
    "_noise_generators",
    "_sparse_weight_key",
    "_sparse_active",
    "_hh_structure_key",
    "_fitted_flag",
    "last_solve",
}
_GRAPH_INTERNALS = {
    "inputs",
    "outputs",
    "_layer_type_counts",
    "_node_to_layer_name",
    "_execution_order_nodes",
    "_execution_order_layers",
    "_execution_plan",
    "_flat_step",
    "_flat_step_cache",
    "_execution_plan_cache",
}


def _snapshot(graph: Any) -> str:
    from resdag.core._execution import _symbolic_operations
    from resdag.facade import _configuration
    from resdag.layers.readouts import ReadoutLayer

    base = set(nn.Module().__dict__)
    readouts = tuple(
        name for name, module in graph.named_modules() if isinstance(module, ReadoutLayer)
    )
    modules = []
    for name, module in graph.named_modules(remove_duplicate=False):
        attributes = {
            key: value
            for key, value in vars(module).items()
            if key not in base and key not in _RUNTIME and (name or key not in _GRAPH_INTERNALS)
        }
        if "seed" in attributes and hasattr(module, "_seed_int"):
            attributes["seed"] = module._seed_int
        hooks = {
            key: tuple(getattr(module, key).values())
            for key in (
                "_forward_hooks",
                "_forward_pre_hooks",
                "_backward_hooks",
                "_backward_pre_hooks",
            )
        }
        is_readout = any(name == prefix or name.startswith(prefix + ".") for prefix in readouts)
        tensors = (
            {}
            if is_readout
            else {
                **dict(module.named_parameters(recurse=False)),
                **{
                    key: value
                    for key, value in module.named_buffers(recurse=False)
                    if key not in module._non_persistent_buffers_set
                },
            }
        )
        parameter_schema = tuple(
            (key, tuple(value.shape), str(value.dtype), value.requires_grad)
            for key, value in module.named_parameters(recurse=False)
        )
        buffer_schema = tuple(
            (key, tuple(value.shape), str(value.dtype))
            for key, value in module.named_buffers(recurse=False)
            if key not in module._non_persistent_buffers_set
        )
        modules.append(
            (
                name,
                type(module).__module__,
                type(module).__qualname__,
                attributes,
                hooks,
                tensors,
                parameter_schema,
                buffer_schema,
            )
        )
    slots = {id(node): i for i, node in enumerate(graph.inputs)}
    calls = []
    for node, layer, parents, siblings in _symbolic_operations(graph):
        outputs = siblings if len(siblings) > 1 else (node,)
        indices = tuple(range(len(slots), len(slots) + len(outputs)))
        calls.append(
            (graph._node_to_layer_name[node], tuple(slots[id(p)] for p in parents), indices)
        )
        slots.update((id(output), index) for output, index in zip(outputs, indices))
    payload = (modules, calls, tuple(slots[id(node)] for node in graph.outputs))
    return hashlib.sha256(repr(_configuration(payload)).encode()).hexdigest()


def builtin_recipe(graph: nn.Module) -> dict[str, Any] | None:
    """Return metadata only while the live graph still matches its recipe."""
    recipe = graph.__dict__.get("_builtin_recipe")
    if recipe is None or recipe.get("snapshot") != _snapshot(graph):
        return None
    return cast(dict[str, Any], recipe)


def record_recipe(family: str) -> Callable:
    def decorate(factory: Callable) -> Callable:
        signature = inspect.signature(factory)

        @wraps(factory)
        def build(*args: Any, **kwargs: Any) -> Any:
            graph = factory(*args, **kwargs)
            # Capturing metadata must not invoke initialization or perturb RNG.
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            options = dict(bound.arguments)
            options.update(options.pop("reservoir_kwargs", {}))
            try:
                options = copy.deepcopy(options)
                recipe = {"family": family, "options": options, "snapshot": _snapshot(graph)}
            except Exception:
                # A user object need not support copying to build a valid graph.
                return graph
            object.__setattr__(graph, "_builtin_recipe", recipe)
            return graph

        return build

    return decorate


def workflow_options(workflow: Any) -> dict[str, Any]:
    """Read current workflow settings, including explicit transform overrides."""
    return {
        "input_names": workflow.input_names,
        "output_names": workflow.output_names,
        "washout": workflow.washout,
        "preprocessing": workflow.preprocessing,
        "prediction_lead": workflow.prediction_leads,
        "feedback": workflow._feedback,
        "resample_dt": workflow.resample_dt,
        "resample_method": workflow.resample_method,
        "input_transforms": {
            name: workflow.input_scalers[i]
            for i, name in enumerate(workflow.input_names)
            if workflow._custom_inputs[i]
        },
        "output_transforms": {
            name: workflow.output_scalers[i]
            for i, name in enumerate(workflow.output_names)
            if workflow._custom_outputs[i]
        },
    }


def input_scaling_supported(options: dict[str, Any]) -> bool:
    from resdag.init.input_feedback import FunctionInitializer, InputFeedbackInitializer
    from resdag.init.input_feedback.registry import _INPUT_FEEDBACK_REGISTRY

    spec = options.get("feedback_initializer")
    if spec is None:
        return True
    if isinstance(spec, InputFeedbackInitializer):
        return not isinstance(spec, FunctionInitializer) and hasattr(spec, "input_scaling")
    name = spec[0] if isinstance(spec, tuple) else spec
    if not isinstance(name, str) or name not in _INPUT_FEEDBACK_REGISTRY:
        return False
    target = _INPUT_FEEDBACK_REGISTRY[name][0]
    if not isinstance(target, type) or not issubclass(target, InputFeedbackInitializer):
        return False
    parameters = inspect.signature(target).parameters
    return "input_scaling" in parameters or any(
        p.kind == p.VAR_KEYWORD for p in parameters.values()
    )


def erdos_renyi_options(options: dict[str, Any]) -> tuple[str, Any, bool] | None:
    from resdag.init.graphs.erdos_renyi import erdos_renyi_graph
    from resdag.init.topology import GraphTopology
    from resdag.init.topology.registry import _TOPOLOGY_REGISTRY

    key = (
        "recurrent_initializer" if options.get("recurrent_initializer") is not None else "topology"
    )
    spec = options.get(key)
    if spec is None:
        return key, ("erdos_renyi", {}), True
    if isinstance(spec, GraphTopology):
        if spec.graph_func is erdos_renyi_graph:
            return key, spec, spec.graph_kwargs.get("self_loops", True)
        return None
    name, settings = spec if isinstance(spec, tuple) else (spec, {})
    if isinstance(name, str) and name in _TOPOLOGY_REGISTRY:
        function, defaults, _, _ = _TOPOLOGY_REGISTRY[name]
        if function is erdos_renyi_graph:
            return (
                key,
                (name, settings),
                settings.get("self_loops", defaults.get("self_loops", True)),
            )
    return None


def sampled_options(options: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Apply shared scalar controls without replacing custom initializer choices."""
    from resdag.init.input_feedback import InputFeedbackInitializer
    from resdag.init.topology import GraphTopology

    supplied = dict(params)
    special = {
        key: supplied.pop(key)
        for key in ("alpha", "input_scaling", "mean_degree")
        if key in supplied
    }
    result = copy.deepcopy({**options, **supplied})
    if "alpha" in special:
        result["readout_alpha"] = special["alpha"]
        readout = dict(result.get("readout_kwargs") or {})
        readout.pop("alpha", None)
        result["readout_kwargs"] = readout
    if "input_scaling" in special:
        scaling = special["input_scaling"]
        if not input_scaling_supported(result):
            raise ValueError(
                "input_scaling requires a shared-scaling initializer; provide a custom factory"
            )
        spec = result.get("feedback_initializer")
        if isinstance(spec, InputFeedbackInitializer):
            spec.input_scaling = scaling
        else:
            name, settings = spec if isinstance(spec, tuple) else (spec or "uniform", {})
            result["feedback_initializer"] = (name, {**settings, "input_scaling": scaling})
    if "mean_degree" in special:
        degree = special["mean_degree"]
        topology = erdos_renyi_options(result)
        if topology is None:
            raise ValueError(
                "mean_degree requires an Erdos-Renyi topology; provide a custom factory"
            )
        key, spec, loops = topology
        width = result["reservoir_size"]
        available = width if loops else max(width - 1, 1)
        if (
            isinstance(degree, bool)
            or not isinstance(degree, (int, float))
            or not 0 < degree <= available
        ):
            raise ValueError(
                "mean_degree must be positive and at most the available entries per row"
            )
        if isinstance(spec, GraphTopology):
            spec.graph_kwargs["p"] = degree / available
            result[key] = spec
        else:
            name, settings = spec
            result[key] = (name, {**settings, "p": degree / available})
    return result
