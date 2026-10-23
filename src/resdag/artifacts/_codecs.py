"""Explicit built-in configuration codecs; no artifact-driven Python imports."""

from __future__ import annotations

import inspect
from collections import OrderedDict
from collections.abc import Mapping
from functools import partial
from typing import Any, cast

import torch
from torch import nn

from ._registry import _TYPES, ComponentCodec, _register

_READY = False
_SYMBOLS: dict[str, Any] = {}


def _fields(value: Any, names: tuple[str, ...]) -> dict[str, Any]:
    return {name: getattr(value, name) for name in names}


def _simple(cls: type, names: tuple[str, ...], *, meta: bool = False) -> None:
    def decode(config: Mapping[str, Any]) -> Any:
        if meta:
            with torch.device("meta"):
                return cls(**config)
        return cls(**config)

    _register(
        ComponentCodec(
            f"resdag.component.{cls.__module__}.{cls.__qualname__}",
            cls,
            lambda value: _fields(value, names),
            decode,
        )
    )


def _constructor_fields(cls: type) -> tuple[str, ...]:
    return tuple(
        name
        for name, parameter in inspect.signature(cls).parameters.items()
        if parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
        and name not in ("device", "dtype")
    )


def _readout_config(value: Any, names: tuple[str, ...]) -> dict[str, Any]:
    config = _fields(value, tuple(n for n in names if n not in ("bias",)))
    config["bias"] = value.bias is not None
    return config


def _readouts() -> None:
    from resdag.layers import readouts

    canonical = (
        "ReadoutLayer",
        "AutoReadoutLayer",
        "CGReadoutLayer",
        "RidgeReadoutLayer",
        "CholeskyReadoutLayer",
        "SVDReadoutLayer",
        "PinvReadoutLayer",
        "IncrementalRidgeReadoutLayer",
        "RLSReadout",
        "FISTAReadoutLayer",
    )
    for name in canonical:
        cls = getattr(readouts, name)
        names = tuple(
            n for n in _constructor_fields(cls) if not (name == "RLSReadout" and n == "alpha")
        )

        def decode(config: Mapping[str, Any], cls: type = cls) -> nn.Module:
            with torch.device("meta"):
                return cast(nn.Module, cls(**config))

        _register(
            ComponentCodec(
                f"resdag.readout.{name}",
                cls,
                partial(_readout_config, names=names),
                decode,
            )
        )
    # Existing exported deprecated subclasses retain their exact type.
    for alias, parent in (
        (readouts.FISTAReadout, readouts.FISTAReadoutLayer),
        (readouts.IncrementalRidgeReadout, readouts.IncrementalRidgeReadoutLayer),
    ):
        names = _constructor_fields(parent)
        _register(
            ComponentCodec(
                f"resdag.readout.{alias.__name__}",
                alias,
                partial(_readout_config, names=names),
                partial(_construct, alias),
            )
        )


def _recipe_portable(value: Any, depth: int = 0) -> bool:
    if depth > 32:
        return False
    if value is None or isinstance(
        value, (str, bool, int, float, torch.Tensor, torch.dtype, torch.device, torch.Generator)
    ):
        return True
    if isinstance(value, (tuple, list)):
        return all(_recipe_portable(v, depth + 1) for v in value)
    if isinstance(value, Mapping):
        return all(
            _recipe_portable(k, depth + 1) and _recipe_portable(v, depth + 1)
            for k, v in value.items()
        )
    if any(value is symbol for symbol in _SYMBOLS.values()):
        return True
    codec = _TYPES.get(type(value))
    return codec is not None and _recipe_portable(codec.encode(value), depth + 1)


def _cell_config(value: Any) -> dict[str, Any]:
    from resdag.layers.cells import LegendreDelayCell, NGCell, TorchRecurrentCell

    if type(value) is NGCell:
        return {
            "kwargs": _fields(
                value,
                ("input_dim", "k", "s", "p", "cumulative", "include_constant", "include_linear"),
            ),
            "weights": value.state_dict(),
            "recipe": {},
        }
    if type(value) is TorchRecurrentCell:
        return {
            "kwargs": {
                "cell": value.recurrent_cell,
                **_fields(value, ("feedback_size", "input_size", "trainable")),
            },
            "weights": value.state_dict(),
            "recipe": {},
        }
    names: tuple[str, ...]
    if type(value) is LegendreDelayCell:
        names = (
            "order",
            "feedback_size",
            "input_size",
            "theta",
            "dt",
            "memory_channels",
            "trainable",
        )
    else:
        names = ("reservoir_size", "feedback_size", "input_size", "trainable")
    kwargs = _fields(value, names)
    clsname = type(value).__name__
    extra = {
        "ESNCell": ("leak_rate", "noise"),
        "ES2NCell": ("beta", "orthogonal_scaling"),
        "ResESNCell": ("alpha", "beta", "orthogonal_scaling"),
        "ContinuousReservoirCell": (
            "dt",
            "substeps",
            "solver",
            "torchdiffeq_method",
            "rtol",
            "atol",
        ),
        "EuSNCell": ("gamma", "substeps", "solver", "torchdiffeq_method", "rtol", "atol"),
        "GatedReservoirCell": ("update_gate_bias",),
        "LIFCell": (
            "tau_mem",
            "tau_syn",
            "dt",
            "synaptic_current",
            "v_threshold",
            "reset",
            "surrogate_beta",
            "trace_decay",
            "output",
        ),
        "CoRNNCell": ("dt", "gamma", "epsilon"),
    }
    kwargs.update(_fields(value, extra.get(clsname, ())))
    if hasattr(value, "_activation_name"):
        kwargs["activation"] = value._activation_name
    if hasattr(value, "_gate_activation_name"):
        kwargs["gate_activation"] = value._gate_activation_name
    if clsname == "EuSNCell":
        kwargs["epsilon"] = value.dt
    if clsname == "ESNCell":
        kwargs["sparse"] = value._sparse_requested
    if clsname not in ("LegendreDelayCell", "NGCell", "TorchRecurrentCell"):
        kwargs["bias"] = value.bias_h is not None
        kwargs["bias_scaling"] = value.bias_scaling if hasattr(value, "bias_scaling") else 0.0
    kwargs["seed"] = getattr(value, "_seed_int", None)
    recipe_names = (
        "topology",
        "recurrent_initializer",
        "feedback_initializer",
        "input_initializer",
        "bias_initializer",
        "initial_state_initializer",
        "spectral_radius",
        "gate_recurrent_initializer",
        "gate_feedback_initializer",
        "gate_input_initializer",
        "gate_bias_initializer",
        "gate_spectral_radius",
    )
    recipe = {
        name: getattr(value, name)
        for name in recipe_names
        if hasattr(value, name) and _recipe_portable(getattr(value, name))
    }
    omitted = tuple(name for name in recipe_names if hasattr(value, name) and name not in recipe)
    return {"kwargs": kwargs, "weights": value.state_dict(), "recipe": recipe, "omitted": omitted}


def _cell_decode(cls: type, config: Mapping[str, Any]) -> nn.Module:
    from resdag.init.input_feedback.base import InputFeedbackInitializer
    from resdag.init.topology.base import TopologyInitializer

    class TensorInput(InputFeedbackInitializer):
        def __init__(self, tensor: torch.Tensor) -> None:
            super().__init__()
            self.tensor = tensor

        def initialize(self, weight: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            with torch.no_grad():
                weight.copy_(self.tensor.reshape_as(weight))
            return weight

    class TensorTopology(TopologyInitializer):
        def __init__(self, tensor: torch.Tensor) -> None:
            self.tensor = tensor

        def initialize(
            self, weight: torch.Tensor, spectral_radius: float | None = None
        ) -> torch.Tensor:
            with torch.no_grad():
                weight.copy_(self.tensor)
            return weight

    kwargs, weights = dict(config["kwargs"]), config["weights"]
    if cls.__name__ not in ("NGCell", "TorchRecurrentCell"):
        reference = weights["weight_feedback"]
        kwargs.update(device=reference.device, dtype=reference.dtype)
        kwargs["feedback_initializer"] = TensorInput(reference)
        if kwargs.get("input_size"):
            kwargs["input_initializer"] = TensorInput(weights["weight_input"])
        if cls.__name__ != "LegendreDelayCell":
            recurrent = weights.get("weight_hh", weights.get("weight_rec"))
            kwargs["recurrent_initializer"] = TensorTopology(recurrent)
            kwargs["spectral_radius"] = None
        if cls.__name__ == "CoRNNCell":
            kwargs["velocity_initializer"] = TensorTopology(weights["weight_hv"])
        if cls.__name__ == "GatedReservoirCell":
            kwargs["gate_recurrent_initializer"] = TensorTopology(weights["weight_hz"])
            if weights.get("weight_gate_feedback") is not None:
                kwargs["gate_feedback_initializer"] = TensorInput(weights["weight_gate_feedback"])
            if weights.get("weight_gate_input") is not None:
                kwargs["gate_input_initializer"] = TensorInput(weights["weight_gate_input"])
    result = cls(**kwargs)
    for name, value in config["recipe"].items():
        setattr(result, name, value)
    for name in config.get("omitted", ()):
        setattr(result, name, None)
    return cast(nn.Module, result)


def _reservoirs_and_cells() -> None:
    from resdag.layers import cells, reservoirs
    from resdag.layers.reservoirs.base_reservoir import BaseReservoirLayer

    for name in cells.__all__:
        cls = getattr(cells, name)
        if inspect.isabstract(cls):
            continue
        _register(
            ComponentCodec(
                f"resdag.cell.{name}",
                cls,
                _cell_config,
                partial(_cell_decode, cls),
            )
        )
    for name in reservoirs.__all__:
        cls = getattr(reservoirs, name)

        def decode(config: Mapping[str, Any], cls: type = cls) -> nn.Module:
            # All sequence wrappers add only cell construction to this base.
            concrete: Any = cls
            result = concrete.__new__(concrete)
            BaseReservoirLayer.__init__(result, config["cell"])
            result.detach_state_between_calls = config["detach_state_between_calls"]
            return cast(nn.Module, result)

        _register(
            ComponentCodec(
                f"resdag.reservoir.{name}",
                cls,
                lambda value: {
                    "cell": value.cell,
                    "detach_state_between_calls": value.detach_state_between_calls,
                },
                decode,
            )
        )


def _transforms() -> None:
    from resdag.core._preprocessing import FittedNormalization
    from resdag.ensemble.aggregators import OutliersFilteredMean
    from resdag.layers import transforms

    names = {
        "Concatenate": (),
        "ExtendedSquare": (),
        "NLAT2": (),
        "NLAT3": (),
        "FeaturePartitioner": ("partitions", "overlap"),
        "Pad": ("padding",),
        "PartialSquare": ("eta",),
        "Power": ("exponent", "sign_preserving"),
        "SelectiveDropout": ("mask",),
        "SelectiveExponentiation": ("index", "exponent", "sign_preserving"),
        "Standardize": ("num_features", "eps"),
    }
    for name, fields in names.items():
        _simple(getattr(transforms, name), fields)
    _simple(FittedNormalization, ("method",))
    _simple(OutliersFilteredMean, ("method", "threshold"))


def _torch_modules() -> None:
    # Exact classes only; unsupported standard/custom modules receive the same
    # explicit registration remedy instead of losing constructor behavior.
    specs = {
        nn.Linear: ("in_features", "out_features"),
        nn.Identity: (),
        nn.ReLU: ("inplace",),
        nn.ReLU6: ("inplace",),
        nn.Tanh: (),
        nn.Sigmoid: (),
        nn.Softsign: (),
        nn.Softplus: ("beta", "threshold"),
        nn.ELU: ("alpha", "inplace"),
        nn.CELU: ("alpha", "inplace"),
        nn.SELU: ("inplace",),
        nn.GELU: ("approximate",),
        nn.LeakyReLU: ("negative_slope", "inplace"),
        nn.PReLU: ("num_parameters", "init"),
        nn.SiLU: ("inplace",),
        nn.Mish: ("inplace",),
        nn.Softmax: ("dim",),
        nn.LogSoftmax: ("dim",),
        nn.Hardsigmoid: ("inplace",),
        nn.Hardswish: ("inplace",),
        nn.Hardtanh: ("min_val", "max_val", "inplace"),
        nn.Tanhshrink: (),
        nn.LogSigmoid: (),
        nn.Threshold: ("threshold", "value", "inplace"),
        nn.Hardshrink: ("lambd",),
        nn.Softshrink: ("lambd",),
        nn.AlphaDropout: ("p", "inplace"),
        nn.FeatureAlphaDropout: ("p", "inplace"),
        nn.Dropout: ("p", "inplace"),
        nn.Flatten: ("start_dim", "end_dim"),
        nn.Unflatten: ("dim", "unflattened_size"),
        nn.LayerNorm: ("normalized_shape", "eps", "elementwise_affine"),
        nn.BatchNorm1d: ("num_features", "eps", "momentum", "affine", "track_running_stats"),
        nn.GRUCell: ("input_size", "hidden_size", "bias"),
        nn.LSTMCell: ("input_size", "hidden_size", "bias"),
        nn.RNNCell: ("input_size", "hidden_size", "bias", "nonlinearity"),
    }
    for cls, fields in specs.items():
        if cls is nn.LayerNorm:
            _register(
                ComponentCodec(
                    "resdag.torch.LayerNorm",
                    cls,
                    lambda value: {
                        "normalized_shape": value.normalized_shape,
                        "eps": value.eps,
                        "elementwise_affine": value.elementwise_affine,
                        "bias": value.bias is not None,
                    },
                    partial(_meta_construct, nn.LayerNorm),
                )
            )
        elif cls is nn.Linear:
            _register(
                ComponentCodec(
                    "resdag.torch.Linear",
                    cls,
                    lambda value: {
                        "in_features": value.in_features,
                        "out_features": value.out_features,
                        "bias": value.bias is not None,
                    },
                    lambda config: _meta_construct(nn.Linear, config),
                )
            )
        else:
            _simple(cls, fields, meta=True)
    for cls in (nn.Sequential, nn.ModuleList, nn.ModuleDict):
        _register(
            ComponentCodec(
                f"resdag.torch.{cls.__name__}",
                cls,
                lambda value: {
                    "children": [
                        (name, child)
                        for name, child in value.named_modules(remove_duplicate=False)
                        if name and "." not in name
                    ]
                },
                partial(_container, cls),
            )
        )


def _construct(cls: type, config: Mapping[str, Any]) -> Any:
    return cls(**config)


def _container(cls: type, config: Mapping[str, Any]) -> Any:
    return (
        cls(OrderedDict(config["children"]))
        if cls in (nn.Sequential, nn.ModuleDict)
        else cls([v for _, v in config["children"]])
    )


def _meta_construct(cls: type, config: Mapping[str, Any]) -> Any:
    with torch.device("meta"):
        return cls(**config)


def _symbolic() -> None:
    from pytorch_symbolic import useful_layers as u

    for name in (
        "AddOpLayer",
        "SubOpLayer",
        "MulOpLayer",
        "ModOpLayer",
        "MatmulOpLayer",
        "UnpackLayer",
        "SliceLayerSymbolicIdx",
    ):
        _simple(getattr(u, name), ())
    for name, fields in (
        ("ConcatLayer", ("dim",)),
        ("StackLayer", ("dim",)),
        ("SliceLayer", ("idx",)),
        ("AggregateLayer", ("op", "dim", "keepdim")),
    ):
        _simple(getattr(u, name), fields)
    for name in ("ReshapeLayer", "ViewCopyLayer"):
        cls = getattr(u, name)
        _register(
            ComponentCodec(
                f"resdag.symbolic.{name}",
                cls,
                lambda value: {"shape": value.shape, "batch_size_included": True},
                partial(_construct, cls),
            )
        )
    methods = {
        "reshape",
        "view",
        "flatten",
        "transpose",
        "permute",
        "contiguous",
        "unsqueeze",
        "squeeze",
        "clone",
        "detach",
        "square",
        "pow",
        "abs",
        "sum",
        "mean",
        "median",
        "argmax",
        "argmin",
        "t",
        "__getitem__",
    }

    def method(config: Mapping[str, Any]) -> nn.Module:
        if config["method_name"] not in methods:
            raise ValueError("Artifact contains an unsupported tensor method")
        return cast(
            nn.Module, u.MethodCall(config["method_name"], *config["args"], **config["kwds"])
        )

    _register(
        ComponentCodec(
            "resdag.symbolic.MethodCall",
            u.MethodCall,
            lambda value: _fields(value, ("method_name", "args", "kwds")),
            method,
        )
    )

    # Scalar functional syntax uses lambdas internally. Persist only code IDs
    # from the installed symbolic implementation, never artifact-provided code.
    from types import CellType, CodeType, FunctionType

    from pytorch_symbolic.symbolic_data import SymbolicTensor

    operators = (
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
        "__matmul__",
        "__rmatmul__",
        "t",
    )
    known = {
        (name, i): code
        for name in operators
        for i, code in enumerate(getattr(SymbolicTensor, name).__code__.co_consts)
        if isinstance(code, CodeType)
    }
    globals_ = SymbolicTensor.__add__.__globals__

    def encode_lambda(value: Any) -> Mapping[str, Any]:
        function = value.forward
        if (
            not isinstance(function, FunctionType)
            or function.__globals__ is not globals_
            or function.__defaults__
            or function.__kwdefaults__
        ):
            from ._registry import ArtifactError

            raise ArtifactError(
                "Custom symbolic lambdas require trusted=True or an explicitly registered module"
            )
        key = next((key for key, code in known.items() if function.__code__ is code), None)
        if key is None:
            from ._registry import ArtifactError

            raise ArtifactError(
                "Custom symbolic lambdas require trusted=True or an explicitly registered module"
            )
        return {
            "operator": key,
            "closure": tuple(cell.cell_contents for cell in function.__closure__ or ()),
        }

    def decode_lambda(config: Mapping[str, Any]) -> nn.Module:
        key = tuple(config["operator"])
        if key not in known:
            raise ValueError("Unknown symbolic scalar operation")
        code = known[key]
        closure = config["closure"]
        if len(closure) != len(code.co_freevars):
            raise ValueError("Invalid symbolic scalar constants")
        function = FunctionType(
            code, globals_, closure=tuple(CellType(value) for value in closure) if closure else None
        )
        return cast(nn.Module, u.LambdaOpLayer(function))

    _register(
        ComponentCodec(
            "resdag.symbolic.LambdaOpLayer", u.LambdaOpLayer, encode_lambda, decode_lambda
        )
    )


def _initializers() -> None:
    from resdag.init import graphs, input_feedback, matrices
    from resdag.init.input_feedback.base import InputFeedbackInitializer
    from resdag.init.input_feedback.function import FunctionInitializer
    from resdag.init.topology.base import GraphTopology, MatrixTopology

    for name in dir(input_feedback):
        cls = getattr(input_feedback, name)
        if (
            isinstance(cls, type)
            and issubclass(cls, InputFeedbackInitializer)
            and not inspect.isabstract(cls)
            and cls is not FunctionInitializer
        ):
            fields = _constructor_fields(cls)
            _simple(cls, fields)
    _register(
        ComponentCodec(
            "resdag.initializer.FunctionInitializer",
            FunctionInitializer,
            lambda value: {"fn": value.fn, "kwargs": value.kwargs},
            lambda config: FunctionInitializer(config["fn"], **config["kwargs"]),
        )
    )
    _simple(GraphTopology, ("graph_func", "graph_kwargs", "prescaled", "rescale"))
    _simple(MatrixTopology, ("matrix_func", "matrix_kwargs", "prescaled", "rescale", "seed"))
    for namespace in (graphs, matrices, torch.nn.init):
        for name in dir(namespace):
            value = getattr(namespace, name)
            if callable(value) and not isinstance(value, type) and not name.startswith("_"):
                _SYMBOLS[f"{namespace.__name__}.{name}"] = value
    for name in (
        "sum",
        "mean",
        "median",
        "argmax",
        "argmin",
        "max",
        "min",
        "add",
        "sub",
        "mul",
        "matmul",
        "tanh",
        "sigmoid",
        "relu",
    ):
        _SYMBOLS[f"torch.{name}"] = getattr(torch, name)


def initialize_codecs() -> None:
    global _READY
    if _READY:
        return
    _readouts()
    _reservoirs_and_cells()
    _transforms()
    _torch_modules()
    _symbolic()
    _initializers()
    _READY = True
