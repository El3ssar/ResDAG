"""Versioned complete artifacts over the standard Torch archive format."""

from __future__ import annotations

import io
import math
import pickle
import platform
import threading
import warnings
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from resdag.core._persistence import _atomic_save, _validate_checkpoint_value

from ._codecs import _SYMBOLS, initialize_codecs
from ._models import initialize_model_codecs
from ._registry import _CODECS, ArtifactError, ComponentCodec, _codec_for_type

_FORMAT = "resdag.model"
_VERSION = 1
_MODELS_READY = False
_INITIALIZE_LOCK = threading.RLock()


def _initialize() -> None:
    global _MODELS_READY
    with _INITIALIZE_LOCK:
        initialize_codecs()
        if not _MODELS_READY:
            initialize_model_codecs()
            _MODELS_READY = True


def _plain(value: Any) -> None:
    """Reject object-bearing extra state before writing a portable artifact."""
    try:
        _validate_checkpoint_value(value, "Artifact state/metadata", set())
    except (TypeError, ValueError) as error:
        raise ArtifactError(f"Artifact state/metadata validation failed: {error}") from error
    _portable_plain(value)


def _portable_plain(value: Any) -> None:
    """Retain the portable format's stricter finite primitive/container policy."""
    if value is None or type(value) in (bool, int, str, bytes) or isinstance(value, torch.Tensor):
        pass
    elif type(value) is float and math.isfinite(value):
        pass
    elif isinstance(value, (tuple, list)):
        for child in value:
            _portable_plain(child)
    elif isinstance(value, dict):
        for key, child in value.items():
            _portable_plain(key)
            _portable_plain(child)
    else:
        raise ArtifactError(
            f"Artifact state/metadata contains unsupported value {type(value).__name__}"
        )
    attributes = getattr(value, "__dict__", None)
    if attributes:
        _portable_plain(attributes)


def _generator_config(value: torch.Generator) -> dict[str, Any]:
    return {
        "seed": value.initial_seed(),
        "state": value.get_state(),
        "device": str(value.device),
        "reference": torch.empty(0, device=value.device),
    }


def _restore_generator(config: Mapping[str, Any]) -> torch.Generator:
    reference = config["reference"]
    if not isinstance(reference, torch.Tensor):
        raise ArtifactError("Invalid generator placement reference")
    target = reference.device
    source = torch.device(config["device"])
    # Torch currently supplies Generator objects for CPU and CUDA. A remap to
    # another backend keeps a CPU seed recipe, with an explicit restart notice.
    generator_device = target if target.type in ("cpu", "cuda") else torch.device("cpu")
    result = torch.Generator(device=generator_device).manual_seed(config["seed"])
    if source.type == target.type and generator_device.type == target.type:
        result.set_state(config["state"].cpu())
    else:
        warnings.warn(
            "Generator RNG state cannot be translated across device types; the seeded stream restarts.",
            UserWarning,
            stacklevel=3,
        )
    return result


class _Encoder:
    def __init__(self, include_state: bool) -> None:
        self.include_state = include_state
        self.records: dict[str, Any] = {}
        self.identities: dict[int, str] = {}
        self.active: set[int] = set()
        self.generator_configs: dict[int, dict[str, Any]] = {}

    def encode(self, value: Any) -> Any:
        if value is None or type(value) in (bool, int, str, bytes):
            return {"tag": "literal", "value": value}
        if type(value) is float:
            if not math.isfinite(value):
                raise ArtifactError("Configuration floats must be finite")
            return {"tag": "literal", "value": value}
        if isinstance(value, torch.Tensor):
            return {"tag": "tensor", "value": value.detach()}
        if isinstance(value, torch.dtype):
            return {"tag": "dtype", "value": str(value).removeprefix("torch.")}
        if isinstance(value, torch.device):
            return {"tag": "device", "value": str(value)}
        if isinstance(value, np.ndarray):
            return {"tag": "numpy", "value": torch.from_numpy(value.copy())}
        if isinstance(value, np.generic):
            return self.encode(value.item())
        if isinstance(value, slice):
            return {
                "tag": "slice",
                "value": [self.encode(v) for v in (value.start, value.stop, value.step)],
            }
        if value is Ellipsis:
            return {"tag": "ellipsis"}
        if isinstance(value, torch.Generator):
            if id(value) not in self.generator_configs:
                self.generator_configs[id(value)] = {"tag": "generator", **_generator_config(value)}
            return self.generator_configs[id(value)]
        if isinstance(value, (tuple, list)):
            return {
                "tag": "tuple" if isinstance(value, tuple) else "list",
                "value": [self.encode(v) for v in value],
            }
        if isinstance(value, Mapping):
            return {
                "tag": "dict",
                "value": [[self.encode(k), self.encode(v)] for k, v in value.items()],
            }
        if isinstance(value, type):
            return {"tag": "class", "name": _codec_for_type(value).name}
        for name, symbol in _SYMBOLS.items():
            if value is symbol:
                return {"tag": "symbol", "name": name}
        identity = id(value)
        if identity in self.active:
            raise ArtifactError("Cyclic component configurations are unsupported")
        if identity in self.identities:
            return {"tag": "reference", "id": self.identities[identity]}
        codec = _codec_for_type(type(value))
        key = str(len(self.identities))
        self.identities[identity] = key
        self.active.add(identity)
        try:
            config = codec.encode(value)
            if not isinstance(config, Mapping):
                raise ArtifactError(f"Codec {codec.name!r} must return a configuration mapping")
            record: dict[str, Any] = {
                "name": codec.name,
                "version": codec.version,
                "config": self.encode(config),
            }
            if isinstance(value, nn.Module):
                state = value.state_dict()
                _plain(state)
                record["state_dict"] = state
                record["live"] = _live_state(value, self.include_state)
            self.records[key] = record
        finally:
            self.active.remove(identity)
        return {"tag": "reference", "id": key}


class _Decoder:
    def __init__(
        self, records: Mapping[str, Any], custom_objects: Mapping[str, ComponentCodec] | None
    ) -> None:
        self.records = records
        self.codecs = {**_CODECS, **dict(custom_objects or {})}
        for name, codec in self.codecs.items():
            if not isinstance(codec, ComponentCodec) or name != codec.name:
                raise ArtifactError(
                    "custom_objects must map component IDs to matching ComponentCodec objects"
                )
        self.objects: dict[str, Any] = {}
        self.active: set[str] = set()
        self.generators: dict[int, torch.Generator] = {}

    def decode(self, value: Any) -> Any:
        if not isinstance(value, dict) or not isinstance(value.get("tag"), str):
            raise ArtifactError("Malformed artifact configuration")
        tag = value["tag"]
        if tag in ("literal", "tensor"):
            result = value["value"]
            _plain(result)
            if tag == "tensor" and not isinstance(result, torch.Tensor):
                raise ArtifactError("Invalid tensor configuration")
            return result
        if tag == "dtype":
            result = getattr(torch, value["value"], None)
            if not isinstance(result, torch.dtype):
                raise ArtifactError("Unknown Torch dtype")
            return result
        if tag == "device":
            return torch.device(value["value"])
        if tag == "numpy":
            return value["value"].detach().cpu().numpy().copy()
        if tag == "ellipsis":
            return Ellipsis
        if tag == "slice":
            return slice(*(self.decode(v) for v in value["value"]))
        if tag == "generator":
            if id(value) not in self.generators:
                self.generators[id(value)] = _restore_generator(value)
            return self.generators[id(value)]
        if tag in ("tuple", "list"):
            children = [self.decode(v) for v in value["value"]]
            return tuple(children) if tag == "tuple" else children
        if tag == "dict":
            return {self.decode(k): self.decode(v) for k, v in value["value"]}
        if tag == "symbol":
            try:
                return _SYMBOLS[value["name"]]
            except KeyError as error:
                raise ArtifactError(
                    "Unknown callable symbol; install its explicit component codec"
                ) from error
        if tag == "class":
            return self._codec(value["name"]).cls
        if tag != "reference":
            raise ArtifactError(f"Unknown artifact configuration tag {tag!r}")
        key = value["id"]
        if key in self.objects:
            return self.objects[key]
        if key in self.active or key not in self.records:
            raise ArtifactError("Invalid or cyclic component reference")
        self.active.add(key)
        try:
            record = self.records[key]
            codec = self._codec(record["name"])
            if type(record["version"]) is not int or record["version"] != codec.version:
                raise ArtifactError(
                    f"Unsupported component version for {codec.name!r}; expected {codec.version}"
                )
            config = self.decode(record["config"])
            if not isinstance(config, Mapping):
                raise ArtifactError("Component configuration must be a mapping")
            result = codec.decode(config)
            if type(result) is not codec.cls:
                raise ArtifactError(
                    f"Codec {codec.name!r} reconstructed a different component type"
                )
            if isinstance(result, nn.Module):
                result.load_state_dict(record["state_dict"], strict=True, assign=True)
                if "supports_framewise" in record["live"]:
                    result.supports_framewise = record["live"]["supports_framewise"]
            self.objects[key] = result
            return result
        finally:
            self.active.remove(key)

    def _codec(self, name: str) -> ComponentCodec:
        try:
            return self.codecs[name]
        except KeyError as error:
            raise ArtifactError(
                f"Unknown component {name!r}. Import its register_component(...) registration "
                "or pass a ComponentCodec in custom_objects."
            ) from error


def _live_state(module: nn.Module, include: bool) -> dict[str, Any]:
    from resdag.layers.reservoirs import BaseReservoirLayer

    result: dict[str, Any] = {}
    if "_trainable" in vars(module):
        result["trainable"] = module.trainable
    if "supports_framewise" in vars(module):
        if type(module.supports_framewise) is not bool:
            raise ArtifactError("supports_framewise must be a bool")
        result["supports_framewise"] = module.supports_framewise
    if include and isinstance(module, BaseReservoirLayer):
        result.update(state=module.state, user_set=module._state_user_set)
    if include and "_noise_generators" in vars(module):
        result["noise"] = {
            str(device): generator.get_state()
            for device, generator in vars(module)["_noise_generators"].items()
        }
    return result


def _restore_live(module: nn.Module, state: Mapping[str, Any]) -> None:
    from resdag.layers.reservoirs import BaseReservoirLayer
    from resdag.utils.general import create_torch_generator

    if "trainable" in state:
        if type(state["trainable"]) is not bool:
            raise ArtifactError("Invalid trainable policy")
        setattr(module, "trainable", state["trainable"])
    if "supports_framewise" in state:
        if type(state["supports_framewise"]) is not bool:
            raise ArtifactError("Invalid supports_framewise declaration")
        setattr(module, "supports_framewise", state["supports_framewise"])
    if isinstance(module, BaseReservoirLayer):
        if "state" in state:
            value = state["state"]
            if value is not None:
                module.cell.validate_state(value)
                reference = next(module.parameters(), None)
                if reference is not None and (
                    value.device != reference.device or value.dtype != reference.dtype
                ):
                    raise ArtifactError(
                        "Saved recurrent state placement differs from model parameters"
                    )
                module.state = value.detach().clone()
            else:
                module.state = None
            module._state_user_set = state["user_set"]
        else:
            module.reset_state()
    if "_noise_generators" in vars(module):
        setattr(module, "_noise_generators", {})
        reference = next(module.parameters(), None)
        if reference is not None:
            for saved_device, rng_state in state.get("noise", {}).items():
                device = reference.device
                if torch.device(saved_device).type != device.type:
                    warnings.warn(
                        "Model RNG stream cannot be preserved across different device types; the seeded stream restarts.",
                        UserWarning,
                        stacklevel=3,
                    )
                    continue
                generator = create_torch_generator(getattr(module, "_seed_int", None), device)
                generator.set_state(rng_state.cpu())
                vars(module)["_noise_generators"][device] = generator


def _behavior(model: nn.Module) -> dict[str, Any]:
    aliases: dict[int, list[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    persistent = set(model.state_dict())
    buffer_aliases: dict[int, list[str]] = {}
    buffer_gradients: dict[str, bool] = {}
    for name, buffer in model.named_buffers(remove_duplicate=False):
        if name in persistent:
            buffer_aliases.setdefault(id(buffer), []).append(name)
            buffer_gradients[name] = buffer.requires_grad
    return {
        "buffer_aliases": [names for names in buffer_aliases.values() if len(names) > 1],
        "buffer_gradients": buffer_gradients,
        "modes": {name: m.training for name, m in model.named_modules(remove_duplicate=False)},
        "gradients": {
            name: p.requires_grad for name, p in model.named_parameters(remove_duplicate=False)
        },
        "aliases": [names for names in aliases.values() if len(names) > 1],
    }


def _restore_behavior(model: nn.Module, behavior: Mapping[str, Any]) -> None:
    for names in behavior["buffer_aliases"]:
        buffer = model.get_buffer(names[0])
        for name in names[1:]:
            parent, _, leaf = name.rpartition(".")
            model.get_submodule(parent).register_buffer(leaf, buffer)
    for name, flag in behavior["buffer_gradients"].items():
        if type(flag) is not bool:
            raise ArtifactError("Invalid buffer gradient flag")
        model.get_buffer(name).requires_grad_(flag)
    for names in behavior["aliases"]:
        parameter = model.get_parameter(names[0])
        for name in names[1:]:
            parent, _, leaf = name.rpartition(".")
            model.get_submodule(parent).register_parameter(leaf, parameter)
    for name, flag in behavior["gradients"].items():
        if type(flag) is not bool:
            raise ArtifactError("Invalid parameter gradient flag")
        model.get_parameter(name).requires_grad_(flag)
    for name, flag in behavior["modes"].items():
        if type(flag) is not bool:
            raise ArtifactError("Invalid module training mode")
        model.get_submodule(name).training = flag


def _cloudpickle() -> Any:
    try:
        import cloudpickle
    except ImportError as error:
        raise ImportError(
            "Trusted Python bundles require `pip install 'resdag[artifacts]'` (cloudpickle)."
        ) from error
    return cloudpickle


def _trusted_encode(
    model: nn.Module, cloudpickle: Any, include_state: bool
) -> tuple[bytes, list[torch.Tensor], list[dict[str, Any]]]:
    """Use standard pickle reduction, keeping aliased storages in Torch's archive.

    Shallow state filtering avoids deepcopy, which breaks distinct Parameter
    views sharing storage and cannot copy non-leaf recurrent tensors.
    """
    from pytorch_symbolic.symbolic_data import SymbolicData

    from resdag.layers.reservoirs import BaseReservoirLayer

    tensors: list[torch.Tensor] = []
    identities: dict[int, int] = {}
    generators: list[dict[str, Any]] = []
    generator_ids: dict[int, int] = {}
    hook_names = {
        "_forward_hooks",
        "_forward_pre_hooks",
        "_backward_hooks",
        "_backward_pre_hooks",
        "_forward_hooks_with_kwargs",
        "_forward_hooks_always_called",
        "_forward_pre_hooks_with_kwargs",
    }

    class TensorPickler(cloudpickle.CloudPickler):
        def persistent_id(self, value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                if id(value) not in identities:
                    identities[id(value)] = len(tensors)
                    tensors.append(value)
                return ("resdag.tensor", identities[id(value)])
            if isinstance(value, torch.Generator):
                if id(value) not in generator_ids:
                    generator_ids[id(value)] = len(generators)
                    generators.append(_generator_config(value))
                return ("resdag.generator", generator_ids[id(value)])
            return None

        def reducer_override(self, value: Any) -> Any:
            if isinstance(value, (nn.Module, SymbolicData)):
                reduction = value.__reduce_ex__(pickle.HIGHEST_PROTOCOL)
                if (
                    not isinstance(reduction, tuple)
                    or len(reduction) < 3
                    or not isinstance(reduction[2], dict)
                ):
                    raise ArtifactError(
                        "Trusted modules must expose a normal dictionary pickle state"
                    )
                state = dict(reduction[2])
                if isinstance(value, nn.Module):
                    for key in hook_names & state.keys():
                        state[key] = type(state[key])()
                    for key in (
                        "_compiled_call_impl",
                        "_flat_step_cache",
                        "_execution_plan_cache",
                        "tuning_report",
                        "_trajectory_system",
                    ):
                        if key in state:
                            state[key] = None
                    if "_compiled_scan_cache" in state:
                        state["_compiled_scan_cache"] = {}
                    if not include_state and isinstance(value, BaseReservoirLayer):
                        state["_buffers"] = {**state["_buffers"], "state": None}
                        state["_state_user_set"] = False
                elif state.get("_parents"):
                    state["_value"] = None
                return (*reduction[:2], state, *reduction[3:])
            return super().reducer_override(value)

    stream = io.BytesIO()
    TensorPickler(stream, protocol=pickle.HIGHEST_PROTOCOL).dump(model)
    return stream.getvalue(), tensors, generators


def _trusted_decode(
    code: bytes, tensors: list[torch.Tensor], generators: list[dict[str, Any]]
) -> nn.Module:
    _cloudpickle()
    restored_generators: dict[int, torch.Generator] = {}

    class TensorUnpickler(pickle.Unpickler):
        def persistent_load(self, identity: Any) -> Any:
            if (
                isinstance(identity, tuple)
                and len(identity) == 2
                and identity[0] == "resdag.generator"
                and type(identity[1]) is int
                and 0 <= identity[1] < len(generators)
            ):
                index = identity[1]
                if index not in restored_generators:
                    restored_generators[index] = _restore_generator(generators[index])
                return restored_generators[index]
            if (
                not isinstance(identity, tuple)
                or len(identity) != 2
                or identity[0] != "resdag.tensor"
                or type(identity[1]) is not int
                or not 0 <= identity[1] < len(tensors)
            ):
                raise ArtifactError("Invalid trusted tensor reference")
            return tensors[identity[1]]

    return cast(nn.Module, TensorUnpickler(io.BytesIO(code)).load())


def save_model(
    model: nn.Module,
    path: str | Path,
    *,
    trusted: bool = False,
    include_state: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save a complete model, preserving recurrent memory by default.

    Portable files contain declarative registered configuration and tensor state.
    `trusted=True` uses the optional cloudpickle package to include Python
    components such as notebook-local classes and lambdas. Loading those files
    requires explicit trust and compatible Python/package versions.

    `include_state=False` resets ordinary recurrent memory on reload. Fitted
    preprocessing and the observed history used by `forecast` are retained.
    Neither option saves an autonomous forecast cursor, optimizer or autograd graph.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if type(trusted) is not bool or type(include_state) is not bool:
        raise TypeError("trusted and include_state must be bool values")
    path = Path(path)
    if path.suffix.lower() not in (".pt", ".pth"):
        raise ArtifactError(
            "Complete model artifacts use .pt or .pth; diagram formats are .svg/.png/.pdf/.dot"
        )
    meta = dict(metadata or {})
    _plain(meta)
    _initialize()
    from resdag.core._execution import validate_fitting_forward
    from resdag.core.model import ESNModel

    for module in model.modules():
        if isinstance(module, ESNModel):
            try:
                validate_fitting_forward(module)
            except ValueError as error:
                raise ArtifactError(
                    "Saving a customized outer ESNModel.forward is unsupported; put the behavior inside a declared graph module."
                ) from error
    payload: dict[str, Any] = {
        "format": _FORMAT,
        "version": _VERSION,
        "producer": {
            "resdag": version("resdag"),
            "torch": str(torch.__version__),
            "python": platform.python_version(),
        },
        "metadata": meta,
        "behavior": _behavior(model),
        "trusted": trusted,
    }
    with torch.random.fork_rng(devices=[]):
        if trusted:
            cloudpickle = _cloudpickle()
            live = {
                name: _live_state(module, include_state) for name, module in model.named_modules()
            }
            code, tensors, generators = _trusted_encode(model, cloudpickle, include_state)
            payload.update(python=code, tensors=tensors, generators=generators, live=live)
        else:
            encoder = _Encoder(include_state)
            payload.update(root=encoder.encode(model), components=encoder.records)
        _atomic_save(payload, path)
    return path


def load(
    path: str | Path,
    *,
    trusted: bool = False,
    map_location: Any = "cpu",
    custom_objects: Mapping[str, ComponentCodec] | None = None,
    return_metadata: bool = False,
) -> Any:
    """Reconstruct a saved model without knowing its original DAG.

    The default uses Torch's restricted tensor/primitive loader and never imports
    artifact-selected Python paths. `trusted=True` permits Python unpickling;
    use it only for a file whose producer and contents you trust. Restricted
    loading reduces code execution exposure but does not prevent malformed-file
    memory/resource exhaustion. Device remapping follows standard `torch.load`.
    """
    if type(trusted) is not bool:
        raise TypeError("trusted must be a bool")
    _initialize()
    try:
        with torch.inference_mode(False):
            payload = torch.load(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError as error:
        raise ArtifactError(
            "Restricted loading rejected Python objects. For a trusted Python bundle only, explicitly pass trusted=True; never enable this for an unknown file."
        ) from error
    if not isinstance(payload, dict) or payload.get("format") != _FORMAT:
        raise ArtifactError(
            "This is not a complete ResDAG artifact; state_dict files require a matching model and load_state_dict"
        )
    if type(payload.get("version")) is not int or payload["version"] != _VERSION:
        raise ArtifactError(
            f"Unsupported ResDAG artifact version {payload.get('version')!r}; this installation supports version {_VERSION}"
        )
    if payload.get("trusted") and not trusted:
        raise ArtifactError(
            "This artifact contains executable Python; explicitly pass trusted=True only if its producer is trusted"
        )
    with torch.inference_mode(False), torch.random.fork_rng(devices=[]):
        if payload.get("trusted"):
            model = _trusted_decode(payload["python"], payload["tensors"], payload["generators"])
            if not isinstance(model, nn.Module):
                raise ArtifactError("Trusted artifact root is not a Torch module")
            for name, module in model.named_modules():
                _restore_live(module, payload["live"].get(name, {}))
        else:
            decoder = _Decoder(payload["components"], custom_objects)
            model = decoder.decode(payload["root"])
            if not isinstance(model, nn.Module):
                raise ArtifactError("Artifact root is not a Torch module")
            # Functional construction runs trace placeholders. Restore final
            # persistent and live state again after all parent graphs are built.
            for key, module in decoder.objects.items():
                if isinstance(module, nn.Module):
                    module.load_state_dict(
                        decoder.records[key]["state_dict"], strict=True, assign=True
                    )
                    _restore_live(module, decoder.records[key]["live"])
        _restore_behavior(model, payload["behavior"])
    meta = payload["metadata"]
    _plain(meta)
    return (model, meta) if return_metadata else model
