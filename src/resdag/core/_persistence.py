"""One checkpoint format for graph, parallel and ensemble models."""

import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast

import torch
from torch import nn

ModelT = TypeVar("ModelT", bound=nn.Module)
_FULL_MODEL_KEYS = ("resdag_full_model", "resdag_full_ensemble", "resdag_parallel_model")


class _StatefulModel(Protocol):
    def reset_reservoirs(self) -> None: ...

    def get_state(self) -> dict[str, torch.Tensor]: ...

    def set_state(self, states: Mapping[str, torch.Tensor], strict: bool = True) -> None: ...

    def set_reservoir_states(self, states: Any, strict: bool = True) -> None: ...


def save_checkpoint(
    model: nn.Module, path: str | Path, include_states: bool, metadata: dict[str, Any]
) -> None:
    """Save the entire module tree, including trainable ensemble aggregators."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"state_dict": model.state_dict(), "metadata": metadata}
    if hasattr(model, "models"):
        payload["n_models"] = len(cast(Sequence[nn.Module], model.models))
    if include_states:
        payload["reservoir_states"] = cast(_StatefulModel, model).get_state()
    torch.save(payload, path)


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    strict: bool,
    load_states: bool,
    map_location: Any,
) -> None:
    """Load native state dicts, accepting the previous ensemble format too."""
    checkpoint = torch.load(path, weights_only=False, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path} is not a save() state-dict checkpoint; use load_full().")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        saved_count = checkpoint.get("n_models")
        if saved_count is not None and hasattr(model, "models"):
            count = len(cast(Sequence[nn.Module], model.models))
            if saved_count != count:
                raise ValueError(
                    f"Checkpoint has {saved_count} sub-model(s), but this ensemble has {count}."
                )
    elif "state_dicts" in checkpoint and hasattr(model, "models"):
        # Older ensemble checkpoints did not store aggregator parameters.
        # Native strict loading reports those missing keys instead of hiding them.
        members = checkpoint["state_dicts"]
        model_members = cast(Sequence[nn.Module], model.models)
        if len(members) != len(model_members):
            raise ValueError(
                f"Checkpoint has {len(members)} sub-model(s), "
                f"but this ensemble has {len(model_members)}."
            )
        state_dict = {
            f"models.{index}.{name}": value
            for index, member in enumerate(members)
            for name, value in member.items()
        }
    else:
        raise ValueError(f"{path} is not a save() state-dict checkpoint; use load_full().")
    model.load_state_dict(state_dict, strict=strict)
    if load_states:
        states = checkpoint.get("reservoir_states")
        if states is None:
            warnings.warn(
                "load_states=True but no reservoir states found in checkpoint",
                UserWarning,
                stacklevel=3,
            )
        elif isinstance(states, list):
            # The old coupled-model state API returned one mapping per member.
            cast(_StatefulModel, model).reset_reservoirs()
            cast(_StatefulModel, model).set_reservoir_states(states, strict=False)
        else:
            cast(_StatefulModel, model).reset_reservoirs()
            cast(_StatefulModel, model).set_state(states, strict=False)


def save_full_model(model: nn.Module, path: str | Path, metadata: dict[str, Any]) -> None:
    """Serialize a trusted Python model with its architecture and current state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"resdag_full_model": model, "metadata": metadata}, path)


def load_full_model(
    cls: type[ModelT], path: str | Path, return_metadata: bool, map_location: Any
) -> ModelT | tuple[ModelT, dict[str, Any]]:
    """Accept current wrappers, older family-specific wrappers and raw modules."""
    payload = torch.load(path, weights_only=False, map_location=map_location)
    model = payload
    metadata: dict[str, Any] = {}
    if isinstance(payload, dict):
        model = next((payload[key] for key in _FULL_MODEL_KEYS if key in payload), None)
        metadata = payload.get("metadata", {})
    if not isinstance(model, cls):
        label = "ensemble" if cls.__name__ == "CoupledEnsembleESNModel" else cls.__name__
        raise ValueError(
            f"{path} does not contain a full {label}. Write one with save_full(); "
            "for a state-dict checkpoint, rebuild the architecture and use load()."
        )
    return (model, metadata) if return_metadata else model
