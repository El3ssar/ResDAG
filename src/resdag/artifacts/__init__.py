"""Complete model artifacts using standard Torch serialization.

Portable artifacts reconstruct registered components without loading embedded
Python code. ``trusted=True`` explicitly enables an optional cloudpickle bundle.
Ordinary ``state_dict`` remains the standard state-only PyTorch interface.
"""

from ._api import load, save_model
from ._registry import ArtifactError, ComponentCodec, register_component

__all__ = ["ArtifactError", "ComponentCodec", "load", "register_component", "save_model"]


def __dir__() -> list[str]:
    """List the supported artifact interfaces for discovery and completion."""
    return list(__all__)
