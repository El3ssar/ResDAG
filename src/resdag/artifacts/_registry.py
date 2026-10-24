"""Explicit application-trusted constructors for declarative model artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


class ArtifactError(ValueError):
    """An artifact cannot be represented or reconstructed as requested."""


@dataclass(frozen=True)
class ComponentCodec:
    """A versioned component constructor installed by the application.

    `encode(component)` returns configuration, not its learned tensor state.
    `decode(config)` constructs that component. Nested registered modules and
    ordinary tensor/primitive configuration values are encoded automatically.
    Registering a codec trusts its Python functions; a model file cannot install
    or import a codec by itself.
    """

    name: str
    cls: type
    encode: Callable[[Any], Mapping[str, Any]]
    decode: Callable[[Mapping[str, Any]], Any]
    version: int = 1


_CODECS: dict[str, ComponentCodec] = {}
_TYPES: dict[type, ComponentCodec] = {}


def register_component(
    name: str,
    cls: type,
    *,
    encode: Callable[[Any], Mapping[str, Any]],
    decode: Callable[[Mapping[str, Any]], Any],
    version: int = 1,
) -> type:
    """Register an exact custom type for portable save/load.

    Example:

    ```text
    rd.register_component(
        "my-project.Scale", Scale,
        encode=lambda layer: {"factor": layer.factor},
        decode=lambda config: Scale(**config),
    )
    ```

    Import the registration in a new process before loading. Subclasses need
    their own registration, so their custom behavior cannot disappear silently.
    Use a new version when the component's configuration contract changes.
    """
    from ._api import _initialize

    _initialize()
    if not isinstance(name, str) or not name or name.startswith("resdag."):
        raise ValueError("Custom component name must be nonempty and not start with 'resdag.'")
    return _register(ComponentCodec(name, cls, encode, decode, version))


def _register(codec: ComponentCodec) -> type:
    if not isinstance(codec.cls, type) or not callable(codec.encode) or not callable(codec.decode):
        raise TypeError("A codec requires a class and callable encode/decode functions")
    if type(codec.version) is not int or codec.version < 1:
        raise ValueError("Component version must be a positive integer")
    if codec.name in _CODECS or codec.cls in _TYPES:
        raise ValueError(f"Component name or exact type is already registered: {codec.name!r}")
    _CODECS[codec.name] = codec
    _TYPES[codec.cls] = codec
    return codec.cls


def _codec_for_type(cls: type) -> ComponentCodec:
    try:
        return _TYPES[cls]
    except KeyError as error:
        raise ArtifactError(
            f"Unregistered component {cls.__module__}.{cls.__qualname__}. "
            "Register its configuration with register_component(...), or save an "
            "explicitly trusted Python bundle with trusted=True."
        ) from error
