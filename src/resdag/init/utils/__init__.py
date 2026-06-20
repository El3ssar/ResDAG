"""Initialization utility functions."""

from .graph_tools import connected_graph
from .resolve import (
    InitializerSpec,
    TopologySpec,
    resolve_initializer,
    resolve_topology,
)

__all__ = [
    "connected_graph",
    "resolve_topology",
    "resolve_initializer",
    "TopologySpec",
    "InitializerSpec",
]


def __dir__() -> list[str]:
    """Restrict ``dir()`` / tab-completion to the public API (:pep:`562`)."""
    return list(__all__)
