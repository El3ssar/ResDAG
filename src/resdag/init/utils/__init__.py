"""Initialization utility functions."""

from .graph_tools import connected_graph
from .resolve import (
    InitializerSpec,
    RecurrentInitializerSpec,
    TopologySpec,
    resolve_initializer,
    resolve_recurrent_initializer,
    resolve_topology,
    select_recurrent_spec,
)

__all__ = [
    "connected_graph",
    "resolve_topology",
    "resolve_initializer",
    "resolve_recurrent_initializer",
    "select_recurrent_spec",
    "TopologySpec",
    "InitializerSpec",
    "RecurrentInitializerSpec",
]


def __dir__() -> list[str]:
    """Restrict ``dir()`` / tab-completion to the public API (:pep:`562`)."""
    return list(__all__)
