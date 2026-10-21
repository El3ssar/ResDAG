"""Discover, register, and configure recurrent weight initializers.

The registry is shared with :mod:`resdag.init.topology`: existing graph and
matrix registrations work with both APIs. A recurrent initializer specifies
the recurrent weights; the reservoir cell specifies the state update equation.
"""

from typing import Any, Callable

from .topology import (
    GraphTopology,
    MatrixTopology,
    TopologyInitializer,
    get_topology,
    register_matrix_topology,
    show_topologies,
)
from .topology.registry import _TOPOLOGY_REGISTRY
from .utils import RecurrentInitializerSpec, resolve_recurrent_initializer, select_recurrent_spec


def register_recurrent_initializer(
    name: str,
    prescaled: bool = False,
    **default_kwargs: Any,
) -> Callable[[Callable], Callable]:
    """Register a direct matrix builder in the shared recurrent registry.

    Builders accept ``n`` followed by their parameters and return an ``(n, n)``
    matrix. ``prescaled=True`` preserves a builder's spectral profile by
    default; an explicit ``rescale=True`` configuration can override it.
    Graph builders continue to use :func:`register_graph_topology`.
    """
    return register_matrix_topology(name, prescaled=prescaled, **default_kwargs)


def get_recurrent_initializer(
    name: str,
    *,
    rescale: bool | None = None,
    **builder_kwargs: Any,
) -> TopologyInitializer:
    """Configure a registered graph or matrix recurrent initializer.

    ``rescale=None`` retains the registration's policy: ordinary builders are
    scaled to a requested layer radius, while prescaled builders preserve
    their own spectrum and warn on a conflicting layer radius.
    ``rescale=False`` explicitly skips outer spectral estimation/scaling.
    ``rescale=True`` explicitly applies a requested layer radius, even for a
    prescaled builder. With no layer radius, neither choice changes weights.

    For example, use
    ``get_recurrent_initializer("fast_spectral_initialization",
    spectral_radius=0.9, rescale=False)`` for FSI's approximate analytic scale
    without an outer eigendecomposition. The old ``get_topology`` API and
    tuple specs continue forwarding arbitrary keywords to custom builders.
    """
    if rescale is not None and not isinstance(rescale, bool):
        raise TypeError("rescale must be True, False, or None")
    initializer = get_topology(name, **builder_kwargs)
    initializer.rescale = rescale
    return initializer


def show_recurrent_initializers(name: str | None = None) -> list[str] | None:
    """List all graph and matrix initializers, or inspect a named builder.

    This is the same registry exposed by :func:`show_topologies`, so existing
    custom registrations are included automatically.
    """
    if name is not None:
        return show_topologies(name)
    names = sorted(_TOPOLOGY_REGISTRY)
    print("\nAvailable recurrent initializers (graph and matrix):\n")
    for registered_name in names:
        _, _, wrapper, _ = _TOPOLOGY_REGISTRY[registered_name]
        kind = "graph" if wrapper is GraphTopology else "matrix"
        print(f"  - {registered_name} ({kind})")
    print(f"\nTotal: {len(names)}\n")
    return names


__all__ = [
    "GraphTopology",
    "MatrixTopology",
    "TopologyInitializer",
    "RecurrentInitializerSpec",
    "get_recurrent_initializer",
    "register_recurrent_initializer",
    "resolve_recurrent_initializer",
    "select_recurrent_spec",
    "show_recurrent_initializers",
]
