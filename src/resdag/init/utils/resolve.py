"""Resolver utilities for topology and initializer specifications.

This module provides helper functions to resolve flexible specification formats
(strings, tuples, callables, or objects) into concrete initializer/topology
objects.
"""

from typing import Any, Callable

from resdag.init.input_feedback import (
    FunctionInitializer,
    InputFeedbackInitializer,
    get_input_feedback,
)
from resdag.init.topology import MatrixTopology, TopologyInitializer, get_topology

# Type aliases for specification formats
TopologySpec = None | str | Callable | tuple[str | Callable, dict[str, Any]] | TopologyInitializer
InitializerSpec = (
    None | str | Callable | tuple[str | Callable, dict[str, Any]] | InputFeedbackInitializer
)


def resolve_topology(spec: TopologySpec) -> TopologyInitializer | None:
    """Resolve a topology specification to a TopologyInitializer object.

    Accepts five formats:

    - ``None`` — returns None (use default random initialization)
    - ``str`` — registry name, uses registered default parameters
    - ``tuple[str, dict]`` — registry name with parameter overrides
    - ``callable`` — any matrix builder ``fn(n, **kw) -> matrix | graph`` or
      in-place ``fn(tensor, **kw)``; wrapped in :class:`MatrixTopology`
    - ``tuple[callable, dict]`` — matrix builder with bound parameters
    - ``TopologyInitializer`` — already resolved, returned as-is

    Parameters
    ----------
    spec : TopologySpec
        Topology specification in one of the accepted formats.

    Returns
    -------
    TopologyInitializer or None
        Resolved topology object, or None if spec was None.

    Raises
    ------
    TypeError
        If spec is not one of the accepted types.

    Examples
    --------
    >>> resolve_topology("erdos_renyi")
    GraphTopology(...)

    >>> resolve_topology(("watts_strogatz", {"k": 6, "p": 0.1}))
    GraphTopology(...)

    >>> resolve_topology(my_matrix_fn)
    MatrixTopology(...)

    >>> resolve_topology((my_matrix_fn, {"blocks": 4}))
    MatrixTopology(...)

    >>> resolve_topology(get_topology("ring_chord"))
    GraphTopology(...)
    """
    if spec is None:
        return None
    if isinstance(spec, TopologyInitializer):
        return spec
    if isinstance(spec, str):
        return get_topology(spec)
    if isinstance(spec, tuple):
        name, params = spec
        if callable(name):
            return MatrixTopology(name, dict(params))
        return get_topology(name, **params)
    if callable(spec):
        return MatrixTopology(spec)
    raise TypeError(
        f"Invalid topology spec type: {type(spec).__name__}. "
        f"Expected str, callable, tuple[str | callable, dict], or TopologyInitializer."
    )


def resolve_initializer(spec: InitializerSpec) -> InputFeedbackInitializer | None:
    """Resolve an initializer specification to an InputFeedbackInitializer object.

    Accepts five formats:

    - ``None`` — returns None (use default random initialization)
    - ``str`` — registry name, uses registered default parameters
    - ``tuple[str, dict]`` — registry name with parameter overrides
    - ``callable`` — any matrix builder ``fn(rows, cols, **kw) -> matrix`` or
      in-place ``fn(tensor, **kw)``; wrapped in :class:`FunctionInitializer`
    - ``tuple[callable, dict]`` — matrix builder with bound parameters
    - ``InputFeedbackInitializer`` — already resolved, returned as-is

    Parameters
    ----------
    spec : InitializerSpec
        Initializer specification in one of the accepted formats.

    Returns
    -------
    InputFeedbackInitializer or None
        Resolved initializer object, or None if spec was None.

    Raises
    ------
    TypeError
        If spec is not one of the accepted types.

    Examples
    --------
    >>> resolve_initializer("pseudo_diagonal")
    PseudoDiagonalInitializer(...)

    >>> resolve_initializer(("chebyshev", {"p": 0.5, "q": 3.0}))
    ChebyshevInitializer(...)

    >>> resolve_initializer(torch.nn.init.xavier_uniform_)
    FunctionInitializer(...)

    >>> resolve_initializer((my_matrix_fn, {"scale": 0.5}))
    FunctionInitializer(...)

    >>> resolve_initializer(get_input_feedback("random"))
    RandomInitializer(...)
    """
    if spec is None:
        return None
    if isinstance(spec, InputFeedbackInitializer):
        return spec
    if isinstance(spec, str):
        return get_input_feedback(spec)
    if isinstance(spec, tuple):
        name, params = spec
        if callable(name):
            return FunctionInitializer(name, **dict(params))
        return get_input_feedback(name, **params)
    if callable(spec):
        return FunctionInitializer(spec)
    raise TypeError(
        f"Invalid initializer spec type: {type(spec).__name__}. "
        f"Expected str, callable, tuple[str | callable, dict], or InputFeedbackInitializer."
    )
