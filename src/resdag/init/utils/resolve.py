"""Resolver utilities for topology and initializer specifications.

This module provides helper functions to resolve flexible specification formats
(strings, tuples, or objects) into concrete initializer/topology objects.
"""

from typing import Any

from ..input_feedback import InputFeedbackInitializer, get_input_feedback
from ..topology import GraphTopology, get_topology

# Type aliases for specification formats
TopologySpec = None | str | tuple[str, dict[str, Any]] | GraphTopology
InitializerSpec = None | str | tuple[str, dict[str, Any]] | InputFeedbackInitializer


def resolve_topology(spec: TopologySpec) -> GraphTopology | None:
    """Resolve topology specification to GraphTopology object.

    Accepts three formats:
    - None: Returns None (use default or no topology)
    - str: Registry name, uses default parameters
    - tuple[str, dict]: (name, params) for custom parameters
    - GraphTopology: Already resolved, returned as-is

    Parameters
    ----------
    spec : TopologySpec
        Topology specification in one of the accepted formats.

    Returns
    -------
    GraphTopology or None
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

    >>> resolve_topology(get_topology("ring_chord"))
    GraphTopology(...)
    """
    if spec is None:
        return None
    if isinstance(spec, GraphTopology):
        return spec
    if isinstance(spec, str):
        return get_topology(spec)
    if isinstance(spec, tuple):
        name, params = spec
        return get_topology(name, **params)
    raise TypeError(
        f"Invalid topology spec type: {type(spec).__name__}. "
        f"Expected str, tuple[str, dict], or GraphTopology."
    )


def resolve_initializer(spec: InitializerSpec) -> InputFeedbackInitializer | None:
    """Resolve initializer specification to InputFeedbackInitializer object.

    Accepts three formats:
    - None: Returns None (use default initializer)
    - str: Registry name, uses default parameters
    - tuple[str, dict]: (name, params) for custom parameters
    - InputFeedbackInitializer: Already resolved, returned as-is

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

    >>> resolve_initializer(get_input_feedback("random_input"))
    RandomInputInitializer(...)
    """
    if spec is None:
        return None
    if isinstance(spec, InputFeedbackInitializer):
        return spec
    if isinstance(spec, str):
        return get_input_feedback(spec)
    if isinstance(spec, tuple):
        name, params = spec
        return get_input_feedback(name, **params)
    raise TypeError(
        f"Invalid initializer spec type: {type(spec).__name__}. "
        f"Expected str, tuple[str, dict], or InputFeedbackInitializer."
    )
