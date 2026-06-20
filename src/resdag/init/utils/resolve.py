"""Resolver utilities for topology and initializer specifications.

This module provides helper functions to resolve flexible specification formats
(strings, tuples, callables, or objects) into concrete initializer/topology
objects.
"""

import inspect
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


def _accepts_seed(func: Callable) -> bool:
    """Return True if ``func`` accepts a ``seed`` keyword argument.

    A builder accepts ``seed`` if its signature names a ``seed`` parameter or
    declares ``**kwargs`` (variadic keyword). When introspection fails (e.g.
    a C builtin without a signature) we assume it does not.
    """
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False
    if "seed" in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _inject_seed(kwargs: dict[str, Any], builder: Callable, seed: int | None) -> dict[str, Any]:
    """Return ``kwargs`` with ``seed`` added if appropriate.

    The seed is injected only when ``seed`` is not None, ``kwargs`` does not
    already pin a ``seed`` (an explicit spec value always wins), and ``builder``
    accepts a ``seed`` argument.
    """
    if seed is None or "seed" in kwargs or not _accepts_seed(builder):
        return kwargs
    return {**kwargs, "seed": seed}


def _topology_builder(topology: TopologyInitializer) -> Callable | None:
    """Return the wrapped builder of a resolved topology, or None."""
    builder = getattr(topology, "graph_func", None)
    if builder is None:
        builder = getattr(topology, "matrix_func", None)
    return builder


def resolve_topology(
    spec: TopologySpec,
    seed: int | None = None,
) -> TopologyInitializer | None:
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
    seed : int, optional
        Seed forwarded to the underlying builder for reproducibility. It is
        applied only to the ``str``, ``tuple[str, dict]``, ``callable``, and
        ``tuple[callable, dict]`` forms, only when the builder accepts a
        ``seed`` argument, and only when the spec did not already pin one (an
        explicit spec seed always wins). Pre-resolved ``TopologyInitializer``
        objects are returned untouched.

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

    >>> resolve_topology("erdos_renyi", seed=42)
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
        topology = get_topology(spec)
        builder = _topology_builder(topology)
        if builder is not None and _accepts_seed(builder) and seed is not None:
            return get_topology(spec, seed=seed)
        return topology
    if isinstance(spec, tuple):
        name, params = spec
        if callable(name):
            return MatrixTopology(name, _inject_seed(dict(params), name, seed))
        builder = _topology_builder(get_topology(name))
        injected = (
            _inject_seed(dict(params), builder, seed) if builder is not None else dict(params)
        )
        return get_topology(name, **injected)
    if callable(spec):
        return MatrixTopology(spec, _inject_seed({}, spec, seed))
    raise TypeError(
        f"Invalid topology spec type: {type(spec).__name__}. "
        f"Expected str, callable, tuple[str | callable, dict], or TopologyInitializer."
    )


def _initializer_seed_target(initializer: InputFeedbackInitializer) -> Callable:
    """Return the callable whose signature decides ``seed`` acceptance.

    For function-wrapped initializers (:class:`FunctionInitializer`) this is
    the wrapped function; for class-based initializers it is the class
    constructor.
    """
    fn: Callable | None = getattr(initializer, "fn", None)
    if fn is not None:
        return fn
    return type(initializer)


def resolve_initializer(
    spec: InitializerSpec,
    seed: int | None = None,
) -> InputFeedbackInitializer | None:
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
    seed : int, optional
        Seed forwarded to the underlying initializer for reproducibility. It
        is applied only to the ``str``, ``tuple[str, dict]``, ``callable``,
        and ``tuple[callable, dict]`` forms, only when the initializer accepts
        a ``seed`` argument, and only when the spec did not already pin one (an
        explicit spec seed always wins). Pre-resolved
        ``InputFeedbackInitializer`` objects are returned untouched.

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

    >>> resolve_initializer("random", seed=42)
    RandomInputInitializer(...)

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
        initializer = get_input_feedback(spec)
        target = _initializer_seed_target(initializer)
        if _accepts_seed(target) and seed is not None:
            return get_input_feedback(spec, seed=seed)
        return initializer
    if isinstance(spec, tuple):
        name, params = spec
        if callable(name):
            return FunctionInitializer(name, **_inject_seed(dict(params), name, seed))
        target = _initializer_seed_target(get_input_feedback(name))
        injected = _inject_seed(dict(params), target, seed)
        return get_input_feedback(name, **injected)
    if callable(spec):
        return FunctionInitializer(spec, **_inject_seed({}, spec, seed))
    raise TypeError(
        f"Invalid initializer spec type: {type(spec).__name__}. "
        f"Expected str, callable, tuple[str | callable, dict], or InputFeedbackInitializer."
    )
