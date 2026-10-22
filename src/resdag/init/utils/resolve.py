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
from resdag.init.input_feedback.registry import _INPUT_FEEDBACK_REGISTRY
from resdag.init.topology import MatrixTopology, TopologyInitializer, get_topology
from resdag.init.topology.base import _accepts_generator
from resdag.init.topology.registry import _TOPOLOGY_REGISTRY
from resdag.utils.general import SeedLike, coerce_seed_to_int

# Type aliases for specification formats
TopologySpec = None | str | Callable | tuple[str | Callable, dict[str, Any]] | TopologyInitializer
RecurrentInitializerSpec = TopologySpec
InitializerSpec = (
    None | str | Callable | tuple[str | Callable, dict[str, Any]] | InputFeedbackInitializer
)


def select_recurrent_spec(
    *,
    topology: TopologySpec = None,
    recurrent_initializer: RecurrentInitializerSpec = None,
) -> RecurrentInitializerSpec:
    """Choose the recurrent specification, rejecting two explicit choices.

    ``topology`` is the original spelling. Both parameters accept graph and
    matrix builders; ``recurrent_initializer`` makes that scope explicit.
    """
    if topology is not None and recurrent_initializer is not None:
        raise ValueError("Specify only one of topology and recurrent_initializer, not both.")
    return recurrent_initializer if recurrent_initializer is not None else topology


def resolve_recurrent_initializer(
    spec: RecurrentInitializerSpec,
    seed: SeedLike = None,
) -> TopologyInitializer | None:
    """Resolve a graph or matrix recurrent initializer using the shared registry.

    Strings, ``(name_or_callable, kwargs)`` tuples, dimension-based or in-place
    callables, and preconfigured objects retain :func:`resolve_topology`'s
    seeding and calling conventions.
    """
    return resolve_topology(spec, seed=seed)


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


def _generator_seed(builder: Callable, params: dict[str, Any], seed: int | None) -> int | None:
    """Return the seed to hand a :class:`MatrixTopology` for ``generator=`` builders.

    In-place initializers like :func:`torch.nn.init.orthogonal_` take a
    ``generator=`` keyword rather than ``seed=``; :meth:`MatrixTopology.initialize`
    then builds a device-correct :class:`torch.Generator` from this seed. The
    seed is forwarded only when it is not None, the builder does *not* already
    take a ``seed`` (that path threads the int through ``kwargs`` instead), the
    builder *does* accept ``generator``, and the spec did not already bind one.
    """
    if _accepts_seed(builder) or "generator" in params:
        return None
    if not _accepts_generator(builder):
        return None
    return coerce_seed_to_int(params.get("seed", seed))


def _callback_kwargs(builder: Callable, params: dict[str, Any], seed: int | None) -> dict[str, Any]:
    """Keep ``seed`` out of callbacks that take a Torch generator instead."""
    kwargs = _inject_seed(dict(params), builder, seed)
    if not _accepts_seed(builder) and _accepts_generator(builder):
        kwargs.pop("seed", None)
    return kwargs


def _topology_builder(topology: TopologyInitializer) -> Callable | None:
    """Return the wrapped builder of a resolved topology, or None."""
    builder = getattr(topology, "graph_func", None)
    if builder is None:
        builder = getattr(topology, "matrix_func", None)
    return builder


def resolve_topology(
    spec: TopologySpec,
    seed: SeedLike = None,
) -> TopologyInitializer | None:
    """Resolve a topology specification to a TopologyInitializer object.

    Accepts six formats:

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
    seed : int or torch.Generator, optional
        Seed forwarded to the underlying builder for reproducibility. It is
        applied only to the ``str``, ``tuple[str, dict]``, ``callable``, and
        ``tuple[callable, dict]`` forms, only when the builder accepts a
        ``seed`` argument, and only when the spec did not already pin one (an
        explicit spec seed always wins). A :class:`torch.Generator` is reduced
        to its ``initial_seed()`` before use. Pre-resolved
        ``TopologyInitializer`` objects are returned untouched.

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
    seed = coerce_seed_to_int(seed)
    if spec is None:
        return None
    if isinstance(spec, TopologyInitializer):
        return spec
    if isinstance(spec, str):
        topology = get_topology(spec)
        builder = _topology_builder(topology)
        if builder is not None and _accepts_seed(builder) and seed is not None:
            return get_topology(spec, seed=seed)
        if isinstance(topology, MatrixTopology) and builder is not None:
            if seed is not None:
                topology.seed = _generator_seed(builder, {}, seed)
        return topology
    if isinstance(spec, tuple):
        name, params = spec
        if callable(name):
            return MatrixTopology(
                name,
                _callback_kwargs(name, dict(params), seed),
                seed=_generator_seed(name, dict(params), seed),
            )
        if name not in _TOPOLOGY_REGISTRY:
            return get_topology(name, **dict(params))
        builder = _TOPOLOGY_REGISTRY[name][0]
        injected = (
            _inject_seed(dict(params), builder, seed) if builder is not None else dict(params)
        )
        topology = get_topology(name, **injected)
        if isinstance(topology, MatrixTopology) and builder is not None:
            topology.seed = _generator_seed(builder, dict(params), seed)
        return topology
    if callable(spec):
        return MatrixTopology(
            spec, _callback_kwargs(spec, {}, seed), seed=_generator_seed(spec, {}, seed)
        )
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


def _function_initializer(
    builder: Callable, params: dict[str, Any], seed: int | None
) -> FunctionInitializer:
    """Bind callback keywords without reserving a callback's ``seed`` argument."""
    initializer = FunctionInitializer(builder, **_callback_kwargs(builder, params, seed))
    initializer._generator_seed = _generator_seed(builder, params, seed)
    return initializer


def resolve_initializer(
    spec: InitializerSpec,
    seed: SeedLike = None,
) -> InputFeedbackInitializer | None:
    """Resolve an initializer specification to an InputFeedbackInitializer object.

    Accepts six formats:

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
    seed : int or torch.Generator, optional
        Seed forwarded to the underlying initializer for reproducibility. It
        is applied only to the ``str``, ``tuple[str, dict]``, ``callable``,
        and ``tuple[callable, dict]`` forms, only when the initializer accepts
        a ``seed`` argument, and only when the spec did not already pin one (an
        explicit spec seed always wins). A :class:`torch.Generator` is reduced
        to its ``initial_seed()`` before use. Pre-resolved
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
    seed = coerce_seed_to_int(seed)
    if spec is None:
        return None
    if isinstance(spec, InputFeedbackInitializer):
        return spec
    if isinstance(spec, str):
        initializer = get_input_feedback(spec)
        target = _initializer_seed_target(initializer)
        if _accepts_seed(target) and seed is not None:
            return get_input_feedback(spec, seed=seed)
        if isinstance(initializer, FunctionInitializer):
            if seed is not None:
                initializer._generator_seed = _generator_seed(initializer.fn, {}, seed)
        return initializer
    if isinstance(spec, tuple):
        name, params = spec
        if callable(name):
            return _function_initializer(name, dict(params), seed)
        if name not in _INPUT_FEEDBACK_REGISTRY:
            return get_input_feedback(name, **dict(params))
        # Inspect the registered constructor, without constructing invalid
        # defaults before the caller's required options have been supplied.
        target = _INPUT_FEEDBACK_REGISTRY[name][0]
        injected = _inject_seed(dict(params), target, seed)
        initializer = get_input_feedback(name, **injected)
        if isinstance(initializer, FunctionInitializer):
            initializer._generator_seed = _generator_seed(initializer.fn, dict(params), seed)
        return initializer
    if callable(spec):
        return _function_initializer(spec, {}, seed)
    raise TypeError(
        f"Invalid initializer spec type: {type(spec).__name__}. "
        f"Expected str, callable, tuple[str | callable, dict], or InputFeedbackInitializer."
    )
