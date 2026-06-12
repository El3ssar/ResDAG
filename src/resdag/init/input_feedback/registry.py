"""Registry for input/feedback weight initializers.

This module provides a registry of initializers for rectangular weight matrices
used in reservoir input and feedback connections.
"""

import inspect
from typing import Any, Callable, get_args, get_origin

from .base import InputFeedbackInitializer
from .function import FunctionInitializer

# Registry of initializer names to (class_or_function, default_kwargs).
# Entries are either InputFeedbackInitializer subclasses (instantiated with
# the merged kwargs) or plain matrix-building functions (wrapped in
# FunctionInitializer with the merged kwargs).
_INPUT_FEEDBACK_REGISTRY: dict[str, tuple[Callable, dict[str, Any]]] = {}


def register_input_feedback(
    name: str,
    **default_kwargs: Any,
) -> Callable[[Callable], Callable]:
    """Decorator to register an input/feedback initializer.

    Registers either an :class:`InputFeedbackInitializer` subclass **or a
    plain matrix-building function** in the registry at definition time,
    making it available by name to ``ESNLayer`` and other components.

    Parameters
    ----------
    name : str
        Name for the initializer (must be unique)
    **default_kwargs
        Default keyword arguments for the initializer constructor (classes)
        or for the function call (plain functions)

    Returns
    -------
    callable
        Decorator function

    Examples
    --------
    Registering a class:

    >>> @register_input_feedback("my_init", scaling=0.5)
    ... class MyInitializer(InputFeedbackInitializer):
    ...     def __init__(self, scaling=1.0):
    ...         self.scaling = scaling
    ...
    ...     def initialize(self, weight, **kwargs):
    ...         # ... initialization logic
    ...         return weight

    Registering a plain function (build style — return the matrix):

    >>> @register_input_feedback("first_neuron", scale=1.0)
    ... def first_neuron(rows, cols, scale=1.0):
    ...     w = torch.zeros(rows, cols)
    ...     w[0, :] = scale
    ...     return w

    Notes
    -----
    - Classes must inherit from ``InputFeedbackInitializer`` and implement
      ``initialize(weight, **kwargs)``.
    - Functions follow the :class:`FunctionInitializer` conventions:
      ``fn(rows, cols, **kwargs) -> matrix`` or in-place ``fn(tensor, **kwargs)``.
    - Registered initializers can be accessed via ``get_input_feedback(name)``.
    """

    def decorator(obj: Callable) -> Callable:
        if name in _INPUT_FEEDBACK_REGISTRY:
            raise ValueError(f"Input/feedback initializer '{name}' is already registered")
        _INPUT_FEEDBACK_REGISTRY[name] = (obj, default_kwargs)
        return obj

    return decorator


def get_input_feedback(
    name: str,
    **override_kwargs: Any,
) -> InputFeedbackInitializer:
    """Get a pre-configured input/feedback initializer by name.

    Parameters
    ----------
    name : str
        Name of the initializer (e.g., "random", "binary_balanced")
    **override_kwargs
        Keyword arguments to override default initializer parameters

    Returns
    -------
    InputFeedbackInitializer
        Initializer instance

    Raises
    ------
    ValueError
        If initializer name is not registered

    Examples
    --------
    >>> initializer = get_input_feedback("binary_balanced", input_scaling=0.5)
    >>> weight = torch.empty(100, 10)
    >>> initializer.initialize(weight)
    """
    if name not in _INPUT_FEEDBACK_REGISTRY:
        available = ", ".join(_INPUT_FEEDBACK_REGISTRY.keys())
        raise ValueError(f"Unknown initializer '{name}'. Available initializers: {available}")

    entry, default_kwargs = _INPUT_FEEDBACK_REGISTRY[name]

    # Merge default kwargs with overrides
    kwargs = {**default_kwargs, **override_kwargs}

    if inspect.isclass(entry) and issubclass(entry, InputFeedbackInitializer):
        return entry(**kwargs)
    return FunctionInitializer(entry, **kwargs)


def show_input_initializers(name: str | None = None) -> list[str] | None:
    """Show available input/feedback initializers or details for a specific one.

    Parameters
    ----------
    name : str, optional
        Name of initializer to inspect. If None, prints all initializers
        *and* returns them as a list.

    Returns
    -------
    list of str or None
        When ``name is None``, returns the sorted list of registered
        initializer names (in addition to printing them).  When ``name``
        is provided, returns ``None`` after printing the parameter table.

    Raises
    ------
    ValueError
        If the specified initializer name is not registered.

    Examples
    --------
    >>> show_input_initializers()
    ['binary_balanced', 'chebyshev', 'chessboard', ...]
    """
    if name is None:
        names = sorted(_INPUT_FEEDBACK_REGISTRY)
        print("\nAvailable input/feedback initializers:\n")
        for n in names:
            print(f"  - {n}")
        print(f"\nTotal: {len(names)}\n")
        return names

    if name not in _INPUT_FEEDBACK_REGISTRY:
        available = "\n".join(sorted(_INPUT_FEEDBACK_REGISTRY.keys()))
        raise ValueError(f"Unknown initializer '{name}'.\nAvailable:\n{available}")

    entry, default_kwargs = _INPUT_FEEDBACK_REGISTRY[name]
    is_class = inspect.isclass(entry)
    sig = inspect.signature(entry.__init__) if is_class else inspect.signature(entry)

    print(f"\nInitializer: {name} ({'class' if is_class else 'function'})\n")
    print("Parameters:\n")

    # Skip non-hyperparameter params: self (classes) and the matrix
    # dimensions / target tensor (functions).
    skip = {"self"} if is_class else {"rows", "cols", "tensor", "weight"}
    for param_name, param in sig.parameters.items():
        if param_name in skip:
            continue

        # Type extraction
        if param.annotation is not inspect.Parameter.empty:
            origin = get_origin(param.annotation)
            if origin is None:
                type_str = param.annotation.__name__
            else:
                args = get_args(param.annotation)
                type_str = " | ".join(a.__name__ for a in args)
        else:
            type_str = "Any"

        # Default resolution
        if param.default is not inspect.Parameter.empty:
            default = param.default
        else:
            default = default_kwargs.get(param_name, "<required>")

        print(f"  - {param_name}")
        print(f"      type:    {type_str}")
        print(f"      default: {default}\n")

    return None
