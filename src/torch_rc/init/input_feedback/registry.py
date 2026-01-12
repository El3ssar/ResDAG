"""Registry for input/feedback weight initializers.

This module provides a registry of initializers for rectangular weight matrices
used in reservoir input and feedback connections.
"""

import inspect
from typing import Any, Callable, Type, get_args, get_origin

from .base import InputFeedbackInitializer

# Registry of initializer names to (class, default_kwargs)
_INPUT_FEEDBACK_REGISTRY: dict[str, tuple[Type[InputFeedbackInitializer], dict[str, Any]]] = {}


def register_input_feedback(
    name: str,
    **default_kwargs: Any,
) -> Callable[[Type[InputFeedbackInitializer]], Type[InputFeedbackInitializer]]:
    """Decorator to register an input/feedback initializer class.

    This decorator registers an initializer class in the registry at definition time,
    making it available for use with ReservoirLayer and other components.

    Parameters
    ----------
    name : str
        Name for the initializer (must be unique)
    **default_kwargs
        Default keyword arguments for the initializer constructor

    Returns
    -------
    callable
        Decorator function

    Examples
    --------
    >>> @register_input_feedback("my_init", scaling=0.5)
    ... class MyInitializer(InputFeedbackInitializer):
    ...     def __init__(self, scaling=1.0):
    ...         self.scaling = scaling
    ...
    ...     def initialize(self, weight, **kwargs):
    ...         # ... initialization logic
    ...         return weight

    Notes
    -----
    - Initializer classes must inherit from InputFeedbackInitializer
    - Initializer classes must implement initialize(weight, **kwargs) method
    - Registered initializers can be accessed via get_input_feedback(name)
    """

    def decorator(init_class: Type[InputFeedbackInitializer]) -> Type[InputFeedbackInitializer]:
        if name in _INPUT_FEEDBACK_REGISTRY:
            raise ValueError(f"Input/feedback initializer '{name}' is already registered")
        _INPUT_FEEDBACK_REGISTRY[name] = (init_class, default_kwargs)
        return init_class

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

    init_class, default_kwargs = _INPUT_FEEDBACK_REGISTRY[name]

    # Merge default kwargs with overrides
    kwargs = {**default_kwargs, **override_kwargs}

    return init_class(**kwargs)


def show_input_initializers(name: str | None = None) -> list[str] | None:
    """Show available input/feedback initializers or details for a specific one.

    Parameters
    ----------
    name : str, optional
        Name of initializer to inspect. If None, returns list of all initializers.

    Returns
    -------
    list[str] | None
        If name is None: sorted list of registered initializer names.
        If name is provided: dict with 'name', 'defaults', and 'parameters' keys.

    Raises
    ------
    ValueError
        If the specified initializer name is not registered.

    Examples
    --------
    >>> show_input_initializers()
    ['binary_balanced', 'chebyshev', 'chessboard', ...]

    >>> show_input_initializers("chebyshev")
    {
        'name': 'chebyshev',
        'defaults': {'input_scaling': 1.0},
        'parameters': {
            'input_scaling': {'type': 'float', 'default': 1.0},
            ...
        }
    }
    """
    if name is None:
        return sorted(_INPUT_FEEDBACK_REGISTRY.keys())

    if name not in _INPUT_FEEDBACK_REGISTRY:
        available = ", ".join(sorted(_INPUT_FEEDBACK_REGISTRY.keys()))
        raise ValueError(f"Unknown initializer '{name}'. Available: {available}")

    init_class, default_kwargs = _INPUT_FEEDBACK_REGISTRY[name]

    sig = inspect.signature(init_class.__init__)
    types: dict[str, str] = {}

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue

        if param.annotation is not inspect.Parameter.empty:
            origin = get_origin(param.annotation)
            if origin is None:
                types[param_name] = param.annotation.__name__
            else:
                args = get_args(param.annotation)
                types[param_name] = " | ".join(a.__name__ for a in args)
        else:
            types[param_name] = "Any"

    info = {
        "name": name,
        "parameters": {
            k: {
                "type": types.get(k, "Any"),
                "default": (
                    sig.parameters[k].default
                    if sig.parameters[k].default is not inspect.Parameter.empty
                    else default_kwargs.get(k, "<required>")
                ),
            }
            for k in sorted(set(types) | set(default_kwargs))
        },
    }

    return _format_init(info)


def _format_init(info: dict) -> str:
    """Format and print initializer information dictionary."""
    lines = [f"\nInitializer: {info['name']}", "", "Parameters:"]
    for name, meta in info["parameters"].items():
        lines.append(f"  - {name}: type={meta['type']}, default={meta['default']}")
    print("\n".join(lines))
