"""Registry for input/feedback weight initializers.

This module provides a registry of initializers for rectangular weight matrices
used in reservoir input and feedback connections.
"""

import inspect
from typing import Any, Callable, Dict, List, Optional, Type, Union

from .base import InputFeedbackInitializer

# Registry of initializer names to (class, default_kwargs)
_INPUT_FEEDBACK_REGISTRY: Dict[str, tuple[Type[InputFeedbackInitializer], Dict[str, Any]]] = {}


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


def show_input_initializers(name: Optional[str] = None) -> Union[List[str], Dict[str, Any]]:
    """Show available input/feedback initializers or details for a specific one.

    Parameters
    ----------
    name : str, optional
        Name of initializer to inspect. If None, returns list of all initializers.

    Returns
    -------
    list of str or dict
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

    # Extract __init__ signature (skip 'self')
    sig = inspect.signature(init_class.__init__)
    parameters = {}
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue

        param_info: Dict[str, Any] = {}

        # Get type annotation if available
        if param.annotation != inspect.Parameter.empty:
            param_info["type"] = getattr(param.annotation, "__name__", str(param.annotation))
        else:
            param_info["type"] = "Any"

        # Get default value
        if param.default != inspect.Parameter.empty:
            param_info["default"] = param.default
        elif param_name in default_kwargs:
            param_info["default"] = default_kwargs[param_name]
        else:
            param_info["default"] = "<required>"

        parameters[param_name] = param_info

    return {
        "name": name,
        "defaults": default_kwargs,
        "parameters": parameters,
    }
