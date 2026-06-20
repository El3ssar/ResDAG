"""
Input/Feedback Initializer Base Class
=====================================

This module provides the abstract base class for all input and feedback
weight initializers used in reservoir layers.

Classes
-------
InputFeedbackInitializer
    Abstract base class for weight initialization.

See Also
--------
resdag.init.input_feedback.registry : Registry of available initializers.
resdag.layers.ESNLayer : Uses these initializers for weight matrices.
"""

from abc import ABC, abstractmethod

import numpy as np
import torch


class InputFeedbackInitializer(ABC):
    """
    Abstract base class for input/feedback weight initialization.

    All input/feedback weight initializers should inherit from this class
    and implement the :meth:`initialize` method. These initializers create
    weight matrices for:

    - Input connections: shape ``(reservoir_size, input_size)``
    - Feedback connections: shape ``(reservoir_size, feedback_size)``

    Subclasses must implement the :meth:`initialize` method which modifies
    the weight tensor in-place and returns it.

    Examples
    --------
    Creating a custom initializer:

    >>> class MyInitializer(InputFeedbackInitializer):
    ...     def initialize(self, weight, **kwargs):
    ...         torch.nn.init.uniform_(weight, -1, 1)
    ...         return weight
    >>>
    >>> initializer = MyInitializer()
    >>> weight = torch.empty(100, 10)
    >>> initializer(weight)

    Using with ESNLayer:

    >>> from resdag.layers import ESNLayer
    >>> reservoir = ESNLayer(
    ...     reservoir_size=100,
    ...     feedback_size=10,
    ...     feedback_initializer=MyInitializer(),
    ... )

    See Also
    --------
    resdag.init.input_feedback.registry : Get initializers by name.
    resdag.layers.ESNLayer : Uses these initializers.
    """

    @abstractmethod
    def initialize(self, weight: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Initialize a weight tensor.

        Parameters
        ----------
        weight : torch.Tensor
            2D tensor of shape ``(reservoir_size, input_size)`` to initialize.
            Modified in-place.
        **kwargs
            Additional keyword arguments for specific initializers.

        Returns
        -------
        torch.Tensor
            The initialized weight tensor (same as input, modified in-place).
        """
        pass

    def __call__(self, weight: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Callable interface for initialization.

        Parameters
        ----------
        weight : torch.Tensor
            Weight tensor to initialize.
        **kwargs
            Additional arguments passed to :meth:`initialize`.

        Returns
        -------
        torch.Tensor
            The initialized weight tensor.
        """
        return self.initialize(weight, **kwargs)


def _resolve_shape(weight: torch.Tensor) -> tuple[int, int]:
    """Resolve tensor shape, ensuring it's 2D."""
    if weight.ndim != 2:
        raise ValueError(f"Weight must be 2D, got shape {weight.shape}")
    return weight.shape[0], weight.shape[1]


def _numpy_compute_dtype(weight_dtype: torch.dtype) -> type[np.floating]:
    """Pick the NumPy dtype to build initializer intermediates in.

    Structured initializers compute their values in NumPy before copying the
    result into the target ``weight``. The intermediate dtype must match the
    target's precision so that no precision is silently truncated: building a
    ``float32`` intermediate for a ``float64`` weight rounds every value to
    single precision before the widening ``.to()`` ever runs.

    The mapping mirrors the target precision exactly where NumPy has a matching
    type, and falls back to ``float64`` for the half-precision torch dtypes
    (``float16``/``bfloat16``), which have no convenient NumPy compute type — the
    final ``.to(dtype=weight.dtype)`` narrows the double-precision result down to
    the target with a single rounding.

    Parameters
    ----------
    weight_dtype : torch.dtype
        Dtype of the target weight tensor.

    Returns
    -------
    type of numpy.floating
        ``numpy.float32`` for ``torch.float32`` targets, otherwise
        ``numpy.float64``. Computing a ``float32`` target in ``float32`` keeps
        its result bit-for-bit identical to the historical behavior, while every
        other target gains genuine double-precision intermediates.

    Examples
    --------
    >>> import torch
    >>> _numpy_compute_dtype(torch.float64) is np.float64
    True
    >>> _numpy_compute_dtype(torch.float32) is np.float32
    True
    """
    if weight_dtype == torch.float32:
        return np.float32
    return np.float64
