"""
Function-Based Input/Feedback Initializer
=========================================

This module provides :class:`FunctionInitializer`, the adapter that turns
any matrix-building callable into an input/feedback initializer — the
rectangular-matrix counterpart of
:class:`~resdag.init.topology.MatrixTopology`.

See Also
--------
resdag.init.input_feedback.base : Abstract base class.
resdag.init.topology.base : MatrixTopology for recurrent weights.
"""

from typing import Any, Callable

import torch

from resdag.init.topology.base import _call_matrix_builder, _coerce_to_matrix

from .base import InputFeedbackInitializer


class FunctionInitializer(InputFeedbackInitializer):
    """
    Input/feedback initializer wrapping any matrix-building callable.

    Lets plain functions act as initializers for the rectangular input and
    feedback weight matrices, without subclassing. Two calling conventions
    are supported, tried in order:

    1. **Build style** — ``fn(rows, cols, **kwargs)`` returning a
       ``(rows, cols)`` array-like (``torch.Tensor`` or ``numpy.ndarray``).
    2. **In-place style** — ``fn(tensor, **kwargs)`` mutating a tensor in
       place, e.g. any ``torch.nn.init.*_`` function.

    Parameters
    ----------
    fn : callable
        The matrix-building function (build style or in-place style).
    **kwargs
        Keyword arguments bound to the function.

    Examples
    --------
    A plain function as an initializer:

    >>> import torch
    >>> def first_neuron_only(rows, cols, scale=1.0):
    ...     w = torch.zeros(rows, cols)
    ...     w[0, :] = scale
    ...     return w
    >>> init = FunctionInitializer(first_neuron_only, scale=0.5)
    >>> weight = torch.empty(100, 3)
    >>> init.initialize(weight)

    Bare callables passed to ``ESNLayer`` are wrapped automatically:

    >>> from resdag.layers import ESNLayer
    >>> layer = ESNLayer(100, feedback_size=3, feedback_initializer=first_neuron_only)
    >>> layer = ESNLayer(
    ...     100,
    ...     feedback_size=3,
    ...     feedback_initializer=(first_neuron_only, {"scale": 0.1}),
    ... )

    ``torch.nn.init`` functions work directly (in-place style):

    >>> layer = ESNLayer(100, feedback_size=3, feedback_initializer=torch.nn.init.xavier_uniform_)

    See Also
    --------
    InputFeedbackInitializer : Base class for class-based initializers.
    resdag.init.input_feedback.register_input_feedback : Register by name.
    """

    def __init__(self, fn: Callable, **kwargs: Any) -> None:
        self.fn = fn
        self.kwargs = kwargs

    def initialize(self, weight: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Initialize a rectangular weight tensor from the wrapped callable.

        Parameters
        ----------
        weight : torch.Tensor
            2D tensor of shape ``(rows, cols)`` to initialize. Modified
            in-place.
        **kwargs
            Per-call overrides merged over the bound keyword arguments (a
            recognized key wins for this call only). This is the same per-call
            contract the class-based initializers honor — see the **Per-call
            overrides** section of
            :class:`~resdag.init.input_feedback.InputFeedbackInitializer`.

        Returns
        -------
        torch.Tensor
            The initialized weight tensor (same as input, modified in-place).

        Raises
        ------
        ValueError
            If ``weight`` is not 2-D, if the callable matches neither
            calling convention, or if the built matrix has the wrong shape.
        """
        if weight.ndim != 2:
            raise ValueError(f"Weight must be 2D, got shape {weight.shape}")

        rows, cols = weight.shape
        call_kwargs = {**self.kwargs, **kwargs}

        result = _call_matrix_builder(self.fn, weight, (rows, cols), call_kwargs)
        matrix = _coerce_to_matrix(result, (rows, cols), weight.device, weight.dtype)

        with torch.no_grad():
            weight.copy_(matrix)

        return weight

    def __repr__(self) -> str:
        """Return string representation."""
        name = getattr(self.fn, "__name__", repr(self.fn))
        return f"{self.__class__.__name__}(fn={name}, kwargs={self.kwargs})"
