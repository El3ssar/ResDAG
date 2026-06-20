"""Chessboard pattern initializer for input/feedback weights."""

import numpy as np
import torch

from .base import InputFeedbackInitializer, _numpy_compute_dtype, _resolve_shape
from .registry import register_input_feedback


@register_input_feedback("chessboard", input_scaling=None)
class ChessboardInitializer(InputFeedbackInitializer):
    """Chessboard pattern initializer with alternating {-1, +1} values.

    Creates a deterministic checkerboard pattern where adjacent elements
    alternate in sign. This creates a structured, high-frequency pattern
    that can be useful for certain reservoir dynamics.

    The pattern is: W[i, j] = (-1)^(i+j)

    Parameters
    ----------
    input_scaling : float, optional
        Uniform magnitude knob from the shared scaling contract (see
        :class:`~resdag.init.input_feedback.InputFeedbackInitializer`). ``None``
        (the default) leaves entries in ``{-1, +1}``; a float ``s`` multiplies
        every entry by ``s`` (entries become ``{-s, +s}``), so ``max|W|`` scales
        linearly with ``s`` (``input_scaling=0.5`` halves it).

    Examples
    --------
    >>> from resdag.init.input_feedback import ChessboardInitializer
    >>>
    >>> init = ChessboardInitializer(input_scaling=0.5)
    >>> weight = torch.empty(5, 10)
    >>> init.initialize(weight)
    >>>
    >>> # Creates alternating pattern:
    >>> # [[ 0.5, -0.5,  0.5, -0.5, ...],
    >>> #  [-0.5,  0.5, -0.5,  0.5, ...],
    >>> #  [ 0.5, -0.5,  0.5, -0.5, ...],
    >>> #  ...]
    """

    def __init__(self, input_scaling: float | None = None) -> None:
        """Initialize the ChessboardInitializer."""
        super().__init__(input_scaling=input_scaling)

    def initialize(self, weight: torch.Tensor, **kwargs) -> torch.Tensor:
        """Initialize weight tensor with chessboard pattern.

        Parameters
        ----------
        weight : torch.Tensor
            Weight tensor of shape (out_features, in_features)

        Returns
        -------
        torch.Tensor
            Initialized weight tensor with chessboard pattern
        """
        out_features, in_features = _resolve_shape(weight)
        device = weight.device
        dtype = weight.dtype
        compute_dtype = _numpy_compute_dtype(dtype)

        # Create chessboard pattern at the target precision so any scaling that
        # is not float32-representable survives a float64 weight intact.
        i = np.arange(out_features)[:, None]
        j = np.arange(in_features)[None, :]
        values = ((-1.0) ** (i + j)).astype(compute_dtype)

        # Apply the shared uniform scaling contract as the documented final transform.
        values = self._apply_scaling(values)

        # Convert to tensor and copy to weight
        weight_data = torch.from_numpy(values).to(device=device, dtype=dtype)

        with torch.no_grad():
            weight.copy_(weight_data)

        return weight

    def __repr__(self) -> str:
        return f"ChessboardInitializer(input_scaling={self.input_scaling})"
