"""Pseudo-diagonal initializer for structured input connections."""

import numpy as np
import torch

from .base import InputFeedbackInitializer, _resolve_shape
from .registry import register_input_feedback


@register_input_feedback("pseudo_diagonal", input_scaling=None, binarize=False, seed=None)
class PseudoDiagonalInitializer(InputFeedbackInitializer):
    """Pseudo-diagonal initializer for structured input connections.

    This initializer creates a structured input weight matrix where each input
    dimension connects to a contiguous block of reservoir neurons. This creates
    a "pseudo-diagonal" pattern that can improve input-to-reservoir mapping,
    especially when input dimensions have semantic meaning.

    The connectivity pattern ensures:
    - Each reservoir neuron receives input from exactly one input dimension
    - Each input dimension connects to approximately N/D reservoir neurons
      (where N=reservoir size, D=input dimension)
    - Connections form contiguous blocks (not random)

    Parameters
    ----------
    input_scaling : float, optional
        Scaling factor applied to all weights. If None, no scaling is applied
        (weights remain in [-1, 1]).
    binarize : bool, default=False
        Whether to binarize weights to {-input_scaling, input_scaling} instead
        of uniform distribution.
    seed : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> from resdag.init.input_feedback import PseudoDiagonalInitializer
    >>>
    >>> init = PseudoDiagonalInitializer(input_scaling=1.0, binarize=False, seed=42)
    >>> weight = torch.empty(200, 5)  # (reservoir_size, input_dim)
    >>> init.initialize(weight)
    >>>
    >>> # Each of the 5 inputs connects to a contiguous block of ~40 neurons
    """

    def __init__(
        self,
        input_scaling: float | None = None,
        binarize: bool = False,
        seed: int | None = None,
    ) -> None:
        """Initialize the PseudoDiagonalInitializer."""
        self.input_scaling = input_scaling
        self.binarize = binarize
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def initialize(self, weight: torch.Tensor, **kwargs) -> torch.Tensor:
        """Initialize weight tensor with pseudo-diagonal structure.

        Parameters
        ----------
        weight : torch.Tensor
            Weight tensor of shape (out_features, in_features)

        Returns
        -------
        torch.Tensor
            Initialized weight tensor with block structure
        """
        out_features, in_features = _resolve_shape(weight)
        device = weight.device
        dtype = weight.dtype

        # Create sparse block-diagonal structure
        values = np.zeros((out_features, in_features), dtype=np.float32)

        # Case 1: out_features >= in_features (typical for reservoirs)
        # Each input feature gets a contiguous block of output neurons
        if out_features >= in_features:
            block_size = out_features // in_features
            remainder = out_features % in_features

            start_row = 0
            for col in range(in_features):
                # Determine block size for this column
                end_row = start_row + block_size + (1 if col < remainder else 0)

                # Fill this block
                block_values = self.rng.uniform(-1.0, 1.0, size=(end_row - start_row,))
                if self.binarize:
                    block_values = np.sign(block_values)

                values[start_row:end_row, col] = block_values
                start_row = end_row

        # Case 2: out_features < in_features (rare case)
        # Each output neuron gets input from a contiguous block of input features
        else:
            block_size = in_features // out_features
            remainder = in_features % out_features

            start_col = 0
            for row in range(out_features):
                # Determine block size for this row
                end_col = start_col + block_size + (1 if row < remainder else 0)

                # Fill this block
                block_values = self.rng.uniform(-1.0, 1.0, size=(end_col - start_col,))
                if self.binarize:
                    block_values = np.sign(block_values)

                values[row, start_col:end_col] = block_values
                start_col = end_col

        # Apply scaling if provided
        if self.input_scaling is not None:
            values *= self.input_scaling

        # Convert to tensor and copy to weight
        weight_data = torch.from_numpy(values).to(device=device, dtype=dtype)

        with torch.no_grad():
            weight.copy_(weight_data)

        return weight

    def __repr__(self) -> str:
        return (
            f"PseudoDiagonalInitializer(input_scaling={self.input_scaling}, "
            f"binarize={self.binarize}, seed={self.seed})"
        )
