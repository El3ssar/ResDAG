"""Selective exponentiation layer for resdag.

Exponentiates even or odd feature indices based on a parity flag.
"""

import torch
import torch.nn as nn


class SelectiveExponentiation(nn.Module):
    """Layer that exponentiates even or odd indices based on parity.

    This layer selectively exponentiates features in a structured way:
    - If `index` is even, exponentiates even positions in the last dimension
    - If `index` is odd, exponentiates odd positions in the last dimension
    - Remaining elements are left unchanged

    Useful for applying non-linear transformations to structured feature subsets,
    particularly in architectures with alternating feature semantics.

    Args:
        index: Integer index determining parity (even vs odd exponentiation)
        exponent: The exponent value to apply to selected features

    Input Shape:
        (batch, ..., features) - any shape with at least 1 dimension

    Output Shape:
        Same as input

    Example:
        >>> layer = SelectiveExponentiation(index=2, exponent=2.0)
        >>> x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        >>> y = layer(x)
        >>> print(y)
        tensor([[ 1.,  4.,  3., 16.]])  # Even indices (0, 2) are squared
    """

    def __init__(self, index: int, exponent: float) -> None:
        """Initialize the SelectiveExponentiation layer.

        Args:
            index: Integer determining which parity to exponentiate (even/odd)
            exponent: Exponent value for transformation
        """
        super().__init__()
        self.index = index
        self.exponent = exponent

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply selective exponentiation based on feature index parity.

        Args:
            input: Input tensor of any shape

        Returns:
            Tensor where either even or odd positions (in last dim) are exponentiated
        """
        dim = input.shape[-1]

        # Create mask for even/odd indices
        # If index is even (0, 2, 4...), we want to exponentiate even positions
        # If index is odd (1, 3, 5...), we want to exponentiate odd positions
        indices = torch.arange(dim, device=input.device)

        # Mask is True where we want to exponentiate
        # index % 2 gives 0 (even) or 1 (odd)
        # We want indices with same parity as index to be exponentiated
        target_parity = self.index % 2
        mask = (indices % 2) == target_parity

        # Convert mask to float and expand to match input shape
        mask_float = mask.float()

        # Separate elements: those to exponentiate and those to keep
        to_exponentiate = input * mask_float
        to_keep = input * (1.0 - mask_float)

        # Apply exponentiation and recombine
        output = torch.pow(to_exponentiate, self.exponent) + to_keep

        return output

    def extra_repr(self) -> str:
        """String representation of layer configuration."""
        parity = "even" if self.index % 2 == 0 else "odd"
        return f"index={self.index}, exponent={self.exponent}, applies_to={parity}_indices"
