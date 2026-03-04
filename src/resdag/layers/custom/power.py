"""Selective exponentiation layer for resdag.

Exponentiates the input to a power element-wise.
"""

import torch
import torch.nn as nn


class Power(nn.Module):
    """Layer that exponentiates to a power.

    This layer exponentiates the input to a power element-wise.

    Args:
        exponent: The exponent value to apply to the input

    Input Shape:
        (batch, ..., features) - any shape with at least 1 dimension

    Output Shape:
        Same as input

    Example:
        >>> layer = Power(exponent=2.0)
        >>> x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        >>> y = layer(x)
        >>> print(y)
        tensor([[ 1.,  4.,  9., 16.]])  # All elements are squared
    """

    def __init__(self, exponent: float) -> None:
        """Initialize the SelectiveExponentiation layer.

        Args:
            index: Integer determining which parity to exponentiate (even/odd)
            exponent: Exponent value for transformation
        """
        super().__init__()
        self.exponent = exponent

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply selective exponentiation based on feature index parity.

        Args:
            input: Input tensor of any shape

        Returns:
            Tensor where either even or odd positions (in last dim) are exponentiated
        """
        return torch.pow(input, self.exponent)

    def extra_repr(self) -> str:
        """String representation of layer configuration."""
        return f"exponent={self.exponent}"
