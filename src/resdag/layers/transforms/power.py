"""Element-wise power transform for reservoir feature augmentation."""

import torch
import torch.nn as nn


class Power(nn.Module):
    """Exponentiate every feature to a fixed power.

    Applies ``torch.pow`` element-wise along the last dimension. Used in
    power-augmented ESN architectures to enrich reservoir states before
    readout.

    Parameters
    ----------
    exponent : float
        Power applied to each element of the input tensor.

    Examples
    --------
    >>> layer = Power(exponent=2.0)
    >>> x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    >>> layer(x)
    tensor([[ 1.,  4.,  9., 16.]])
    """

    def __init__(self, exponent: float) -> None:
        """Store the exponent used in the forward pass.

        Parameters
        ----------
        exponent : float
            Value passed to ``torch.pow`` for every element.
        """
        super().__init__()
        self.exponent = exponent

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Raise the input to ``self.exponent``.

        Parameters
        ----------
        input : torch.Tensor
            Tensor of shape ``(batch, ..., features)``.

        Returns
        -------
        torch.Tensor
            Same shape as *input*, with each element raised to ``exponent``.
        """
        return torch.pow(input, self.exponent)

    def extra_repr(self) -> str:
        """String representation of layer configuration."""
        return f"exponent={self.exponent}"
