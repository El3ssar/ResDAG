"""Selective exponentiation by feature index parity."""

import torch
import torch.nn as nn


class SelectiveExponentiation(nn.Module):
    """Exponentiate even or odd feature indices based on parity.

    If ``index`` is even, even positions in the last dimension are raised to
    ``exponent``; if ``index`` is odd, odd positions are exponentiated. Other
    elements are unchanged. Used in Ott-style state-augmented ESNs.

    Parameters
    ----------
    index : int
        Parity selector: even ``index`` targets even indices; odd ``index``
        targets odd indices.
    exponent : float
        Power applied to the selected positions.

    Examples
    --------
    >>> layer = SelectiveExponentiation(index=2, exponent=2.0)
    >>> x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    >>> layer(x)
    tensor([[ 1.,  4.,  3., 16.]])
    """

    def __init__(self, index: int, exponent: float) -> None:
        """Store parity selector and exponent.

        Parameters
        ----------
        index : int
            Determines which feature indices are exponentiated (even vs odd).
        exponent : float
            Power applied to selected positions.
        """
        super().__init__()
        self.index = index
        self.exponent = exponent

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply selective exponentiation along the last dimension.

        Parameters
        ----------
        input : torch.Tensor
            Tensor of shape ``(batch, ..., features)``.

        Returns
        -------
        torch.Tensor
            Same shape as *input*; selected indices are raised to ``exponent``,
            others are unchanged.
        """
        dim = input.shape[-1]

        indices = torch.arange(dim, device=input.device)
        target_parity = self.index % 2
        mask = (indices % 2) == target_parity

        mask_float = mask.float()

        to_exponentiate = input * mask_float
        to_keep = input * (1.0 - mask_float)

        output = torch.pow(to_exponentiate, self.exponent) + to_keep

        return output

    def extra_repr(self) -> str:
        """String representation of layer configuration."""
        parity = "even" if self.index % 2 == 0 else "odd"
        return f"index={self.index}, exponent={self.exponent}, applies_to={parity}_indices"
