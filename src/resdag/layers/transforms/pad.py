"""Constant-column padding transform for reservoir feature augmentation."""

import torch
import torch.nn as nn


class Pad(nn.Module):
    r"""Append a constant column to the feature dimension.

    Concatenates a fixed constant (default ``1.0``) as one extra feature at the
    end of the last dimension, mapping ``x`` of shape ``(..., F)`` to
    ``[x, padding]`` of shape ``(..., F + 1)``. This is a bias-like augmentation:
    a constant reservoir feature lets an affine readout absorb an intercept term
    directly from the state.

    .. math::

        \tilde{x} = \begin{bmatrix} x \\ \text{padding} \end{bmatrix}

    This is the ResDAG equivalent of ``Pad`` in ReservoirComputing.jl
    (SciML/ReservoirComputing.jl, ``src/states.jl``). RC.jl appends the padding
    as a trailing row of the (feature-major) state matrix; here features live on
    the last axis, so the constant is appended along ``dim=-1``.

    Parameters
    ----------
    padding : float, default=1.0
        Constant value appended as the extra feature.

    Examples
    --------
    >>> import torch
    >>> from resdag.layers.transforms import Pad
    >>>
    >>> layer = Pad()
    >>> x = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    >>> layer(x)
    tensor([[0., 1., 2., 3., 1.]])

    A custom padding value:

    >>> Pad(padding=-2.0)(torch.zeros(2, 5, 3)).shape
    torch.Size([2, 5, 4])

    See Also
    --------
    ExtendedSquare : Appends the elementwise square of every feature.
    Concatenate : Concatenates arbitrary tensors along the feature dimension.
    """

    def __init__(self, padding: float = 1.0) -> None:
        """Store the constant padding value appended in the forward pass.

        Parameters
        ----------
        padding : float, default=1.0
            Constant value appended as the trailing feature.
        """
        super().__init__()
        self.padding = padding

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Append the constant ``padding`` column along the last dimension.

        Rank-agnostic on the feature (last) dimension: ``(batch, features)``,
        ``(batch, time, features)`` and any higher rank are all accepted, so the
        layer sits in both the full-sequence and the single-step autoregressive
        ``forecast`` paths. The padding column inherits the device and dtype of
        *input*.

        Parameters
        ----------
        input : torch.Tensor
            Tensor of shape ``(..., features)``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(..., features + 1)`` with a constant ``padding``
            column appended. Gradients flow through the original features
            unchanged; the constant column carries no gradient.
        """
        pad_shape = (*input.shape[:-1], 1)
        pad_column = input.new_full(pad_shape, self.padding)
        return torch.cat([input, pad_column], dim=-1)

    def extra_repr(self) -> str:
        """String representation of layer configuration."""
        return f"padding={self.padding}"
