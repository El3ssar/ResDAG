"""
Concatenation Layer
===================

This module provides :class:`Concatenate`, a layer for combining multiple
tensor inputs along the feature dimension.
"""

import torch
import torch.nn as nn


class Concatenate(nn.Module):
    """
    Concatenate multiple tensors along the feature (last) dimension.

    This layer takes any number of input tensors and concatenates them
    along the last dimension. Useful for combining outputs from parallel
    branches in multi-reservoir or ensemble architectures.

    All input tensors must have the same shape except for the last
    dimension (features).

    Attributes
    ----------
    None
        This layer has no learnable parameters.

    Examples
    --------
    Basic concatenation:

    >>> import torch
    >>> from resdag.layers.custom import Concatenate
    >>>
    >>> concat = Concatenate()
    >>> x1 = torch.randn(32, 50, 100)  # (batch, time, features1)
    >>> x2 = torch.randn(32, 50, 200)  # (batch, time, features2)
    >>> y = concat(x1, x2)
    >>> print(y.shape)
    torch.Size([32, 50, 300])

    Multiple inputs:

    >>> x3 = torch.randn(32, 50, 50)
    >>> y = concat(x1, x2, x3)
    >>> print(y.shape)
    torch.Size([32, 50, 350])

    See Also
    --------
    FeaturePartitioner : Splits features into overlapping partitions.
    """

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """
        Concatenate input tensors along the last dimension.

        Parameters
        ----------
        *inputs : torch.Tensor
            Variable number of input tensors to concatenate. All tensors
            must have the same shape except for the last dimension.

        Returns
        -------
        torch.Tensor
            Concatenated tensor with combined feature dimension.

        Raises
        ------
        RuntimeError
            If tensors have incompatible shapes for concatenation.

        Examples
        --------
        >>> concat = Concatenate()
        >>> a = torch.randn(4, 10, 20)
        >>> b = torch.randn(4, 10, 30)
        >>> result = concat(a, b)
        >>> print(result.shape)
        torch.Size([4, 10, 50])
        """
        return torch.cat(inputs, dim=-1)
