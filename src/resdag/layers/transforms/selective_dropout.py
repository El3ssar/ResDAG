"""Selective dropout layer for resdag.

Zeros out specific features based on a fixed mask, useful for analyzing
how shutting off specific neurons affects model predictions.
"""

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn


class SelectiveDropout(nn.Module):
    """Layer that zeros out specific features based on a fixed mask.

    This layer zeros out features at specified indices across all timesteps and batches.
    Unlike standard dropout which is stochastic, this layer uses a fixed mask provided
    at initialization. Useful for ablation studies and feature importance analysis.

    Args:
        mask: Boolean mask where True indicates features to zero out.
              Can be array-like (list, numpy array) or torch.Tensor of shape (features,)

    Input Shape:
        (batch, timesteps, features)

    Output Shape:
        (batch, timesteps, features)

    Example:
        >>> import numpy as np
        >>> mask = np.array([False, True, False, True])  # Drop indices 1 and 3
        >>> layer = SelectiveDropout(mask)
        >>> x = torch.randn(2, 5, 4)
        >>> y = layer(x)
        >>> # Features at indices 1 and 3 are zeroed out
    """

    def __init__(self, mask: Sequence[bool] | np.ndarray | torch.Tensor) -> None:
        """Initialize the SelectiveDropout layer.

        Args:
            mask: Boolean mask indicating which features to zero out (True = drop)

        Raises:
            ValueError: If mask is not 1-dimensional
        """
        super().__init__()

        # Convert to torch tensor
        if isinstance(mask, torch.Tensor):
            mask_tensor = mask.bool()
        else:
            mask_tensor = torch.tensor(mask, dtype=torch.bool)

        # Validate shape
        if mask_tensor.ndim != 1:
            raise ValueError(f"Mask must be 1D, but got shape {mask_tensor.shape}")

        # Register as buffer (non-trainable, but part of state_dict)
        self.register_buffer("mask", mask_tensor)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply selective dropout using the stored mask.

        Args:
            input: Input tensor of shape (batch, timesteps, features)

        Returns:
            Input tensor with masked features set to zero

        Raises:
            ValueError: If input shape doesn't match expected format
        """
        if input.dim() != 3:
            raise ValueError(
                f"Expected input shape (batch, timesteps, features), "
                f"got {input.dim()}D tensor with shape {input.shape}"
            )

        feature_dim = input.shape[-1]
        if self.mask.shape[0] != feature_dim:
            raise ValueError(
                f"Mask size ({self.mask.shape[0]}) does not match feature dimension ({feature_dim})"
            )

        # Apply mask: where mask is True, output 0; otherwise keep input
        return torch.where(self.mask, torch.zeros_like(input), input)

    def extra_repr(self) -> str:
        """String representation of layer configuration."""
        num_dropped = self.mask.sum().item()
        total_features = self.mask.shape[0]
        return f"features={total_features}, dropped={num_dropped}"
