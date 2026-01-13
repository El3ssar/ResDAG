"""Chebyshev mapping initializer for input/feedback weights."""

import numpy as np
import torch

from .base import InputFeedbackInitializer, _resolve_shape
from .registry import register_input_feedback


@register_input_feedback("chebyshev", p=0.3, q=5.9, k=3.8, input_scaling=None)
class ChebyshevInitializer(InputFeedbackInitializer):
    """Chebyshev mapping initializer for deterministic chaotic initialization.

    This initializer constructs a weight matrix based on the Chebyshev polynomial
    map, ensuring structured, chaotic initialization while maintaining a controlled
    range.

    The Chebyshev polynomial recurrence exhibits **deterministic chaos**, making it
    a structured alternative to purely random weight initialization. This enhances
    the richness of how the input signal is connected to the reservoir neurons.

    The Chebyshev map is applied column-wise:
    - First column: W[:, 0] = p * sin((i / (rows+1)) * (π / q))
    - Subsequent columns: W[:, j] = cos(k * arccos(W[:, j-1]))

    where k controls chaotic behavior (optimal range: 2 < k < 4).

    Parameters
    ----------
    p : float, default=0.3
        Scaling factor for the initial sinusoidal weights. Should be in (0, 1).
    q : float, default=5.9
        Parameter controlling the initial sinusoidal distribution.
    k : float, default=3.8
        Control parameter of the Chebyshev map. Must be in (2, 4) for chaotic
        behavior.
    input_scaling : float, optional
        Additional scaling factor applied after generation.

    Raises
    ------
    ValueError
        If k is not in the valid range (2, 4).

    References
    ----------
    M. Xie, Q. Wang, and S. Yu, "Time Series Prediction of ESN Based on Chebyshev
    Mapping and Strongly Connected Topology," Neural Process Lett, vol. 56, no. 1,
    p. 30, Feb. 2024.

    Examples
    --------
    >>> from resdag.init.input_feedback import ChebyshevInitializer
    >>>
    >>> init = ChebyshevInitializer(p=0.3, k=3.5, input_scaling=0.8)
    >>> weight = torch.empty(100, 10)
    >>> init.initialize(weight)
    """

    def __init__(
        self,
        p: float = 0.3,
        q: float = 5.9,
        k: float = 3.8,
        input_scaling: float | None = None,
    ) -> None:
        """Initialize the ChebyshevInitializer."""
        if not (2.0 < k < 4.0):
            raise ValueError(f"Parameter k={k} must be in range (2, 4) for chaotic behavior")

        self.p = p
        self.q = q
        self.k = k
        self.input_scaling = input_scaling

    def initialize(self, weight: torch.Tensor, **kwargs) -> torch.Tensor:
        """Initialize weight tensor using Chebyshev mapping.

        Parameters
        ----------
        weight : torch.Tensor
            Weight tensor of shape (out_features, in_features)

        Returns
        -------
        torch.Tensor
            Initialized weight tensor with Chebyshev structure
        """
        out_features, in_features = _resolve_shape(weight)
        device = weight.device
        dtype = weight.dtype

        # Initialize matrix
        values = np.zeros((out_features, in_features), dtype=np.float32)

        # First column: sinusoidal initialization
        row_indices = np.arange(1, out_features + 1, dtype=np.float32)
        values[:, 0] = self.p * np.sin((row_indices / (out_features + 1)) * (np.pi / self.q))

        # Apply Chebyshev recurrence column-wise
        for j in range(1, in_features):
            values[:, j] = np.cos(self.k * np.arccos(np.clip(values[:, j - 1], -1.0, 1.0)))

        # Apply additional scaling if provided
        if self.input_scaling is not None:
            values *= self.input_scaling

        # Convert to tensor and copy to weight
        weight_data = torch.from_numpy(values).to(device=device, dtype=dtype)

        with torch.no_grad():
            weight.copy_(weight_data)

        return weight

    def __repr__(self) -> str:
        return (
            f"ChebyshevInitializer(p={self.p}, q={self.q}, k={self.k}, "
            f"input_scaling={self.input_scaling})"
        )
