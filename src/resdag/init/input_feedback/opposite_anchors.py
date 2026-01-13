"""Opposite anchors initializer for ring topologies."""

import numpy as np
import torch

from .base import InputFeedbackInitializer, _resolve_shape
from .registry import register_input_feedback


@register_input_feedback("opposite_anchors", gain=1.0)
class OppositeAnchorsInitializer(InputFeedbackInitializer):
    """Initializer that connects each input to two opposite anchors on an n-node ring.

    Each input channel connects to two anchor nodes on opposite sides of the ring,
    with equal weights on both anchors (gain normalized by sqrt(2), so total channel
    energy equals 'gain'). If the two anchors coincide (n=1), all weight goes to that
    single node.

    This is useful for ring/cycle topologies where you want inputs to be distributed
    evenly around the ring with bipolar activation patterns.

    Parameters
    ----------
    gain : float, default=1.0
        Global input gain per channel.

    Examples
    --------
    >>> from resdag.init.input_feedback import OppositeAnchorsInitializer
    >>>
    >>> init = OppositeAnchorsInitializer(gain=1.0)
    >>> weight = torch.empty(100, 5)  # (reservoir_size, num_inputs)
    >>> init.initialize(weight)
    >>>
    >>> # Each input connects to two opposite points on the ring
    """

    def __init__(self, gain: float = 1.0) -> None:
        """Initialize the OppositeAnchorsInitializer."""
        if not np.isfinite(gain) or gain <= 0:
            raise ValueError("gain must be a positive finite float.")
        self.gain = float(gain)

    def initialize(self, weight: torch.Tensor, **kwargs) -> torch.Tensor:
        """Initialize weight tensor with opposite anchor pattern.

        Parameters
        ----------
        weight : torch.Tensor
            Weight tensor of shape (out_features, in_features)
            out_features = number of ring nodes (n)
            in_features = number of input channels (m)

        Returns
        -------
        torch.Tensor
            Initialized weight tensor
        """
        n, m = _resolve_shape(weight)  # n=ring nodes, m=input channels
        device = weight.device
        dtype = weight.dtype

        if m <= 0 or n <= 0:
            raise ValueError(f"m and n must be positive; received: (m={m}, n={n})")

        values = np.zeros((n, m), dtype=np.float32)
        half = n // 2

        # Special case: n == 1
        if n == 1:
            values[0, :] = self.gain
        else:
            # Evenly spaced anchors on the semicircle
            j0 = np.floor((np.arange(m) + 0.5) * half / m).astype(int)
            j1 = (j0 + half) % n

            w = self.gain / np.sqrt(2.0)
            values[j0, np.arange(m)] = w
            values[j1, np.arange(m)] = -w  # Negative for bipolar pattern

        # Convert to tensor and copy to weight
        weight_data = torch.from_numpy(values).to(device=device, dtype=dtype)

        with torch.no_grad():
            weight.copy_(weight_data)

        return weight

    def __repr__(self) -> str:
        return f"OppositeAnchorsInitializer(gain={self.gain})"
