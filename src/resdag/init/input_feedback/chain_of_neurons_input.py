"""Chain-of-neurons specific input initializer."""

from typing import Sequence

import numpy as np
import torch

from .base import InputFeedbackInitializer, _resolve_shape
from .registry import register_input_feedback


@register_input_feedback("chain_of_neurons_input", features=None, weights=1.0)
class ChainOfNeuronsInputInitializer(InputFeedbackInitializer):
    """Input initializer for chain-of-neurons reservoirs.

    For reservoirs organized as multiple parallel chains, this initializer
    connects each input to the first neuron of its corresponding chain.

    Parameters
    ----------
    features : int
        Number of chains (must equal number of inputs).
    weights : float or Sequence[float], default=1.0
        Either a single float (same weight for all input→chain pairs) or
        a sequence of floats (one weight per input/chain).

    Examples
    --------
    >>> from resdag.init.input_feedback import ChainOfNeuronsInputInitializer
    >>>
    >>> # 3 chains, uniform weight
    >>> init = ChainOfNeuronsInputInitializer(features=3, weights=1.0)
    >>> weight = torch.empty(150, 3)  # (reservoir_size, num_inputs)
    >>> init.initialize(weight)
    >>>
    >>> # Each input connects only to the first neuron of its chain
    """

    def __init__(
        self,
        features: int,
        weights: float | Sequence[float] = 1.0,
    ):
        """Initialize the ChainOfNeuronsInputInitializer."""
        if features is None:
            raise ValueError("'features' must be provided.")
        if features < 1:
            raise ValueError(f"'features' must be >= 1, got {features}.")
        self.features = features
        self.weights = weights

    def initialize(self, weight: torch.Tensor, **kwargs) -> torch.Tensor:
        """Initialize weight tensor for chain-of-neurons topology.

        Parameters
        ----------
        weight : torch.Tensor
            Weight tensor of shape (out_features, in_features)
            out_features = reservoir_size (units)
            in_features = num_inputs (must equal features)

        Returns
        -------
        torch.Tensor
            Initialized weight tensor
        """
        units, input_dim = _resolve_shape(weight)
        device = weight.device
        dtype = weight.dtype

        if input_dim != self.features:
            raise ValueError(
                f"input_dim ({input_dim}) must equal 'features' ({self.features}) "
                "to have one input per chain."
            )
        if units % self.features != 0:
            raise ValueError(
                f"Number of units ({units}) must be a multiple of 'features' "
                f"({self.features}) to align chains with inputs."
            )

        values = np.zeros((units, input_dim), dtype=np.float32)

        # Resolve per-input weights
        if isinstance(self.weights, (list, tuple, np.ndarray)):
            if len(self.weights) != input_dim:
                raise ValueError(
                    "When 'weights' is a sequence, its length must equal input_dim; "
                    f"got len(weights)={len(self.weights)}, input_dim={input_dim}."
                )
            in_weights = [float(w) for w in self.weights]
        else:
            w = float(self.weights)
            in_weights = [w] * input_dim

        block_len = units // self.features

        # Deterministic: input i → first unit of chain i
        for i in range(input_dim):
            start = i * block_len
            values[start, i] = in_weights[i]

        # Convert to tensor and copy to weight
        weight_data = torch.from_numpy(values).to(device=device, dtype=dtype)

        with torch.no_grad():
            weight.copy_(weight_data)

        return weight

    def __repr__(self) -> str:
        return f"ChainOfNeuronsInputInitializer(features={self.features}, weights={self.weights})"
