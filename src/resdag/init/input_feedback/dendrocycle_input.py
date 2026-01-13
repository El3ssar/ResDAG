"""Dendrocycle-specific input initializer."""

import numpy as np
import torch

from .base import InputFeedbackInitializer, _resolve_shape
from .registry import register_input_feedback


@register_input_feedback("dendrocycle_input", c=None, C=None, input_scaling=1.0, seed=None)
class DendrocycleInputInitializer(InputFeedbackInitializer):
    """Input initializer for dendro-cycle reservoirs.

    Generates a matrix where only the core (cycle) nodes receive input connections.
    All other entries are zero. This is specific to dendrocycle topologies where
    inputs should only connect to the core ring.

    Parameters
    ----------
    c : float, optional
        Fraction of nodes forming the cycle (0 < c <= 1). Provide either c or C.
    C : int, optional
        Number of cycle (core) nodes. If provided, c is ignored.
    input_scaling : float, default=1.0
        Half-width of the uniform distribution U[-input_scaling, input_scaling].
    seed : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> from resdag.init.input_feedback import DendrocycleInputInitializer
    >>>
    >>> # Initialize for dendrocycle with 20% core nodes
    >>> init = DendrocycleInputInitializer(c=0.2, input_scaling=0.5, seed=42)
    >>> weight = torch.empty(100, 8)  # (reservoir_size, num_inputs)
    >>> init.initialize(weight)
    >>>
    >>> # Only first 20 neurons (core) have non-zero weights
    """

    def __init__(
        self,
        c: float | None = None,
        C: int | None = None,
        input_scaling: float = 1.0,
        seed: int | None = None,
    ) -> None:
        """Initialize the DendrocycleInputInitializer."""
        if (c is None) == (C is None):
            raise ValueError("Provide exactly one of c or C.")
        self.c = c
        self.C = C
        self.input_scaling = input_scaling
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def initialize(self, weight: torch.Tensor, **kwargs) -> torch.Tensor:
        """Initialize weight tensor for dendrocycle topology.

        Parameters
        ----------
        weight : torch.Tensor
            Weight tensor of shape (out_features, in_features)
            out_features = reservoir_size (N)
            in_features = num_inputs (M)

        Returns
        -------
        torch.Tensor
            Initialized weight tensor (only core nodes have non-zero weights)
        """
        N, M = _resolve_shape(weight)
        device = weight.device
        dtype = weight.dtype

        C = self.C
        if C is None:
            if not (0 < self.c <= 1):
                raise ValueError("c must be in (0, 1].")
            C = max(1, int(round(self.c * N)))
        if not (1 <= C <= N):
            raise ValueError("C must be in [1, N].")

        values = np.zeros((N, M), dtype=np.float32)

        # Case 1: fewer inputs than cores
        if M <= C:
            mapping = [int(np.floor(i * M / C)) for i in range(C)]
            for core_idx, input_idx in enumerate(mapping):
                values[core_idx, input_idx] = self.rng.uniform(
                    -self.input_scaling, self.input_scaling
                )
        # Case 2: more inputs than cores
        else:
            mapping = [int(np.floor(i * C / M)) for i in range(M)]
            for input_idx, core_idx in enumerate(mapping):
                values[core_idx, input_idx] = self.rng.uniform(
                    -self.input_scaling, self.input_scaling
                )

        # Convert to tensor and copy to weight
        weight_data = torch.from_numpy(values).to(device=device, dtype=dtype)

        with torch.no_grad():
            weight.copy_(weight_data)

        return weight

    def __repr__(self) -> str:
        return (
            f"DendrocycleInputInitializer(c={self.c}, C={self.C}, "
            f"input_scaling={self.input_scaling}, seed={self.seed})"
        )
