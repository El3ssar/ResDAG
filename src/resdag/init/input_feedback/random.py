"""Random uniform initializer for input/feedback weights."""

import numpy as np
import torch

from .base import InputFeedbackInitializer, _resolve_shape
from .registry import register_input_feedback


@register_input_feedback("random", input_scaling=None, seed=None)
class RandomInputInitializer(InputFeedbackInitializer):
    """Random initializer for feedback/input weight matrices.

    This initializer creates random weight matrices connecting inputs (feedback
    or external) to reservoir neurons. Values are sampled uniformly from [-1, 1]
    and optionally scaled by an input scaling factor.

    This is a simple, commonly used initializer for input connections. It provides
    random, unstructured connectivity from inputs to the reservoir.

    Parameters
    ----------
    input_scaling : float, optional
        Scaling factor applied to all weights. Controls the strength of input
        signals entering the reservoir. If None, no scaling is applied (weights
        remain in [-1, 1]). Typical values: 0.1-5.0. Higher values create
        stronger input signals.
    seed : int, optional
        Random seed for reproducibility. Ensures the same weight matrix is
        generated for the same seed and matrix size.

    Notes
    -----
    **Input Scaling:**

    The input_scaling parameter controls how strongly input signals affect the
    reservoir:
    - Low scaling (0.1-0.5): Weak input influence, reservoir dynamics dominate
    - Moderate scaling (0.5-1.0): Balanced input and reservoir dynamics
    - High scaling (1.0-5.0): Strong input influence, input-driven dynamics

    **Usage:**

    This initializer is typically used for:
    - Feedback weights: How feedback signals enter the reservoir
    - Input weights: How external inputs enter the reservoir (if used)

    **Best Practices:**

    - Start with input_scaling=1.0 and tune based on performance
    - Lower scaling often works better for chaotic systems
    - Higher scaling can help when inputs are weak or noisy
    - Use seed for reproducibility

    Examples
    --------
    >>> from resdag.init.input_feedback import RandomInputInitializer
    >>>
    >>> # Create initializer
    >>> init = RandomInputInitializer(input_scaling=1.0, seed=42)
    >>>
    >>> # Initialize a weight tensor
    >>> weight = torch.empty(100, 10)  # (reservoir_size, feedback_size)
    >>> init.initialize(weight)
    >>>
    >>> # Use in ReservoirLayer
    >>> reservoir = ReservoirLayer(
    ...     reservoir_size=100,
    ...     feedback_size=10,
    ...     feedback_initializer=init
    ... )
    """

    def __init__(
        self,
        input_scaling: float | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the RandomInputInitializer."""
        self.input_scaling = input_scaling
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def initialize(self, weight: torch.Tensor, **kwargs) -> torch.Tensor:
        """Initialize weight tensor with uniform random values.

        Parameters
        ----------
        weight : torch.Tensor
            Weight tensor of shape (out_features, in_features)

        Returns
        -------
        torch.Tensor
            Initialized weight tensor
        """
        out_features, in_features = _resolve_shape(weight)
        device = weight.device
        dtype = weight.dtype

        # Generate random values in [-1, 1]
        values = self.rng.uniform(-1.0, 1.0, size=(out_features, in_features))

        # Apply scaling if provided
        if self.input_scaling is not None:
            values *= self.input_scaling

        # Convert to tensor and copy to weight
        weight_data = torch.from_numpy(values).to(device=device, dtype=dtype)

        with torch.no_grad():
            weight.copy_(weight_data)

        return weight

    def __repr__(self) -> str:
        return f"RandomInputInitializer(input_scaling={self.input_scaling}, seed={self.seed})"
