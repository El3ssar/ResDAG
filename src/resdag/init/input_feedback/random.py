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
        Uniform magnitude knob from the shared scaling contract (see
        :class:`~resdag.init.input_feedback.InputFeedbackInitializer`). ``None``
        (the default) applies no scaling, leaving entries in ``[-1, 1]``; a float
        ``s`` multiplies every entry by ``s``, so ``max|W|`` scales linearly with
        ``s`` (``input_scaling=0.5`` halves it). Controls the strength of input
        signals entering the reservoir. Typical values: 0.1-5.0.
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
    >>> # Use in ESNLayer
    >>> reservoir = ESNLayer(
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
        super().__init__(input_scaling=input_scaling, seed=seed)

    def initialize(self, weight: torch.Tensor, **kwargs) -> torch.Tensor:
        """Initialize weight tensor with uniform random values.

        The RNG is constructed from ``self.seed`` on every call, so the produced
        matrix is a pure function of ``(seed, shape)``. Repeated calls on the
        same instance with equal shapes therefore yield identical matrices.
        Pass ``seed=None`` for a fresh draw on each call.

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

        rng = np.random.default_rng(self.seed)

        # Generate random values in [-1, 1]
        values = rng.uniform(-1.0, 1.0, size=(out_features, in_features))

        # Apply the shared uniform scaling contract as the documented final transform.
        values = self._apply_scaling(values)

        # Convert to tensor and copy to weight
        weight_data = torch.from_numpy(values).to(device=device, dtype=dtype)

        with torch.no_grad():
            weight.copy_(weight_data)

        return weight

    def __repr__(self) -> str:
        return f"RandomInputInitializer(input_scaling={self.input_scaling}, seed={self.seed})"
