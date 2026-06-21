"""Binary random initializer for input/feedback weights."""

from typing import Any

import numpy as np
import torch

from .base import InputFeedbackInitializer, _resolve_shape
from .registry import register_input_feedback


@register_input_feedback("random_binary", input_scaling=None, seed=None)
class RandomBinaryInitializer(InputFeedbackInitializer):
    """Binary random initializer for input/feedback weight matrices.

    This initializer creates binary weight matrices with values in {-1, +1}.
    Each weight is randomly chosen to be either -1 or +1, optionally scaled.

    Binary weights can be advantageous for:
    - Memory efficiency (can be stored as bits)
    - Computational efficiency (multiplication becomes addition/subtraction)
    - Improved robustness in some cases
    - Easier interpretation

    Parameters
    ----------
    input_scaling : float, optional
        Uniform magnitude knob from the shared scaling contract (see
        :class:`~resdag.init.input_feedback.InputFeedbackInitializer`). ``None``
        (the default) leaves entries in ``{-1, +1}``; a float ``s`` multiplies
        every entry by ``s`` (entries become ``{-s, +s}``), so ``max|W|`` scales
        linearly with ``s`` (``input_scaling=0.5`` halves it).
    seed : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> from resdag.init.input_feedback import RandomBinaryInitializer
    >>>
    >>> # Create binary initializer
    >>> init = RandomBinaryInitializer(input_scaling=0.5, seed=42)
    >>>
    >>> # Initialize weights
    >>> weight = torch.empty(100, 10)
    >>> init.initialize(weight)
    >>>
    >>> # All values will be either -0.5 or +0.5
    >>> unique_values = torch.unique(weight)
    >>> print(unique_values)  # tensor([-0.5, 0.5])
    """

    def __init__(
        self,
        input_scaling: float | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the RandomBinaryInitializer."""
        super().__init__(input_scaling=input_scaling, seed=seed)

    def initialize(self, weight: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Initialize weight tensor with binary random values.

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
            Initialized weight tensor with binary values
        """
        out_features, in_features = _resolve_shape(weight)
        device = weight.device
        dtype = weight.dtype

        rng = np.random.default_rng(self.seed)

        # Generate binary values {-1, +1}
        values = rng.choice([-1.0, 1.0], size=(out_features, in_features))

        # Apply the shared uniform scaling contract as the documented final transform.
        values = self._apply_scaling(values)

        # Convert to tensor and copy to weight
        weight_data = torch.from_numpy(values).to(device=device, dtype=dtype)

        with torch.no_grad():
            weight.copy_(weight_data)

        return weight

    def __repr__(self) -> str:
        return f"RandomBinaryInitializer(input_scaling={self.input_scaling}, seed={self.seed})"
