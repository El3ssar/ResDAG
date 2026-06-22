"""Chebyshev mapping initializer for input/feedback weights."""

from typing import Any

import numpy as np
import torch

from .base import InputFeedbackInitializer, _numpy_compute_dtype, _resolve_shape
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

    The seed column (column 0) lives on the small amplitude ``[-p, p]`` while the
    chaotic columns span ``[-1, 1]``, so the raw map systematically de-weights the
    first feedback dimension (its ``max|W|`` is ~``p`` against ~``1`` elsewhere).
    To honor the shared scaling contract — whose magnitude statistic is ``max|W|``
    per column for this elementwise initializer — each column is normalized to unit
    peak amplitude (divided by its own ``max|W|``) before ``input_scaling`` applies.
    Every column then peaks at the same ``input_scaling`` (or ``1`` when
    ``input_scaling is None``), and the 1D-feedback matrix peaks at ``input_scaling``
    rather than the unscaled seed amplitude ``p``.

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
        Uniform magnitude knob from the shared scaling contract (see
        :class:`~resdag.init.input_feedback.InputFeedbackInitializer`). ``None``
        (the default) applies no scaling; a float ``s`` multiplies every entry by
        ``s`` as the documented final transform, so ``max|W|`` scales linearly
        with ``s`` (``input_scaling=0.5`` halves it).

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

        super().__init__(input_scaling=input_scaling)
        self.p = p
        self.q = q
        self.k = k

    def initialize(self, weight: torch.Tensor, **kwargs: Any) -> torch.Tensor:
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
        compute_dtype = _numpy_compute_dtype(dtype)

        # Initialize matrix at the target precision so float64 weights are not
        # silently truncated to float32 before the final widening copy.
        values = np.zeros((out_features, in_features), dtype=compute_dtype)

        # First column: sinusoidal initialization
        row_indices = np.arange(1, out_features + 1, dtype=compute_dtype)
        values[:, 0] = self.p * np.sin((row_indices / (out_features + 1)) * (np.pi / self.q))

        # Apply Chebyshev recurrence column-wise
        for j in range(1, in_features):
            values[:, j] = np.cos(self.k * np.arccos(np.clip(values[:, j - 1], -1.0, 1.0)))

        # Normalize each column to unit peak amplitude so the small-amplitude seed
        # column (~|p|) is no longer ~1/p weaker than the chaotic columns (~1). This
        # makes the magnitude statistic ``max|W|`` uniform across feedback dimensions
        # before the scaling contract applies. Columns that are entirely zero (e.g. a
        # degenerate p=0 seed) are left untouched to avoid dividing by zero.
        col_peak = np.max(np.abs(values), axis=0, keepdims=True)
        np.divide(values, col_peak, out=values, where=col_peak > 0.0)

        # Apply the shared uniform scaling contract as the documented final transform.
        values = self._apply_scaling(values)

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
