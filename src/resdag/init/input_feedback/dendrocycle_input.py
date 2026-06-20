"""Dendrocycle-specific input initializer."""

from typing import Any

import numpy as np
import torch

from .base import InputFeedbackInitializer, _numpy_compute_dtype, _resolve_shape
from .registry import register_input_feedback


@register_input_feedback(
    "dendrocycle_input",
    c=None,
    C=None,
    draw_width=1.0,
    input_scaling=None,
    connectivity=None,
    seed=None,
)
class DendrocycleInputInitializer(InputFeedbackInitializer):
    """Input initializer for dendro-cycle reservoirs.

    Generates a matrix where only the core (cycle) nodes receive input connections.
    All other entries are zero. This is specific to dendrocycle topologies where
    inputs should only connect to the core ring.

    Each active connection draws a weight from ``U[-draw_width, draw_width]`` and
    is then scaled by the shared ``input_scaling`` contract. ``draw_width`` is the
    *draw half-width* (the spread of the raw distribution); ``input_scaling`` is the
    *uniform final magnitude knob* shared by every initializer. With
    ``input_scaling=0.5`` every drawn weight is halved, so ``max|W|`` scales
    linearly with ``input_scaling`` (``max|W| <= draw_width * input_scaling``).

    .. note::
       Prior to the unified scaling contract, ``input_scaling`` here meant the
       draw half-width (defaulting to ``1.0``). That role is now ``draw_width``;
       ``input_scaling`` is the uniform multiplicative transform shared with every
       other initializer and defaults to ``None`` (no scaling). To reproduce the
       old ``DendrocycleInputInitializer(input_scaling=s)`` draw, pass
       ``draw_width=s``.

    Parameters
    ----------
    c : float, optional
        Fraction of nodes forming the cycle (0 < c <= 1). Provide either c or C.
    C : int, optional
        Number of cycle (core) nodes. If provided, c is ignored.
    draw_width : float, default=1.0
        Half-width of the uniform draw ``U[-draw_width, draw_width]`` for each
        active connection.
    input_scaling : float, optional
        Uniform multiplicative scaling applied as the final transform. ``None``
        (the default) applies no scaling. ``input_scaling=0.5`` halves every
        weight, so ``max|W|`` is halved.
    connectivity : float, optional
        Unused by this initializer: dendrocycle defines its own (core-only)
        connectivity pattern, so this knob is accepted for API uniformity but
        ignored. Always ``None`` in practice.
    seed : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> from resdag.init.input_feedback import DendrocycleInputInitializer
    >>>
    >>> # Initialize for dendrocycle with 20% core nodes, weights in U[-0.5, 0.5]
    >>> init = DendrocycleInputInitializer(c=0.2, draw_width=0.5, seed=42)
    >>> weight = torch.empty(100, 8)  # (reservoir_size, num_inputs)
    >>> init.initialize(weight)
    >>>
    >>> # Only first 20 neurons (core) have non-zero weights
    """

    def __init__(
        self,
        c: float | None = None,
        C: int | None = None,
        draw_width: float = 1.0,
        input_scaling: float | None = None,
        connectivity: float | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the DendrocycleInputInitializer."""
        super().__init__(input_scaling=input_scaling, connectivity=connectivity, seed=seed)
        if (c is None) == (C is None):
            raise ValueError("Provide exactly one of c or C.")
        if not (np.isfinite(draw_width) and draw_width > 0):
            raise ValueError("draw_width must be a positive finite float.")
        self.c = c
        self.C = C
        self.draw_width = float(draw_width)

    def initialize(self, weight: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Initialize weight tensor for dendrocycle topology.

        The RNG is constructed from ``self.seed`` on every call, so the produced
        matrix is a pure function of ``(seed, shape)``. Repeated calls on the
        same instance with equal shapes therefore yield identical matrices.
        Pass ``seed=None`` for a fresh draw on each call.

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
        compute_dtype = _numpy_compute_dtype(dtype)

        rng = np.random.default_rng(self.seed)

        C = self.C
        if C is None:
            if not (0 < self.c <= 1):
                raise ValueError("c must be in (0, 1].")
            C = max(1, int(round(self.c * N)))
        if not (1 <= C <= N):
            raise ValueError("C must be in [1, N].")

        # Build at the target precision so the float64 ``rng.uniform`` draws are
        # not truncated to float32 in a float64 weight.
        values = np.zeros((N, M), dtype=compute_dtype)

        # Case 1: fewer inputs than cores
        if M <= C:
            mapping = [int(np.floor(i * M / C)) for i in range(C)]
            for core_idx, input_idx in enumerate(mapping):
                values[core_idx, input_idx] = rng.uniform(-self.draw_width, self.draw_width)
        # Case 2: more inputs than cores
        else:
            mapping = [int(np.floor(i * C / M)) for i in range(M)]
            for input_idx, core_idx in enumerate(mapping):
                values[core_idx, input_idx] = rng.uniform(-self.draw_width, self.draw_width)

        # Apply the shared uniform scaling contract as the documented final transform.
        values = self._apply_scaling(values)

        # Convert to tensor and copy to weight
        weight_data = torch.from_numpy(values).to(device=device, dtype=dtype)

        with torch.no_grad():
            weight.copy_(weight_data)

        return weight

    def __repr__(self) -> str:
        return (
            f"DendrocycleInputInitializer(c={self.c}, C={self.C}, "
            f"draw_width={self.draw_width}, input_scaling={self.input_scaling}, "
            f"seed={self.seed})"
        )
