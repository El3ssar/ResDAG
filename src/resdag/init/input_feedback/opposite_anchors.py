"""Opposite anchors initializer for ring topologies."""

from typing import Any

import numpy as np
import torch

from .base import (
    InputFeedbackInitializer,
    _numpy_compute_dtype,
    _resolve_scaling_alias,
    _resolve_shape,
)
from .registry import register_input_feedback


@register_input_feedback("opposite_anchors", input_scaling=None, gain=None)
class OppositeAnchorsInitializer(InputFeedbackInitializer):
    """Initializer that connects each input to two opposite anchors on an n-node ring.

    Each input channel connects to two anchor nodes on opposite sides of the ring,
    with equal-magnitude bipolar weights on both anchors (``input_scaling`` normalized
    by sqrt(2), so the per-channel L2 norm equals ``input_scaling``). If the two
    anchors coincide (n=1), all weight goes to that single node.

    The first (positive) anchor of channel ``i`` is placed at ``round(i * n / m) % n``,
    spreading the ``m`` anchors evenly around the **full** ring; the second (negative)
    anchor is the diametrically opposite node ``(anchor + n // 2) % n``. Spreading over
    the full ring (rather than a semicircle) keeps the columns distinct for any
    ``in_features <= reservoir_size``, so ``W_in``/``W_fb`` stays full column rank.

    This is useful for ring/cycle topologies where you want inputs distributed evenly
    around the ring with bipolar activation patterns.

    Scaling
    -------
    For this structured initializer the magnitude statistic governed by the shared
    contract is the **per-channel L2 norm**, which equals ``input_scaling``. Hence
    ``input_scaling=0.5`` makes every channel's column have L2 norm ``0.5`` (and,
    since each column is two equal-magnitude bipolar entries, ``max|W| = 0.5/sqrt(2)``).
    The per-channel L2 norm therefore scales linearly with ``input_scaling``.

    Capacity limit
    --------------
    The ring has only ``n = reservoir_size`` nodes, so at most ``n`` channels can be
    assigned distinct anchors. ``initialize`` raises ``ValueError`` when
    ``in_features > reservoir_size`` (more channels than nodes), since duplicate
    columns would be unavoidable.

    Parameters
    ----------
    input_scaling : float, optional
        Per-channel L2 norm of each input column. Defaults to ``1.0`` when neither
        ``input_scaling`` nor the deprecated ``gain`` is given.
    gain : float, optional
        Deprecated alias for ``input_scaling`` (same meaning). Emits a
        ``DeprecationWarning``; passing both ``input_scaling`` and ``gain`` raises.

    Raises
    ------
    ValueError
        If the resolved scaling is not a positive finite float (at construction),
        if both ``input_scaling`` and ``gain`` are supplied, or if
        ``in_features > reservoir_size`` when :meth:`initialize` is called.

    Examples
    --------
    >>> from resdag.init.input_feedback import OppositeAnchorsInitializer
    >>>
    >>> init = OppositeAnchorsInitializer(input_scaling=1.0)
    >>> weight = torch.empty(100, 5)  # (reservoir_size, num_inputs)
    >>> init.initialize(weight)
    >>>
    >>> # Each input connects to two opposite points on the ring
    """

    def __init__(
        self,
        input_scaling: float | None = None,
        gain: float | None = None,
    ) -> None:
        """Initialize the OppositeAnchorsInitializer."""
        resolved = _resolve_scaling_alias(input_scaling, gain, default=1.0)
        if resolved is None or not np.isfinite(resolved) or resolved <= 0:
            raise ValueError("input_scaling must be a positive finite float.")
        super().__init__(input_scaling=resolved)

    @property
    def gain(self) -> float | None:
        """Deprecated alias for :attr:`input_scaling`."""
        return self.input_scaling

    def initialize(self, weight: torch.Tensor, **kwargs: Any) -> torch.Tensor:
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

        Raises
        ------
        ValueError
            If ``m`` or ``n`` is non-positive, or if ``m > n`` (more input channels
            than ring nodes), which would force duplicate columns.
        """
        n, m = _resolve_shape(weight)  # n=ring nodes, m=input channels
        device = weight.device
        dtype = weight.dtype
        compute_dtype = _numpy_compute_dtype(dtype)

        if m <= 0 or n <= 0:
            raise ValueError(f"m and n must be positive; received: (m={m}, n={n})")

        if m > n:
            raise ValueError(
                "opposite_anchors requires in_features <= reservoir_size: the ring has "
                f"only n={n} nodes, so at most {n} channels can be assigned distinct "
                f"anchors, but received in_features={m}. Reduce in_features or grow the "
                "reservoir."
            )

        # Build at the target precision so input_scaling / sqrt(2) keeps full
        # precision in a float64 weight instead of being truncated to float32.
        values = np.zeros((n, m), dtype=compute_dtype)
        half = n // 2

        # ``input_scaling`` is the per-channel L2 norm target (never None here:
        # the constructor resolves the alias and defaults it to 1.0).
        assert self.input_scaling is not None
        scaling = float(self.input_scaling)

        # Special case: n == 1
        if n == 1:
            # All weight on the single node; |w| == per-channel L2 norm.
            values[0, :] = scaling
        else:
            # Spread the positive anchors evenly around the *full* ring so distinct
            # channels never collide while m <= n. The negative anchor is the
            # diametrically opposite node.
            j0 = np.round(np.arange(m) * n / m).astype(int) % n
            j1 = (j0 + half) % n

            # Two equal-magnitude entries per column -> per-channel L2 norm is
            # sqrt(2) * w, so w = scaling / sqrt(2) makes the norm equal scaling.
            w = scaling / np.sqrt(2.0)
            values[j0, np.arange(m)] = w
            values[j1, np.arange(m)] = -w  # Negative for bipolar pattern

        # Convert to tensor and copy to weight
        weight_data = torch.from_numpy(values).to(device=device, dtype=dtype)

        with torch.no_grad():
            weight.copy_(weight_data)

        return weight

    def __repr__(self) -> str:
        return f"OppositeAnchorsInitializer(input_scaling={self.input_scaling})"
