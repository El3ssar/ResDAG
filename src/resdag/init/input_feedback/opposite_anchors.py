"""Opposite anchors initializer for ring topologies."""

import numpy as np
import torch

from .base import InputFeedbackInitializer, _numpy_compute_dtype, _resolve_shape
from .registry import register_input_feedback


@register_input_feedback("opposite_anchors", gain=1.0)
class OppositeAnchorsInitializer(InputFeedbackInitializer):
    """Initializer that connects each input to two opposite anchors on an n-node ring.

    Each input channel connects to two anchor nodes on opposite sides of the ring,
    with equal-magnitude bipolar weights on both anchors (gain normalized by sqrt(2),
    so total channel energy equals 'gain'). If the two anchors coincide (n=1), all
    weight goes to that single node.

    The first (positive) anchor of channel ``i`` is placed at ``round(i * n / m) % n``,
    spreading the ``m`` anchors evenly around the **full** ring; the second (negative)
    anchor is the diametrically opposite node ``(anchor + n // 2) % n``. Spreading over
    the full ring (rather than a semicircle) keeps the columns distinct for any
    ``in_features <= reservoir_size``, so ``W_in``/``W_fb`` stays full column rank.

    This is useful for ring/cycle topologies where you want inputs distributed evenly
    around the ring with bipolar activation patterns.

    Capacity limit
    --------------
    The ring has only ``n = reservoir_size`` nodes, so at most ``n`` channels can be
    assigned distinct anchors. ``initialize`` raises ``ValueError`` when
    ``in_features > reservoir_size`` (more channels than nodes), since duplicate
    columns would be unavoidable.

    Parameters
    ----------
    gain : float, default=1.0
        Global input gain per channel.

    Raises
    ------
    ValueError
        If ``gain`` is not a positive finite float (at construction), or if
        ``in_features > reservoir_size`` when :meth:`initialize` is called.

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

        # Build at the target precision so gain / sqrt(2) keeps full precision
        # in a float64 weight instead of being truncated to float32.
        values = np.zeros((n, m), dtype=compute_dtype)
        half = n // 2

        # Special case: n == 1
        if n == 1:
            values[0, :] = self.gain
        else:
            # Spread the positive anchors evenly around the *full* ring so distinct
            # channels never collide while m <= n. The negative anchor is the
            # diametrically opposite node.
            j0 = np.round(np.arange(m) * n / m).astype(int) % n
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
