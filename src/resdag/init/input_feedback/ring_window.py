"""Ring window initializer for dendrocycle+chords topologies."""

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


@register_input_feedback(
    "ring_window",
    c=None,
    window=None,
    taper="flat",
    signed="allpos",
    input_scaling=None,
    gain=None,
)
class RingWindowInputInitializer(InputFeedbackInitializer):
    """Deterministic windowed input initializer for ring-based topologies.

    Feeds each input channel into a contiguous window on the core ring of a
    dendrocycle(+chords) reservoir. Only the first C = round(c * n) columns (core)
    receive nonzeros.

    Scaling
    -------
    For this structured initializer the magnitude statistic governed by the shared
    contract is the **per-channel L2 norm**, which equals ``input_scaling``: each
    channel's window vector is renormalized so its L2 norm is exactly
    ``input_scaling``. Hence ``input_scaling=0.5`` makes every channel's column have
    L2 norm ``0.5``; the per-channel L2 norm scales linearly with ``input_scaling``.

    Parameters
    ----------
    c : float
        Fraction of core ring nodes. First round(c*n) columns are core.
    window : int or float
        If int >= 1: number of core nodes per channel window.
        If float in (0, 1]: fraction of C per channel window (rounded >= 1).
    taper : {"flat", "triangle", "cosine"}, default="flat"
        Weight profile within a channel's window centered on its center index.
    signed : {"allpos", "alt_ring", "alt_inputs"}, default="allpos"
        Sign policy for weights.
    input_scaling : float, optional
        Per-channel L2-norm after taper/sign are applied. Defaults to ``1.0`` when
        neither ``input_scaling`` nor the deprecated ``gain`` is given.
    gain : float, optional
        Deprecated alias for ``input_scaling`` (same meaning). Emits a
        ``DeprecationWarning``; passing both ``input_scaling`` and ``gain`` raises.

    Examples
    --------
    >>> from resdag.init.input_feedback import RingWindowInputInitializer
    >>>
    >>> init = RingWindowInputInitializer(
    ...     c=0.5, window=10, taper="cosine", signed="alt_ring", input_scaling=1.0
    ... )
    >>> weight = torch.empty(100, 5)  # (reservoir_size, num_inputs)
    >>> init.initialize(weight)
    >>>
    >>> # Each input connects to a windowed region of the core ring
    """

    def __init__(
        self,
        c: float,
        window: int | float,
        taper: str = "flat",
        signed: str = "allpos",
        input_scaling: float | None = None,
        gain: float | None = None,
    ) -> None:
        """Initialize the RingWindowInputInitializer."""
        if not (0 < c <= 1):
            raise ValueError("c must be in (0,1].")
        self.c = float(c)

        self.window: int | float
        if isinstance(window, int):
            if window < 1:
                raise ValueError("window int must be >= 1.")
            self.window = window
            self.window_is_frac = False
        elif isinstance(window, float):
            if not (0 < window <= 1):
                raise ValueError("window float must be in (0,1].")
            self.window = float(window)
            self.window_is_frac = True
        else:
            raise TypeError("window must be int or float.")

        if taper not in {"flat", "triangle", "cosine"}:
            raise ValueError("taper must be 'flat', 'triangle', or 'cosine'.")
        if signed not in {"allpos", "alt_ring", "alt_inputs"}:
            raise ValueError("signed must be 'allpos', 'alt_ring', or 'alt_inputs'.")

        resolved = _resolve_scaling_alias(input_scaling, gain, default=1.0)
        if resolved is None or not (np.isfinite(resolved) and resolved > 0):
            raise ValueError("input_scaling must be a positive finite float.")
        super().__init__(input_scaling=resolved)

        self.taper = taper
        self.signed = signed

    @property
    def gain(self) -> float | None:
        """Deprecated alias for :attr:`input_scaling`."""
        return self.input_scaling

    def _window_size(self, C: int) -> int:
        """Compute window size."""
        if self.window_is_frac:
            W = int(round(self.window * C))
        else:
            W = int(self.window)
        return max(1, min(C, W))

    @staticmethod
    def _taper_vector(W: int, kind: str) -> np.ndarray:
        """Generate taper vector."""
        if W <= 2:
            return np.ones(W, dtype=np.float64)

        idx = np.arange(W, dtype=np.float64)
        if kind == "flat":
            w = np.ones(W, dtype=np.float64)
        elif kind == "triangle":
            center = (W - 1) / 2.0
            w = 1.0 - np.abs(idx - center) / center
        elif kind == "cosine":
            center = (W - 1) / 2.0
            x = (idx - center) / center
            w = 0.5 * (1.0 + np.cos(np.pi * x))
        else:
            raise RuntimeError("unreachable")
        return np.clip(w, 0.0, None)

    def initialize(self, weight: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Initialize weight tensor with ring window pattern.

        Parameters
        ----------
        weight : torch.Tensor
            Weight tensor of shape (out_features, in_features)
            out_features = reservoir_size (n)
            in_features = num_inputs (m)

        Returns
        -------
        torch.Tensor
            Initialized weight tensor
        """
        n, m = _resolve_shape(weight)
        device = weight.device
        dtype = weight.dtype
        compute_dtype = _numpy_compute_dtype(dtype)

        if m <= 0 or n <= 0:
            raise ValueError("m and n must be positive.")

        C = max(1, int(round(self.c * n)))  # core size
        W = self._window_size(C)

        # ``input_scaling`` is the per-channel L2 norm target (never None here:
        # the constructor resolves the alias and defaults it to 1.0).
        assert self.input_scaling is not None
        scaling = float(self.input_scaling)

        # Build at the target precision so the float64 taper/norm computation is
        # not truncated to float32 before reaching a float64 weight.
        values = np.zeros((n, m), dtype=compute_dtype)
        base = self._taper_vector(W, self.taper).astype(compute_dtype)

        def window_indices(center: int) -> np.ndarray:
            start = center - (W // 2)
            return (np.arange(start, start + W) % C).astype(int)

        for k in range(m):
            start = int(np.floor(k * C / m)) % C
            center = (start + (W - 1) // 2) % C

            cols_core = window_indices(center)
            row_vals = base.copy()

            signs: float | np.ndarray
            if self.signed == "allpos":
                signs = 1.0
            elif self.signed == "alt_ring":
                signs = (1.0 - 2.0 * (cols_core % 2)).astype(compute_dtype)
            else:  # "alt_inputs"
                signs = 1.0 if (k % 2 == 0) else -1.0

            row_vals = row_vals * signs
            norm = float(np.linalg.norm(row_vals))
            scaled = row_vals if norm == 0.0 else (scaling / norm) * row_vals

            values[cols_core, k] = scaled  # Only core; rest stay zero

        # Convert to tensor and copy to weight
        weight_data = torch.from_numpy(values).to(device=device, dtype=dtype)

        with torch.no_grad():
            weight.copy_(weight_data)

        return weight

    def __repr__(self) -> str:
        return (
            f"RingWindowInputInitializer(c={self.c}, window={self.window}, "
            f"taper={self.taper}, signed={self.signed}, input_scaling={self.input_scaling})"
        )
