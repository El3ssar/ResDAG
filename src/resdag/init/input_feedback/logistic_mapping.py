"""Logistic-mapping input initializer (Wang et al. 2022; RC.jl ``logistic_mapping``)."""

from typing import Any

import numpy as np
import torch

from .base import InputFeedbackInitializer, _numpy_compute_dtype, _resolve_shape
from .registry import register_input_feedback


@register_input_feedback(
    "logistic_mapping",
    amplitude=0.3,
    sine_divisor=5.9,
    logistic_parameter=3.7,
    input_scaling=None,
)
class LogisticMappingInitializer(InputFeedbackInitializer):
    """Logistic-map input initializer (RC.jl ``logistic_mapping`` [Wang2022]).

    A deterministic-chaos initializer, structurally the logistic-map sibling of
    :class:`~resdag.init.input_feedback.ChebyshevInitializer` but generated
    **row-wise** (down the reservoir axis) rather than column-wise. The first row
    is a sine seed and each subsequent row is one logistic-map iteration of the
    row above.

    Construction (mirrors RC.jl ``logistic_mapping``):

    - First row: ``W[0, j] = amplitude * sin((j + 1) * pi / (sine_divisor *
      in_size))`` for column ``j`` (RC.jl is 1-indexed, so column ``j`` uses
      ``j + 1``).
    - Subsequent rows: ``W[i, j] = logistic_parameter * W[i-1, j] * (1 -
      W[i-1, j])``.

    Unlike Chebyshev, entries are **not** in ``[-1, 1]`` — the logistic orbit
    lives in ``[0, 1]`` for the default parameters — and columns are *not*
    peak-normalized, faithfully reproducing RC.jl.

    Parameters
    ----------
    amplitude : float, default=0.3
        Sine amplitude for the first-row seed (RC.jl ``amplitude``).
    sine_divisor : float, default=5.9
        Divisor in the sine phase (RC.jl ``sine_divisor``).
    logistic_parameter : float, default=3.7
        Growth rate ``r`` of the logistic recurrence (RC.jl
        ``logistic_parameter``).
    input_scaling : float, optional
        Uniform magnitude knob from the shared scaling contract. ``None`` (the
        default) leaves the raw logistic orbit; a float ``s`` multiplies every
        entry by ``s`` as the documented final transform.

    References
    ----------
    X. Wang et al. (logistic-map ESN input initialization), 2022.

    ReservoirComputing.jl ``logistic_mapping`` (``src/inits/inits_input.jl``).

    Examples
    --------
    >>> from resdag.init.input_feedback import LogisticMappingInitializer
    >>>
    >>> init = LogisticMappingInitializer()
    >>> weight = torch.empty(8, 3)
    >>> init.initialize(weight)
    """

    def __init__(
        self,
        amplitude: float = 0.3,
        sine_divisor: float = 5.9,
        logistic_parameter: float = 3.7,
        input_scaling: float | None = None,
    ) -> None:
        """Initialize the LogisticMappingInitializer."""
        super().__init__(input_scaling=input_scaling)
        self.amplitude = float(amplitude)
        self.sine_divisor = float(sine_divisor)
        self.logistic_parameter = float(logistic_parameter)

    def initialize(self, weight: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Initialize the weight via the logistic map.

        Parameters
        ----------
        weight : torch.Tensor
            Weight tensor of shape ``(out_features, in_features)``.
        **kwargs
            Per-call keyword overrides. A recognized ``input_scaling`` key
            overrides the bound ``self.input_scaling`` for this call only (the
            shared per-call contract); other keys are ignored.

        Returns
        -------
        torch.Tensor
            The initialized weight tensor (modified in-place).
        """
        out_features, in_features = _resolve_shape(weight)
        device = weight.device
        dtype = weight.dtype
        compute_dtype = _numpy_compute_dtype(dtype)

        values = np.zeros((out_features, in_features), dtype=compute_dtype)

        # First row: sine seed. RC.jl is 1-indexed, so column j uses (j + 1).
        cols = np.arange(1, in_features + 1, dtype=compute_dtype)
        values[0, :] = self.amplitude * np.sin(cols * np.pi / (self.sine_divisor * in_features))

        # Row-wise logistic recurrence.
        for i in range(1, out_features):
            prev = values[i - 1, :]
            values[i, :] = self.logistic_parameter * prev * (1.0 - prev)

        # Forward ``**kwargs`` so a per-call ``input_scaling`` override wins.
        values = self._apply_scaling(values, **kwargs)

        weight_data = torch.from_numpy(values).to(device=device, dtype=dtype)
        with torch.no_grad():
            weight.copy_(weight_data)
        return weight

    def __repr__(self) -> str:
        return (
            f"LogisticMappingInitializer(amplitude={self.amplitude}, "
            f"sine_divisor={self.sine_divisor}, logistic_parameter={self.logistic_parameter}, "
            f"input_scaling={self.input_scaling})"
        )
