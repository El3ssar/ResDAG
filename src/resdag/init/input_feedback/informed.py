"""Informed input initializer for hybrid/knowledge ESNs (RC.jl `informed_init`)."""

from typing import Any

import numpy as np
import torch

from resdag.utils.general import coerce_seed_to_int, create_rng

from .base import InputFeedbackInitializer, _numpy_compute_dtype, _resolve_shape
from .registry import register_input_feedback


@register_input_feedback(
    "informed_init",
    model_in_size=None,
    gamma=0.5,
    input_scaling=0.1,
    seed=None,
)
class InformedInputInitializer(InputFeedbackInitializer):
    """Informed input layer for hybrid/knowledge ESNs (RC.jl `informed_init` [Pathak2018]).

    Splits the reservoir units into a "state" group and a "model" group so a
    hybrid ESN can wire the raw system state and the knowledge-based model output
    into disjoint fractions of the reservoir. The input columns are partitioned
    the same way: the first `state_size = in_size - model_in_size` columns are
    the raw state channels; the trailing `model_in_size` columns are the model
    channels.

    Construction (mirrors RC.jl `informed_init`):

    1. `state_size = in_size - model_in_size` (must be `> 0`).
    2. `num_for_state = floor(reservoir_size * gamma)` reservoir rows receive a
       single random connection to a randomly chosen **state** column
       `[0, state_size)`.
    3. `num_for_model = floor(reservoir_size * (1 - gamma))` reservoir rows
       receive a single random connection to a randomly chosen **model** column
       `[state_size, in_size)`.

    Each chosen entry is a random value in `[-input_scaling, input_scaling]`
    (RC.jl: `(rand - 0.5) * 2 * scaling`). Candidate rows are drawn only from
    rows that are still all-zero, so a unit gets **at most one** input
    connection; when fewer zero rows remain than requested the loop stops early
    (RC.jl's `isempty(...) && break`). All other entries are zero.

    `gamma` is the fraction of the reservoir devoted to the raw state, so
    `gamma` near `1` weights the reservoir toward the state and near `0`
    toward the knowledge model. A later `HybridESN` phase consumes this
    initializer to build its input matrix.

    Parameters
    ----------
    model_in_size : int
        Number of *model* input channels (RC.jl `model_in_size`). The weight's
        `in_features` must exceed this so `state_size > 0`. Required — the
        bare-name registry default is `None` and raises at `initialize` time
        if left unset.
    gamma : float, default=0.5
        Fraction of reservoir units wired to the raw state block (RC.jl
        `gamma`). Must lie in `[0, 1]`.
    input_scaling : float, default=0.1
        Half-width of the uniform draw for each connection: values land in
        `[-input_scaling, input_scaling]` (RC.jl `scaling`). Shared
        scaling-contract knob, so `max|W|` scales linearly with it.
    seed : int, torch.Generator, or None, optional
        Reproducibility seed for the row/column choices and value draws. `None`
        draws fresh each call.

    References
    ----------
    J. Pathak et al., "Hybrid forecasting of chaotic processes: Using machine
    learning in conjunction with a knowledge-based model," Chaos, vol. 28, no. 4,
    041101, 2018.

    ReservoirComputing.jl `informed_init` (`src/inits/inits_input.jl`).

    Examples
    --------
    >>> from resdag.init.input_feedback import InformedInputInitializer
    >>>
    >>> # 3 input channels: 1 model column, 2 state columns; half the reservoir
    >>> # wired to the state block.
    >>> init = InformedInputInitializer(model_in_size=1, gamma=0.5, seed=42)
    >>> weight = torch.empty(10, 3)
    >>> init.initialize(weight)
    """

    def __init__(
        self,
        model_in_size: int | None = None,
        gamma: float = 0.5,
        input_scaling: float | None = 0.1,
        seed: int | None = None,
    ) -> None:
        """Initialize the InformedInputInitializer.

        Raises
        ------
        ValueError
            If `gamma` is not in `[0, 1]`, or `model_in_size` is negative.
        """
        super().__init__(input_scaling=input_scaling, seed=seed)
        if not (0.0 <= float(gamma) <= 1.0):
            raise ValueError(f"gamma must be in [0, 1]; got {gamma!r}.")
        if model_in_size is not None and model_in_size < 0:
            raise ValueError(f"model_in_size must be >= 0; got {model_in_size!r}.")
        self.model_in_size = model_in_size
        self.gamma = float(gamma)

    def initialize(self, weight: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Fill the informed (state/model split) input weight.

        Parameters
        ----------
        weight : torch.Tensor
            Weight tensor of shape `(out_features, in_features)`.
        **kwargs
            Per-call keyword overrides. A recognized `input_scaling` key
            overrides the bound `self.input_scaling` for this call only (the
            shared per-call contract); other keys are ignored.

        Returns
        -------
        torch.Tensor
            The initialized weight tensor (modified in-place).

        Raises
        ------
        ValueError
            If `model_in_size` is unset, or `in_features <= model_in_size`
            (so `state_size <= 0`).
        """
        out_features, in_features = _resolve_shape(weight)
        device = weight.device
        dtype = weight.dtype
        compute_dtype = _numpy_compute_dtype(dtype)

        if self.model_in_size is None:
            raise ValueError(
                "InformedInputInitializer requires 'model_in_size'; pass it explicitly, "
                "e.g. get_input_feedback('informed_init', model_in_size=1)."
            )
        model_in_size = self.model_in_size
        state_size = in_features - model_in_size
        if state_size <= 0:
            raise ValueError(
                f"in_features ({in_features}) must be greater than model_in_size "
                f"({model_in_size}) so state_size > 0."
            )

        rng = create_rng(coerce_seed_to_int(self.seed))
        values = np.zeros((out_features, in_features), dtype=compute_dtype)

        num_for_state = int(np.floor(out_features * self.gamma))
        num_for_model = int(np.floor(out_features * (1.0 - self.gamma)))

        # Draw the per-call scaling for the ``[-scaling, scaling]`` value range so
        # a per-call ``input_scaling`` override is honored (RC.jl bakes the scale
        # into each entry rather than as a final uniform multiply, so it is
        # resolved here rather than via ``_apply_scaling``).
        scaling = self._resolve_input_scaling(kwargs)

        def zero_row_indices() -> np.ndarray:
            """Rows still entirely zero — candidates for a new connection."""
            return np.flatnonzero(~values.any(axis=1))

        def wire(count: int, col_low: int, col_high: int) -> None:
            """Wire `count` still-zero rows to a random column in `[col_low, col_high)`."""
            for _ in range(count):
                candidates = zero_row_indices()
                if candidates.size == 0:
                    break
                row = int(rng.choice(candidates))
                col = int(rng.integers(col_low, col_high))
                # RC.jl: (rand - 0.5) * 2 * scaling -> uniform in [-scaling, scaling].
                draw = (rng.random() - 0.5) * 2.0
                values[row, col] = draw if scaling is None else draw * scaling

        wire(num_for_state, 0, state_size)
        wire(num_for_model, state_size, in_features)

        weight_data = torch.from_numpy(values).to(device=device, dtype=dtype)
        with torch.no_grad():
            weight.copy_(weight_data)
        return weight

    def __repr__(self) -> str:
        return (
            f"InformedInputInitializer(model_in_size={self.model_in_size}, gamma={self.gamma}, "
            f"input_scaling={self.input_scaling}, seed={self.seed})"
        )
