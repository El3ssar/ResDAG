"""Minimal-complexity input initializer (Rodan & Tiňo 2011; RC.jl `minimal_init`)."""

from typing import Any

import torch

from resdag.utils.general import coerce_seed_to_int, create_rng

from ._sign_sampling import build_sign_matrix
from .base import InputFeedbackInitializer, _numpy_compute_dtype, _resolve_shape
from .registry import register_input_feedback


@register_input_feedback(
    "minimal_init",
    weight=0.1,
    sign_rule="bernoulli",
    input_scaling=None,
    seed=None,
)
class MinimalInputInitializer(InputFeedbackInitializer):
    """Minimal-complexity input weights (Rodan & Tiňo; RC.jl `minimal_init`).

    Every entry of the input matrix has the **same magnitude** `weight`; only
    the sign varies, chosen by a deterministic or stochastic `sign_rule`. This
    is the "minimal complexity" input layer of Rodan & Tiňo — a single weight
    magnitude across all input-to-reservoir connections, paired with a simple
    cycle reservoir — which they show rivals fully random ESNs on many tasks.

    The matrix is **dense**: every reservoir unit receives every input (unlike
    `WeightedMinimalInitializer`, whose connections are block-diagonal).

    Construction (mirrors RC.jl `minimal_init`):

    1. Fill an `(reservoir_size, in_size)` matrix with the constant `weight`.
    2. Apply the `sign_rule` in RC.jl's column-major traversal order.

    Sign rules (RC.jl `sampling_type`):

    - `"bernoulli"` (default) — each entry keeps its `+` sign with
      probability `positive_prob` and is negated otherwise (RC.jl
      `bernoulli_sample!`). This is the RC.jl default. Requires a `seed` for
      reproducibility.
    - `"irrational"` — deterministic: an entry is negated when the
      corresponding decimal digit of pi (offset by `start`) is odd (RC.jl
      `irrational_sample!`). Fully reproducible, no RNG.
    - `"regular"` — deterministic: every `strides`-th entry (in flat
      column-major order) is negated (RC.jl `_regular_sample!`).
    - `"none"` — leave every sign positive (RC.jl `no_sample`).

    Parameters
    ----------
    weight : float, default=0.1
        The single magnitude assigned to every input weight (RC.jl `weight`).
    sign_rule : {"bernoulli", "irrational", "regular", "none"}, default="bernoulli"
        The sign-assignment rule (RC.jl `sampling_type` without the
        `_sample!` suffix).
    positive_prob : float, default=0.5
        Probability an entry stays positive under `sign_rule="bernoulli"`
        (RC.jl `positive_prob`).
    start : int, default=1
        First pi-digit offset under `sign_rule="irrational"` (RC.jl `start`).
    strides : int, default=2
        Stride under `sign_rule="regular"`; every `strides`-th flat entry is
        negated (RC.jl `strides`).
    input_scaling : float, optional
        Uniform magnitude knob from the shared scaling contract (see
        `resdag.init.input_feedback.InputFeedbackInitializer`). `None`
        (the default) leaves entries at magnitude `weight`; a float `s`
        multiplies every entry by `s`, so `max|W|` scales linearly with `s`
        (`input_scaling=0.5` halves it). Applied *on top of* `weight`.
    seed : int, torch.Generator, or None, optional
        Reproducibility seed for the `"bernoulli"` sign draw. Ignored by the
        deterministic rules. `None` draws fresh each call.

    References
    ----------
    A. Rodan and P. Tiňo, "Minimum Complexity Echo State Network," IEEE Trans.
    Neural Netw., vol. 22, no. 1, pp. 131-144, Jan. 2011.

    ReservoirComputing.jl `minimal_init` (`src/inits/inits_input.jl`).

    Examples
    --------
    >>> from resdag.init.input_feedback import MinimalInputInitializer
    >>>
    >>> # Deterministic pi-digit signs, magnitude 0.1
    >>> init = MinimalInputInitializer(weight=0.1, sign_rule="irrational")
    >>> weight = torch.empty(8, 3)
    >>> init.initialize(weight)  # dense {-0.1, +0.1} matrix
    """

    def __init__(
        self,
        weight: float = 0.1,
        sign_rule: str = "bernoulli",
        positive_prob: float = 0.5,
        start: int = 1,
        strides: int = 2,
        input_scaling: float | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the MinimalInputInitializer."""
        super().__init__(input_scaling=input_scaling, seed=seed)
        self.weight = float(weight)
        self.sign_rule = sign_rule
        self.positive_prob = float(positive_prob)
        self.start = int(start)
        self.strides = int(strides)

    def initialize(self, weight: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Initialize a dense constant-magnitude weight with signed entries.

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
        """
        out_features, in_features = _resolve_shape(weight)
        device = weight.device
        dtype = weight.dtype
        compute_dtype = _numpy_compute_dtype(dtype)

        rng = create_rng(coerce_seed_to_int(self.seed))
        signs = build_sign_matrix(
            out_features,
            in_features,
            self.sign_rule,
            positive_prob=self.positive_prob,
            start=self.start,
            strides=self.strides,
            rng=rng,
        )

        # Dense constant magnitude, then per-entry sign.
        values = (self.weight * signs).astype(compute_dtype)

        # Apply the shared uniform scaling contract as the documented final
        # transform. Forward ``**kwargs`` so a per-call ``input_scaling`` override
        # wins over the bound value.
        values = self._apply_scaling(values, **kwargs)

        weight_data = torch.from_numpy(values).to(device=device, dtype=dtype)
        with torch.no_grad():
            weight.copy_(weight_data)
        return weight

    def __repr__(self) -> str:
        return (
            f"MinimalInputInitializer(weight={self.weight}, sign_rule={self.sign_rule!r}, "
            f"input_scaling={self.input_scaling}, seed={self.seed})"
        )
