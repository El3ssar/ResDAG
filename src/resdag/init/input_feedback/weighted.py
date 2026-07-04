"""Weighted block-diagonal input initializers (Lu et al. 2017; RC.jl parity).

Two initializers that share RC.jl's "weighted" block-diagonal wiring, where each
input channel connects to a *disjoint* block of ``q = floor(reservoir_size /
in_size)`` consecutive reservoir units:

- :class:`WeightedInputInitializer` (RC.jl ``weighted_init``) — random uniform
  block values in ``[-input_scaling, input_scaling]``.
- :class:`WeightedMinimalInitializer` (RC.jl ``weighted_minimal``) — constant
  magnitude block values with signed entries (the minimal-complexity variant).

RC.jl's originals *recompute* the reservoir size to ``q * in_size`` and warn when
it differs from the requested one, leaving any trailing rows unconnected. resdag
initializers must fill the exact ``(reservoir_size, in_size)`` weight they are
given, so these keep the ``q``-per-channel block structure and leave the trailing
``reservoir_size - q * in_size`` rows as zeros (equivalent to RC.jl's unconnected
tail), rather than resizing the matrix.
"""

from typing import Any

import numpy as np
import torch

from resdag.utils.general import coerce_seed_to_int

from ._sign_sampling import build_sign_matrix
from .base import InputFeedbackInitializer, _numpy_compute_dtype, _resolve_shape
from .registry import register_input_feedback


def _block_size(out_features: int, in_features: int) -> int:
    """RC.jl ``q = floor(res_size / in_size)`` — units per input channel."""
    if in_features <= 0:
        raise ValueError(f"in_features must be >= 1; got {in_features}.")
    return out_features // in_features


@register_input_feedback("weighted_init", input_scaling=0.1, seed=None)
class WeightedInputInitializer(InputFeedbackInitializer):
    """Weighted block-diagonal random input layer (RC.jl ``weighted_init`` [Lu2017]).

    Each of the ``in_size`` input channels connects to a *disjoint* contiguous
    block of ``q = floor(reservoir_size / in_size)`` reservoir units. Block
    values are drawn uniformly from ``[-input_scaling, input_scaling]``; every
    off-block entry is zero. Any trailing ``reservoir_size - q * in_size`` rows
    are left unconnected (all-zero), matching RC.jl's unconnected tail (RC.jl
    resizes the matrix instead; resdag keeps the requested shape — see the module
    docstring).

    This differs from :class:`PseudoDiagonalInitializer` only in the block
    sizing: ``weighted_init`` gives every channel *exactly* ``q`` units (leaving
    a zero tail when ``reservoir_size`` is not a multiple of ``in_size``), while
    ``pseudo_diagonal`` distributes the remainder so every unit is connected.

    Parameters
    ----------
    input_scaling : float, default=0.1
        Half-width of the uniform block draw: nonzero entries are sampled from
        ``[-input_scaling, input_scaling]`` (RC.jl ``scaling``). This is the
        shared scaling-contract knob, so ``max|W|`` scales linearly with it.
    seed : int, torch.Generator, or None, optional
        Reproducibility seed for the uniform block draws. ``None`` draws fresh
        each call.

    References
    ----------
    Z. Lu et al., "Reservoir observers: Model-free inference of unmeasured
    variables in chaotic systems," Chaos, vol. 27, no. 4, 041102, 2017.

    ReservoirComputing.jl ``weighted_init`` (``src/inits/inits_input.jl``).

    Examples
    --------
    >>> from resdag.init.input_feedback import WeightedInputInitializer
    >>>
    >>> init = WeightedInputInitializer(input_scaling=0.1, seed=42)
    >>> weight = torch.empty(9, 3)  # 3 blocks of q=3 units each
    >>> init.initialize(weight)
    """

    def __init__(
        self,
        input_scaling: float | None = 0.1,
        seed: int | None = None,
    ) -> None:
        """Initialize the WeightedInputInitializer."""
        super().__init__(input_scaling=input_scaling, seed=seed)

    def initialize(self, weight: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Fill a block-diagonal weight with uniform block values.

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

        rng = np.random.default_rng(coerce_seed_to_int(self.seed))
        q = _block_size(out_features, in_features)

        # Natural range is [-1, 1]; the shared scaling contract multiplies by
        # ``input_scaling`` to land the RC.jl ``[-scaling, scaling]`` block range.
        values = np.zeros((out_features, in_features), dtype=compute_dtype)
        for idx in range(in_features):
            block = rng.uniform(-1.0, 1.0, size=(q,)).astype(compute_dtype)
            values[idx * q : (idx + 1) * q, idx] = block

        # Forward ``**kwargs`` so a per-call ``input_scaling`` override wins.
        values = self._apply_scaling(values, **kwargs)

        weight_data = torch.from_numpy(values).to(device=device, dtype=dtype)
        with torch.no_grad():
            weight.copy_(weight_data)
        return weight

    def __repr__(self) -> str:
        return f"WeightedInputInitializer(input_scaling={self.input_scaling}, seed={self.seed})"


@register_input_feedback(
    "weighted_minimal",
    weight=0.1,
    sign_rule="none",
    input_scaling=None,
    seed=None,
)
class WeightedMinimalInitializer(InputFeedbackInitializer):
    """Minimal weighted block-diagonal input layer (RC.jl ``weighted_minimal``).

    The deterministic-magnitude sibling of :class:`WeightedInputInitializer`:
    each input channel connects to a disjoint block of ``q = floor(reservoir_size
    / in_size)`` reservoir units, but every nonzero entry has the **same
    magnitude** ``weight`` (the minimal-complexity construction of Rodan & Tiňo,
    applied to the weighted block wiring inspired by Lu et al. 2017). Only the
    signs vary, chosen by ``sign_rule``.

    The trailing ``reservoir_size - q * in_size`` rows are left unconnected
    (all-zero), matching RC.jl's unconnected tail (RC.jl resizes instead; resdag
    keeps the requested shape — see the module docstring).

    Sign rules (RC.jl ``sampling_type``) are applied over the **full**
    ``(reservoir_size, in_size)`` matrix in RC.jl's column-major traversal order,
    exactly as RC.jl runs the sampler after building the block matrix; off-block
    zeros are unaffected by a sign flip. See
    :class:`~resdag.init.input_feedback.MinimalInputInitializer` for the rule
    definitions.

    Parameters
    ----------
    weight : float, default=0.1
        The single magnitude for every nonzero block entry (RC.jl ``weight``).
    sign_rule : {"none", "bernoulli", "irrational", "regular"}, default="none"
        Sign-assignment rule (RC.jl ``sampling_type`` without ``_sample!``).
        RC.jl's ``weighted_minimal`` defaults to ``:no_sample`` (all positive).
    positive_prob : float, default=0.5
        Probability an entry stays positive under ``sign_rule="bernoulli"``.
    start : int, default=1
        First pi-digit offset under ``sign_rule="irrational"``.
    strides : int, default=2
        Stride under ``sign_rule="regular"``.
    input_scaling : float, optional
        Uniform magnitude knob from the shared scaling contract. ``None`` leaves
        nonzero entries at magnitude ``weight``; a float ``s`` multiplies every
        entry by ``s`` (applied on top of ``weight``).
    seed : int, torch.Generator, or None, optional
        Reproducibility seed for the ``"bernoulli"`` sign draw; ignored by the
        deterministic rules.

    References
    ----------
    A. Rodan and P. Tiňo, "Minimum Complexity Echo State Network," IEEE Trans.
    Neural Netw., vol. 22, no. 1, pp. 131-144, Jan. 2011.

    Z. Lu et al., "Reservoir observers," Chaos, vol. 27, no. 4, 041102, 2017.

    ReservoirComputing.jl ``weighted_minimal`` (``src/inits/inits_input.jl``).

    Examples
    --------
    >>> from resdag.init.input_feedback import WeightedMinimalInitializer
    >>>
    >>> init = WeightedMinimalInitializer(weight=0.99)
    >>> weight = torch.empty(9, 3)  # 3 blocks of q=3 units, all +0.99
    >>> init.initialize(weight)
    """

    def __init__(
        self,
        weight: float = 0.1,
        sign_rule: str = "none",
        positive_prob: float = 0.5,
        start: int = 1,
        strides: int = 2,
        input_scaling: float | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the WeightedMinimalInitializer."""
        super().__init__(input_scaling=input_scaling, seed=seed)
        self.weight = float(weight)
        self.sign_rule = sign_rule
        self.positive_prob = float(positive_prob)
        self.start = int(start)
        self.strides = int(strides)

    def initialize(self, weight: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Fill a block-diagonal weight with constant-magnitude signed entries.

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

        rng = np.random.default_rng(coerce_seed_to_int(self.seed))
        q = _block_size(out_features, in_features)

        # Build the constant-magnitude block matrix (off-block entries stay 0).
        values = np.zeros((out_features, in_features), dtype=compute_dtype)
        for idx in range(in_features):
            values[idx * q : (idx + 1) * q, idx] = self.weight

        # RC.jl runs the sampler over the whole matrix after building the blocks.
        signs = build_sign_matrix(
            out_features,
            in_features,
            self.sign_rule,
            positive_prob=self.positive_prob,
            start=self.start,
            strides=self.strides,
            rng=rng,
        )
        values = values * signs.astype(compute_dtype)

        # Forward ``**kwargs`` so a per-call ``input_scaling`` override wins.
        values = self._apply_scaling(values, **kwargs)

        weight_data = torch.from_numpy(values).to(device=device, dtype=dtype)
        with torch.no_grad():
            weight.copy_(weight_data)
        return weight

    def __repr__(self) -> str:
        return (
            f"WeightedMinimalInitializer(weight={self.weight}, sign_rule={self.sign_rule!r}, "
            f"input_scaling={self.input_scaling}, seed={self.seed})"
        )
