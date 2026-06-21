"""Bounded uniform initializer for input/feedback weights."""

from typing import Any

import torch

from resdag.utils.general import SeedLike

from .base import InputFeedbackInitializer, _resolve_shape
from .registry import register_input_feedback


@register_input_feedback(
    "uniform",
    low=-1.0,
    high=1.0,
    connectivity=None,
    input_scaling=None,
    seed=None,
)
class UniformInputInitializer(InputFeedbackInitializer):
    """Bounded uniform initializer for input/feedback weight matrices.

    Draws every entry independently from ``Uniform(low, high)``. Unlike the
    baseline ``random`` initializer (fixed ``[-1, 1]``), the bounds are
    configurable, so asymmetric or narrow ranges are expressible directly.
    Optionally sparsified per input channel via ``connectivity`` and scaled
    uniformly via the shared ``input_scaling`` contract.

    This is the migration target for reservoirpy's
    :func:`reservoirpy.mat_gen.uniform` (and the ``"uniform"`` distribution of
    :func:`reservoirpy.mat_gen.random_sparse`).

    Parameters
    ----------
    low : float, optional
        Inclusive lower bound of the uniform draw. Defaults to ``-1.0``. Must be
        strictly less than ``high``.
    high : float, optional
        Exclusive upper bound of the uniform draw. Defaults to ``1.0``.
    connectivity : float, optional
        Density knob from the shared scaling contract (see
        :class:`~resdag.init.input_feedback.InputFeedbackInitializer`): the
        fraction of nonzero entries kept per column (input channel), in
        ``(0, 1]``. ``None`` (the default) leaves the dense matrix untouched;
        ``connectivity=0.1`` keeps ~10% of each column's entries (at least one),
        zeroing the rest. The kept fraction is statistically exact:
        ``round(connectivity * reservoir_size)`` entries per column.
    input_scaling : float, optional
        Uniform magnitude knob from the shared scaling contract. ``None`` (the
        default) applies no scaling; a float ``s`` multiplies every entry by
        ``s``, so ``max|W|`` scales linearly with ``s`` (``input_scaling=0.5``
        halves it). Applied *after* the ``[low, high]`` draw, so it rescales the
        bounds rather than replacing them.
    seed : int, torch.Generator, or None, optional
        Reproducibility seed for the uniform draw *and* the connectivity mask.
        Accepts a plain ``int``, a :class:`torch.Generator` (whose
        ``initial_seed()`` is used, so a generator and the equivalent int
        agree), or ``None`` (defer to torch's global RNG). The value draw is
        **device-native**: it happens directly on the target weight's device via
        a torch generator, so the same ``seed`` is reproducible per device (CPU
        and CUDA each reproduce, though their RNG streams differ from each
        other).

    Notes
    -----
    The produced matrix is a pure function of ``(low, high, connectivity,
    input_scaling, seed, shape, device)``: repeated calls on the same instance
    with equal shapes on the same device yield identical matrices. Pass
    ``seed=None`` for a draw tied to torch's global RNG (reproducible under
    ``torch.manual_seed``).

    Examples
    --------
    >>> from resdag.init.input_feedback import UniformInputInitializer
    >>>
    >>> # A narrow, asymmetric uniform input matrix.
    >>> init = UniformInputInitializer(low=0.0, high=0.5, seed=42)
    >>> weight = torch.empty(100, 10)  # (reservoir_size, feedback_size)
    >>> init.initialize(weight)
    >>>
    >>> # Use in ESNLayer.
    >>> from resdag.layers import ESNLayer
    >>> reservoir = ESNLayer(
    ...     reservoir_size=100,
    ...     feedback_size=10,
    ...     feedback_initializer=init,
    ... )
    """

    def __init__(
        self,
        low: float = -1.0,
        high: float = 1.0,
        connectivity: float | None = None,
        input_scaling: float | None = None,
        seed: SeedLike = None,
    ) -> None:
        """Initialize the UniformInputInitializer."""
        super().__init__(
            input_scaling=input_scaling,
            connectivity=connectivity,
            seed=seed,
        )
        low = float(low)
        high = float(high)
        if not low < high:
            raise ValueError(
                f"low must be strictly less than high; got low={low!r}, high={high!r}."
            )
        self.low = low
        self.high = high

    def initialize(self, weight: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Initialize ``weight`` with ``Uniform(low, high)`` entries.

        Parameters
        ----------
        weight : torch.Tensor
            Weight tensor of shape ``(out_features, in_features)``. Modified
            in-place.

        Returns
        -------
        torch.Tensor
            The initialized weight tensor (same object, modified in-place).
        """
        out_features, in_features = _resolve_shape(weight)
        device = weight.device
        dtype = weight.dtype

        generator = self._torch_generator_for(device)

        # Draw uniform values in [low, high) directly on the target device.
        values = torch.empty(out_features, in_features, device=device, dtype=dtype)
        values.uniform_(self.low, self.high, generator=generator)

        # Apply the shared connectivity then scaling contract, in that order
        # (zeroing then scaling is identical either way; this matches the base
        # class's documented composition ``_apply_scaling(_apply_connectivity)``).
        values = self._apply_connectivity(values)
        values = self._apply_scaling(values)

        with torch.no_grad():
            weight.copy_(values)

        return weight

    def __repr__(self) -> str:
        return (
            f"UniformInputInitializer(low={self.low}, high={self.high}, "
            f"connectivity={self.connectivity}, input_scaling={self.input_scaling}, "
            f"seed={self.seed})"
        )
