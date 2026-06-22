"""Chain-of-neurons specific input initializer."""

from typing import Any, Sequence, cast

import numpy as np
import torch

from .base import InputFeedbackInitializer, _numpy_compute_dtype, _resolve_shape
from .registry import register_input_feedback


@register_input_feedback("chain_of_neurons_input", features=None, weights=1.0)
class ChainOfNeuronsInputInitializer(InputFeedbackInitializer):
    """Input initializer for chain-of-neurons reservoirs.

    For reservoirs organized as multiple parallel chains, this initializer
    connects each input to the first neuron of its corresponding chain.

    Parameters
    ----------
    features : int, optional
        Number of chains. Must equal the number of inputs (the weight's
        ``in_features``). ``None`` (the default) infers it from the weight's
        column count at :meth:`initialize` time, so the bare-name registry
        path ``get_input_feedback("chain_of_neurons_input")`` works. When an
        explicit value is given it is validated against the weight shape.
    weights : float or Sequence[float], default=1.0
        Either a single float (same weight for all input→chain pairs) or
        a sequence of floats (one weight per input/chain).

    Examples
    --------
    >>> from resdag.init.input_feedback import ChainOfNeuronsInputInitializer
    >>>
    >>> # Infer the number of chains from the weight (3 columns -> 3 chains)
    >>> init = ChainOfNeuronsInputInitializer(weights=1.0)
    >>> weight = torch.empty(150, 3)  # (reservoir_size, num_inputs)
    >>> init.initialize(weight)
    >>>
    >>> # Or pin it explicitly (validated against the weight shape)
    >>> init = ChainOfNeuronsInputInitializer(features=3, weights=1.0)
    >>>
    >>> # Each input connects only to the first neuron of its chain
    """

    def __init__(
        self,
        features: int | None = None,
        weights: float | Sequence[float] = 1.0,
    ):
        """Initialize the ChainOfNeuronsInputInitializer.

        Raises
        ------
        ValueError
            If ``features`` is given but ``< 1``.
        """
        if features is not None and features < 1:
            raise ValueError(f"'features' must be >= 1, got {features}.")
        self.features = features
        self.weights = weights

    def initialize(self, weight: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Initialize weight tensor for chain-of-neurons topology.

        Parameters
        ----------
        weight : torch.Tensor
            Weight tensor of shape (out_features, in_features)
            out_features = reservoir_size (units)
            in_features = num_inputs (must equal features; when ``features`` is
            ``None`` it is inferred from this column count)

        Returns
        -------
        torch.Tensor
            Initialized weight tensor

        Raises
        ------
        ValueError
            If an explicit ``features`` does not equal ``in_features``, or
            ``units`` is not a multiple of the number of chains.
        """
        units, input_dim = _resolve_shape(weight)
        device = weight.device
        dtype = weight.dtype
        compute_dtype = _numpy_compute_dtype(dtype)

        # ``features`` equals the number of chains, which equals the weight's
        # column count (one input per chain). Infer it from the weight when the
        # caller left it unset; otherwise validate the explicit value matches.
        if self.features is None:
            features = input_dim
        else:
            features = self.features
            if input_dim != features:
                raise ValueError(
                    f"input_dim ({input_dim}) must equal 'features' ({features}) "
                    "to have one input per chain."
                )

        if units % features != 0:
            raise ValueError(
                f"Number of units ({units}) must be a multiple of 'features' "
                f"({features}) to align chains with inputs."
            )

        # Build at the target precision so non-float32-representable per-input
        # weights survive a float64 weight intact.
        values = np.zeros((units, input_dim), dtype=compute_dtype)

        # Resolve per-input weights
        if isinstance(self.weights, (list, tuple, np.ndarray)):
            if len(self.weights) != input_dim:
                raise ValueError(
                    "When 'weights' is a sequence, its length must equal input_dim; "
                    f"got len(weights)={len(self.weights)}, input_dim={input_dim}."
                )
            in_weights = [float(w) for w in self.weights]
        else:
            # Reached only when ``self.weights`` is not a list/tuple/ndarray, i.e.
            # the scalar branch of the ``float | Sequence[float]`` union; cast for mypy.
            w = float(cast(float, self.weights))
            in_weights = [w] * input_dim

        block_len = units // features

        # Deterministic: input i → first unit of chain i
        for i in range(input_dim):
            start = i * block_len
            values[start, i] = in_weights[i]

        # Convert to tensor and copy to weight
        weight_data = torch.from_numpy(values).to(device=device, dtype=dtype)

        with torch.no_grad():
            weight.copy_(weight_data)

        return weight

    def __repr__(self) -> str:
        return f"ChainOfNeuronsInputInitializer(features={self.features}, weights={self.weights})"
