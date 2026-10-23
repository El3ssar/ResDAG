"""Explicit historical weight recipes for fixed algorithm-quality fixtures only."""

from typing import Any

import torch


def _sequential_uniform(
    tensor: torch.Tensor,
    *,
    seed: int,
    reservoir_size: int,
    feedback_size: int,
    named_feedback: bool,
    bias: bool,
) -> torch.Tensor:
    """Replay the admitted historical draws on the destination dtype/device."""
    generator = torch.Generator(device=tensor.device).manual_seed(seed)
    # The original default generator advanced for feedback only when feedback
    # used the default. Named feedback initializers owned a separate stream.
    if not named_feedback:
        tensor.new_empty(reservoir_size, feedback_size).uniform_(-1, 1, generator=generator)
    if bias:
        tensor.new_empty(reservoir_size, reservoir_size).uniform_(-1, 1, generator=generator)
    return tensor.uniform_(-1, 1, generator=generator)


def sequential_uniform_initializers(
    reservoir_size: int,
    feedback_size: int,
    seed: int,
    *,
    named_feedback: bool = False,
) -> dict[str, Any]:
    """Pin recurrent/bias weights while keeping current production defaults.

    These explicit public callback specs reproduce the historical sequential
    uniform fixture: default feedback, then recurrence, then bias. Set
    ``named_feedback=True`` when feedback already used a named initializer and
    therefore did not consume the old default stream. The constructor still
    performs its ordinary spectral normalization. This helper is deliberately
    test-only; it is not a compatibility mode or a new-default quality claim.
    """
    options = dict(
        seed=seed,
        reservoir_size=reservoir_size,
        feedback_size=feedback_size,
        named_feedback=named_feedback,
    )
    return {
        "recurrent_initializer": (_sequential_uniform, {**options, "bias": False}),
        "bias_initializer": (_sequential_uniform, {**options, "bias": True}),
    }
