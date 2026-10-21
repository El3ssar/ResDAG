"""Shared input and time-alignment contract for forecast implementations."""

import operator
from typing import Literal

import torch

from resdag._contracts import validate_sequence_inputs

ForecastAlignment = Literal["next", "legacy"]


def validate_alignment(alignment: str) -> None:
    if alignment not in ("next", "legacy"):
        raise ValueError(f"alignment must be 'next' or 'legacy', got {alignment!r}")


def positive_step_count(value: int, name: str) -> int:
    """Accept integer index types while rejecting booleans and truncation."""
    try:
        count = operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be a positive integer, got {value}") from None
    if isinstance(value, bool) or count < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return count


def prepare_forecast_inputs(
    warmup_inputs: tuple[torch.Tensor, ...] | torch.Tensor,
    forecast_inputs: tuple[torch.Tensor, ...] | torch.Tensor | None,
    *,
    num_inputs: int,
    horizon: int,
    initial_feedback: torch.Tensor | None,
    alignment: ForecastAlignment,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], int, bool]:
    """Validate BTF streams; return inputs, drivers, horizon, boundary flag.

    The boundary prediction is the final warmup output. It already estimates
    the first unseen sample and needs no future driver. Explicit seeds instead
    supply an additional input after warmup and consume one driver per output.
    """
    validate_alignment(alignment)
    count = positive_step_count(horizon, "horizon")
    inputs = (warmup_inputs,) if isinstance(warmup_inputs, torch.Tensor) else tuple(warmup_inputs)
    if not inputs:
        raise ValueError("At least one warmup input (feedback) is required")
    if len(inputs) != num_inputs:
        raise ValueError(f"Expected {num_inputs} warmup input(s), got {len(inputs)}")
    for i, stream in enumerate(inputs):
        if not isinstance(stream, torch.Tensor):
            raise TypeError(f"warmup_inputs[{i}] must be a torch.Tensor")
        if stream.dim() != 3:
            raise ValueError(f"warmup_inputs[{i}] must be 3-D (batch, time, features)")
        if stream.shape[:2] != inputs[0].shape[:2]:
            raise ValueError("All warmup inputs must match feedback batch/time dimensions")
        if stream.shape[0] == 0 or stream.shape[-1] == 0:
            raise ValueError("Warmup batch and feature dimensions must be nonempty")
    validate_sequence_inputs(
        inputs, feature_sizes=[stream.shape[-1] for stream in inputs], label="warmup_inputs"
    )
    if inputs[0].shape[1] == 0 and initial_feedback is None:
        raise ValueError("An empty warmup requires initial_feedback to seed the forecast")
    if initial_feedback is not None:
        expected = (inputs[0].shape[0], 1, inputs[0].shape[-1])
        if tuple(initial_feedback.shape) != expected:
            raise ValueError(
                f"initial_feedback must have shape {expected}; got {tuple(initial_feedback.shape)}"
            )
        validate_sequence_inputs(
            (initial_feedback,),
            feature_sizes=(expected[-1],),
            reference=inputs[0],
            label="initial_feedback",
        )

    boundary = alignment == "next" and initial_feedback is None
    steps = count - int(boundary)
    num_drivers = num_inputs - 1
    drivers: tuple[torch.Tensor, ...]
    if isinstance(forecast_inputs, torch.Tensor):
        drivers = (forecast_inputs,)
    else:
        drivers = () if forecast_inputs is None else tuple(forecast_inputs)
    if not num_drivers and drivers:
        raise ValueError("This feedback-only model does not accept forecast driving inputs")
    if num_drivers and not drivers:
        if steps:
            raise ValueError(
                f"Model has {num_drivers} driving input(s). forecast_inputs must be provided "
                f"with at least {steps} steps."
            )
        drivers = tuple(stream[:, :0, :] for stream in inputs[1:])
    if len(drivers) != num_drivers:
        raise ValueError(f"Expected {num_drivers} forecast driver(s), got {len(drivers)}")
    for i, driver in enumerate(drivers):
        if not isinstance(driver, torch.Tensor):
            raise TypeError(f"forecast_inputs[{i}] must be a torch.Tensor")
        if driver.dim() != 3:
            raise ValueError(f"forecast_inputs[{i}] must be 3-D (batch, time, features)")
        if driver.shape[0] != inputs[0].shape[0]:
            raise ValueError(f"forecast_inputs[{i}] batch size must match warmup feedback")
        if driver.shape[-1] != inputs[i + 1].shape[-1]:
            raise ValueError(f"forecast_inputs[{i}] feature size must match its warmup driver")
        if driver.shape[1] < steps:
            raise ValueError(
                f"forecast_inputs[{i}] has {driver.shape[1]} steps, expected at least {steps}. "
                "Drivers must start right after the warmup window and pair with the "
                "feedback sample consumed at the same time."
            )
        validate_sequence_inputs(
            (driver,),
            feature_sizes=(inputs[i + 1].shape[-1],),
            reference=inputs[i + 1],
            label=f"forecast_inputs[{i}]",
        )
    return inputs, drivers, count, boundary
