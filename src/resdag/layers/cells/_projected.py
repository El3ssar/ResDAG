"""Private input projection and validation helpers for dense reservoir cells."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from resdag.init.utils import InitializerSpec, resolve_initializer
from resdag.utils.general import SeedLike, coerce_seed_to_int, create_torch_generator

from .base_cell import ReservoirCell


def positive_size(name: str, value: int) -> int:
    """Reject ambiguous/nonintegral dimensions before allocating tensors."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def finite_scalar(name: str, value: float, *, positive: bool = False) -> float:
    """Validate a finite nonnegative (or strictly positive) scalar."""
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        bound = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {bound}")
    return result


def role_seed(seed: SeedLike, offset: int) -> int | None:
    """Give independent parameter groups reproducible RNG streams."""
    value = coerce_seed_to_int(seed)
    return None if value is None else (value + offset) % (2**63 - 1)


def initialize_input(weight: Tensor, spec: InitializerSpec, seed: SeedLike) -> None:
    """Initialize on the target tensor's device and dtype, without retaining a callable."""
    initializer = resolve_initializer(spec, seed=seed)
    with torch.no_grad():
        if initializer is None:
            nn.init.uniform_(
                weight, -1.0, 1.0, generator=create_torch_generator(seed, weight.device)
            )
        else:
            weight.copy_(initializer.initialize(weight))
        if not torch.isfinite(weight).all():
            raise ValueError("Input initializer produced non-finite weights")


def validate_inputs(inputs: list[Tensor], feedback_size: int, input_size: int | None) -> None:
    """Validate either a timestep or a sequence; never move/cast caller tensors."""
    expected = 1 if input_size is None else 2
    if len(inputs) != expected:
        raise ValueError(f"Expected {expected} input streams (feedback first); got {len(inputs)}")
    feedback = inputs[0]
    if feedback.ndim not in (2, 3) or feedback.shape[-1] != feedback_size:
        raise ValueError(
            f"Feedback must have shape (batch, {feedback_size}) or "
            f"(batch, time, {feedback_size}); got {tuple(feedback.shape)}"
        )
    if input_size is not None:
        driver = inputs[1]
        if driver.shape[:-1] != feedback.shape[:-1] or driver.shape[-1:] != (input_size,):
            raise ValueError("Driver must match feedback batch/time axes and configured input_size")
        if driver.device != feedback.device or driver.dtype != feedback.dtype:
            raise ValueError("Feedback and driver must have the same dtype and device")


class ProjectedInputCell(ReservoirCell):
    """Common affine feedback/driver projection, with live trainability."""

    def __init__(
        self,
        projection_size: int,
        feedback_size: int,
        input_size: int | None,
        *,
        bias: bool,
        bias_scaling: float,
        feedback_initializer: InitializerSpec,
        input_initializer: InitializerSpec,
        seed: SeedLike,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        self.feedback_size = positive_size("feedback_size", feedback_size)
        self.input_size = None if input_size is None else positive_size("input_size", input_size)
        if not isinstance(bias, bool):
            raise TypeError("bias must be a bool")
        if dtype is not None and not dtype.is_floating_point:
            raise TypeError("dtype must be a floating point dtype")
        bias_scaling = finite_scalar("bias_scaling", bias_scaling)
        # Initialize on CPU for portable MPS RNG/linalg support (in particular,
        # matrix_exp is unavailable); concrete cells move once complete.
        if device is not None and torch.device(device).type == "mps":
            if (dtype or torch.get_default_dtype()) == torch.float64:
                raise TypeError("MPS does not support float64; use float32 or CPU")
            device = torch.device("cpu")
        self.weight_feedback = nn.Parameter(
            torch.empty(projection_size, feedback_size, device=device, dtype=dtype),
            requires_grad=False,
        )
        self.weight_input: nn.Parameter | None
        self.bias_h: nn.Parameter | None
        if input_size is None:
            self.register_parameter("weight_input", None)
        else:
            self.weight_input = nn.Parameter(
                torch.empty(projection_size, input_size, device=device, dtype=dtype),
                requires_grad=False,
            )
        if bias:
            self.bias_h = nn.Parameter(
                torch.empty(projection_size, device=device, dtype=dtype), requires_grad=False
            )
        else:
            self.register_parameter("bias_h", None)
        initialize_input(self.weight_feedback, feedback_initializer, role_seed(seed, 11))
        if self.weight_input is not None:
            initialize_input(self.weight_input, input_initializer, role_seed(seed, 23))
        if self.bias_h is not None:
            nn.init.uniform_(
                self.bias_h,
                -bias_scaling,
                bias_scaling,
                generator=create_torch_generator(role_seed(seed, 37), self.bias_h.device),
            )
        self._trainable = False

    @property
    def trainable(self) -> bool:
        """Whether the encoder/recurrent parameters receive gradients."""
        return self._trainable

    @trainable.setter
    def trainable(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("trainable must be a bool")
        self._trainable = value
        for parameter in self.parameters():
            parameter.requires_grad_(value)

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.zeros(batch_size, self.state_size, device=device, dtype=dtype)

    def project_inputs(self, inputs: list[Tensor]) -> Tensor:
        validate_inputs(inputs, self.feedback_size, self.input_size)
        projected = F.linear(inputs[0], self.weight_feedback, self.bias_h)
        if self.weight_input is not None:
            projected = projected + F.linear(inputs[1], self.weight_input)
        return projected

    def forward(self, inputs: list[Tensor], state: Tensor) -> tuple[Tensor, Tensor]:
        return self.step(self.project_inputs(inputs), state)

    def _check_step(self, projected: Tensor, state: Tensor, projection_size: int) -> None:
        self.validate_state(state)
        if projected.ndim != 2 or projected.shape != (state.shape[0], projection_size):
            raise ValueError(f"Projected timestep must have shape (batch, {projection_size})")
