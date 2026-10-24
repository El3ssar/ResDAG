"""Fixed Legendre delay-network memory with exact zero-order-hold discretization."""

import math

import torch
from torch import Tensor

from resdag.init.utils import InitializerSpec
from resdag.utils.general import SeedLike

from ._projected import ProjectedInputCell, finite_scalar, positive_size


class LegendreDelayCell(ProjectedInputCell):
    r"""Linear Legendre delay network (LDN), rather than a full nonlinear LMU.

    Implements equations (1), (2), and (4) of Voelker et al., NeurIPS 2019.
    The analytic continuous matrices are fixed buffers, with `theta` already
    divided into `continuous_A` and `continuous_B`. An augmented matrix
    exponential computes exact zero-order-hold `A`/`B` for interval `dt`;
    no matrix inverse or Euler approximation is used.

    `order` is the number of Legendre coefficients **per channel**.
    `memory_channels=None` uses `feedback_size + (input_size or 0)` channels,
    one for each input dimension; their default encoders concatenate the raw
    feedback and driver. When channels equal that sum, identity encoders are
    also used by default. Other channel counts use random linear encoders with
    uniform bounds `1/sqrt(total input width)`. Explicit feedback/input
    initializer specs override the respective encoder.

    State and output both have width `memory_channels * order`. They flatten
    `(batch, memory_channels, order)` in channel-major order. `theta` is a
    memory window in the same time units as `dt` (default dt=1 means samples).
    `trainable=True` trains only input encoders; the memory dynamics stay fixed.
    The linear memory should be followed by nonlinear features/model components
    when the task needs nonlinear predictions.

    The scaled continuous matrix 1-norm must not exceed `2**20`; extreme
    `dt/theta` ratios or orders beyond this numerical bound are rejected
    before exponentiation. MPS construction initializes on CPU then moves once.

    Reference: https://papers.nips.cc/paper/9689-legendre-memory-units-continuous-time-representation-in-recurrent-neural-networks
    """

    A: Tensor
    B: Tensor

    def __init__(
        self,
        order: int,
        feedback_size: int,
        input_size: int | None = None,
        *,
        theta: float = 100.0,
        dt: float = 1.0,
        memory_channels: int | None = None,
        feedback_initializer: InitializerSpec = None,
        input_initializer: InitializerSpec = None,
        trainable: bool = False,
        seed: SeedLike = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        device = torch.device(device) if device is not None else torch.get_default_device()
        order = positive_size("order", order)
        positive_size("feedback_size", feedback_size)
        if input_size is not None:
            positive_size("input_size", input_size)
        total_width = feedback_size + (input_size or 0)
        channels = (
            total_width
            if memory_channels is None
            else positive_size("memory_channels", memory_channels)
        )
        theta = finite_scalar("theta", theta, positive=True)
        dt = finite_scalar("dt", dt, positive=True)
        super().__init__(
            channels,
            feedback_size,
            input_size,
            bias=False,
            bias_scaling=0.0,
            feedback_initializer=feedback_initializer,
            input_initializer=input_initializer,
            seed=seed,
            device=device,
            dtype=dtype,
        )
        self._order = order
        self._memory_channels = channels
        self.reservoir_size = order * channels
        self._theta = theta
        self._dt = dt
        with torch.no_grad():
            if channels == total_width:
                encoder = torch.eye(
                    channels, device=self.weight_feedback.device, dtype=self.weight_feedback.dtype
                )
                if feedback_initializer is None:
                    self.weight_feedback.copy_(encoder[:, :feedback_size])
                if input_initializer is None and self.weight_input is not None:
                    self.weight_input.copy_(encoder[:, feedback_size:])
            else:
                scale = 1.0 / math.sqrt(total_width)
                if feedback_initializer is None:
                    self.weight_feedback.mul_(scale)
                if input_initializer is None and self.weight_input is not None:
                    self.weight_input.mul_(scale)
        # Half/bfloat matrix_exp is not numerically supported. Compute once in
        # float32 then cast; requested float64 remains float64 throughout.
        parameter_dtype = self.weight_feedback.dtype
        work_dtype = torch.float64 if parameter_dtype == torch.float64 else torch.float32
        i = torch.arange(order, device=self.weight_feedback.device)
        rows, columns = i[:, None], i[None, :]
        sign = torch.where((rows - columns + 1).remainder(2) == 0, 1.0, -1.0)
        A = ((2 * rows + 1).to(work_dtype) * torch.where(rows < columns, -1.0, sign)) / theta
        B = ((2 * i + 1).to(work_dtype) * torch.where(i.remainder(2) == 0, 1.0, -1.0)) / theta
        if not torch.isfinite(A).all() or not torch.isfinite(B).all():
            raise ValueError(
                "theta/order produce non-finite continuous memory dynamics in the requested dtype"
            )
        augmented = torch.zeros(order + 1, order + 1, device=A.device, dtype=work_dtype)
        augmented[:order, :order] = A
        augmented[:order, order] = B
        scaled = augmented * dt
        # A finite elementwise matrix can still overflow norm estimation or
        # require an unreasonable number of scaling/squaring iterations. Bound
        # the numerical work before entering the backend (which may otherwise
        # hang on Inf or extreme finite inputs). Ordinary memory-window ratios
        # are many orders below this limit.
        norm = torch.linalg.matrix_norm(scaled, ord=1)
        if not torch.isfinite(norm) or norm > 2**20:
            raise ValueError(
                "dt/theta/order exceed the supported ZOH matrix 1-norm (2**20); "
                "reduce dt/theta or order"
            )
        if order == 1:
            # The order-1 system is a scalar leaky integrator. Its closed form
            # also avoids roundoff in matrix_exp's small 2x2 approximation.
            exponent = A[0, 0] * dt
            discrete_A = exponent.exp().reshape(1, 1)
            discrete_B = -exponent.expm1().reshape(1)
        else:
            transition = torch.matrix_exp(scaled)
            discrete_A = transition[:order, :order]
            discrete_B = transition[:order, order]
        self.continuous_A: Tensor
        self.continuous_B: Tensor
        self.register_buffer("continuous_A", A.to(parameter_dtype))
        self.register_buffer("continuous_B", B.to(parameter_dtype))
        self.register_buffer("A", discrete_A.to(parameter_dtype).contiguous())
        self.register_buffer("B", discrete_B.to(parameter_dtype).contiguous())
        if any(
            not torch.isfinite(value).all()
            for value in (self.continuous_A, self.continuous_B, self.A, self.B)
        ):
            raise ValueError(
                "theta/dt/order produce non-finite memory dynamics in the requested dtype"
            )
        self.trainable = trainable
        if device is not None and torch.device(device).type == "mps":
            self.to(device)

    @property
    def order(self) -> int:
        """Number of Legendre coefficients retained per memory channel."""
        return self._order

    @property
    def memory_channels(self) -> int:
        """Number of independently encoded input channels."""
        return self._memory_channels

    @property
    def theta(self) -> float:
        """Memory-window length used when constructing the fixed dynamics."""
        return self._theta

    @property
    def dt(self) -> float:
        """Sampling interval used to discretize the memory dynamics."""
        return self._dt

    @property
    def state_size(self) -> int:
        """Packed memory width, `memory_channels * order`."""
        return self.reservoir_size

    @property
    def output_size(self) -> int:
        """Readable memory width, equal to `state_size`."""
        return self.reservoir_size

    def step(self, projected_t: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        """Advance one zero-order-hold interval and return readable and carried memory."""
        self._check_step(projected_t, state, self.memory_channels)
        memory = state.reshape(state.shape[0], self.memory_channels, self.order)
        new_memory = memory @ self.A.T + projected_t.unsqueeze(-1) * self.B
        output = new_memory.reshape(state.shape[0], self.state_size)
        return output, output

    def extra_repr(self) -> str:
        return (
            f"order={self.order}, memory_channels={self.memory_channels}, "
            f"feedback_size={self.feedback_size}, input_size={self.input_size}, "
            f"theta={self.theta}, dt={self.dt}, trainable={self.trainable}"
        )
