"""Coupled oscillators from Rusch and Mishra, ICLR 2021 (arXiv:2010.00951)."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from resdag.init.topology import scale_to_spectral_radius
from resdag.init.utils import (
    InitializerSpec,
    RecurrentInitializerSpec,
    TopologySpec,
    resolve_recurrent_initializer,
    select_recurrent_spec,
)
from resdag.utils.general import SeedLike, create_torch_generator

from ._projected import ProjectedInputCell, finite_scalar, positive_size, role_seed


class CoRNNCell(ProjectedInputCell):
    r"""Coupled nonlinear oscillators, frozen by default.

    Uses the explicit-damping variant of equation (3) in the coRNN paper::

        v_new = v + dt * (tanh(W_hh h + W_hv v + W_fb x + W_in u + b)
                          - gamma*h - epsilon*v)
        h_new = h + dt * v_new

    The packed state is ``cat([h, v], -1)`` with width ``2*reservoir_size``;
    the output is position ``h_new`` with width ``reservoir_size``.
    ``recurrent_initializer`` configures the position matrix ``weight_hh``;
    ``velocity_initializer`` configures the velocity matrix ``weight_hv``.
    ``spectral_radius`` scales only the position matrix, respecting a builder's
    rescaling policy. It does not guarantee stability of the oscillator update.
    ``dt``, ``gamma`` and ``epsilon`` are finite positive/nonnegative scalars.
    Both recurrent matrices default to uniform(-1/sqrt(size), 1/sqrt(size)).
    Input projections default to uniform(-1, 1). Weights are initialized directly
    in ``dtype`` on ``device`` (MPS initializes on CPU then transfers once). ``trainable=True`` enables ordinary backpropagation.

    Reference: https://arxiv.org/abs/2010.00951
    """

    def __init__(
        self,
        reservoir_size: int,
        feedback_size: int,
        input_size: int | None = None,
        *,
        dt: float = 0.1,
        gamma: float = 1.0,
        epsilon: float = 1.0,
        recurrent_initializer: RecurrentInitializerSpec = None,
        velocity_initializer: RecurrentInitializerSpec = None,
        topology: TopologySpec = None,
        spectral_radius: float | None = None,
        feedback_initializer: InitializerSpec = None,
        input_initializer: InitializerSpec = None,
        bias: bool = True,
        bias_scaling: float = 1.0,
        trainable: bool = False,
        seed: SeedLike = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        device = torch.device(device) if device is not None else torch.get_default_device()
        size = positive_size("reservoir_size", reservoir_size)
        dt = finite_scalar("dt", dt, positive=True)
        gamma = finite_scalar("gamma", gamma)
        epsilon = finite_scalar("epsilon", epsilon)
        if spectral_radius is not None:
            spectral_radius = finite_scalar("spectral_radius", spectral_radius)
        position_spec = select_recurrent_spec(
            topology=topology, recurrent_initializer=recurrent_initializer
        )
        super().__init__(
            size,
            feedback_size,
            input_size,
            bias=bias,
            bias_scaling=bias_scaling,
            feedback_initializer=feedback_initializer,
            input_initializer=input_initializer,
            seed=seed,
            device=device,
            dtype=dtype,
        )
        self.reservoir_size = size
        self.dt = dt
        self.gamma = gamma
        self.epsilon = epsilon
        self.spectral_radius = spectral_radius
        self.weight_hh = nn.Parameter(
            torch.empty(size, size, device=self.weight_feedback.device, dtype=dtype),
            requires_grad=False,
        )
        self.weight_hv = nn.Parameter(
            torch.empty(size, size, device=self.weight_feedback.device, dtype=dtype),
            requires_grad=False,
        )
        for weight, spec, offset, radius in (
            (self.weight_hh, position_spec, 41, spectral_radius),
            (self.weight_hv, velocity_initializer, 53, None),
        ):
            initializer = resolve_recurrent_initializer(spec, seed=role_seed(seed, offset))
            with torch.no_grad():
                if initializer is None:
                    bound = 1.0 / math.sqrt(size)
                    nn.init.uniform_(
                        weight,
                        -bound,
                        bound,
                        generator=create_torch_generator(role_seed(seed, offset), weight.device),
                    )
                    if radius is not None:
                        weight.copy_(scale_to_spectral_radius(weight, radius))
                else:
                    weight.copy_(initializer.initialize(weight, spectral_radius=radius))
                if not torch.isfinite(weight).all():
                    raise ValueError("Recurrent initializer produced non-finite weights")
        self.trainable = trainable
        if device is not None and torch.device(device).type == "mps":
            self.to(device)

    @property
    def dt(self) -> float:
        """Positive integration interval; assignments affect the next update."""
        return self._dt

    @dt.setter
    def dt(self, value: float) -> None:
        self._dt = finite_scalar("dt", value, positive=True)

    @property
    def gamma(self) -> float:
        """Nonnegative position restoring coefficient."""
        return self._gamma

    @gamma.setter
    def gamma(self, value: float) -> None:
        self._gamma = finite_scalar("gamma", value)

    @property
    def epsilon(self) -> float:
        """Nonnegative velocity damping coefficient."""
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float) -> None:
        self._epsilon = finite_scalar("epsilon", value)

    @property
    def state_size(self) -> int:
        """Packed position-plus-velocity width, twice ``reservoir_size``."""
        return 2 * self.reservoir_size

    @property
    def output_size(self) -> int:
        """Position output width, equal to ``reservoir_size``."""
        return self.reservoir_size

    def step(self, projected_t: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        """Return updated positions and packed position/velocity state for one frame."""
        self._check_step(projected_t, state, self.reservoir_size)
        h, v = state.split(self.reservoir_size, dim=-1)
        force = torch.tanh(projected_t + F.linear(h, self.weight_hh) + F.linear(v, self.weight_hv))
        new_v = v + self.dt * (force - self.gamma * h - self.epsilon * v)
        new_h = h + self.dt * new_v
        return new_h, torch.cat((new_h, new_v), dim=-1)

    def extra_repr(self) -> str:
        return (
            f"reservoir_size={self.reservoir_size}, feedback_size={self.feedback_size}, "
            f"input_size={self.input_size}, dt={self.dt}, gamma={self.gamma}, "
            f"epsilon={self.epsilon}, trainable={self.trainable}"
        )
