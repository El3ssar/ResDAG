"""Coupled oscillator sequence layer."""

import torch

from resdag.init.utils import InitializerSpec, RecurrentInitializerSpec, TopologySpec
from resdag.layers.cells.cornn_cell import CoRNNCell
from resdag.utils.general import SeedLike

from ._cell_layer import CellReservoirLayer


class CoRNNLayer(CellReservoirLayer):
    """Stateful coRNN layer; see `CoRNNCell` for dynamics and packed states.

    `forward` returns position sequences, `forward_stateless` additionally
    returns packed position/velocity state. The shared reservoir layer supports
    chunked state, differentiable scans, and projected inference/forecasting.
    """

    _live_attributes = frozenset({"dt", "gamma", "epsilon"})

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
        super().__init__(
            CoRNNCell(
                reservoir_size,
                feedback_size,
                input_size,
                dt=dt,
                gamma=gamma,
                epsilon=epsilon,
                recurrent_initializer=recurrent_initializer,
                velocity_initializer=velocity_initializer,
                topology=topology,
                spectral_radius=spectral_radius,
                feedback_initializer=feedback_initializer,
                input_initializer=input_initializer,
                bias=bias,
                bias_scaling=bias_scaling,
                trainable=trainable,
                seed=seed,
                device=device,
                dtype=dtype,
            )
        )
