"""Legendre delay-network memory sequence layer."""

import torch

from resdag.init.utils import InitializerSpec
from resdag.layers.cells.legendre_delay_cell import LegendreDelayCell
from resdag.utils.general import SeedLike

from ._cell_layer import CellReservoirLayer


class LegendreDelayLayer(CellReservoirLayer):
    """Stateful fixed LDN memory; see `LegendreDelayCell` for configuration.

    Returns `order * memory_channels` coefficients per timestep, flattened
    channel first. Default channels preserve all feedback and driver dimensions.
    `theta` and `dt` use the same time units and fix the discretization at
    construction. This is linear delay memory, not a full nonlinear LMU.
    """

    _fixed_attributes = frozenset({"order", "memory_channels", "theta", "dt"})

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
        super().__init__(
            LegendreDelayCell(
                order,
                feedback_size,
                input_size,
                theta=theta,
                dt=dt,
                memory_channels=memory_channels,
                feedback_initializer=feedback_initializer,
                input_initializer=input_initializer,
                trainable=trainable,
                seed=seed,
                device=device,
                dtype=dtype,
            )
        )
