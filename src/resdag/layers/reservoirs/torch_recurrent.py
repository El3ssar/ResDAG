"""Use native PyTorch recurrent cells in stateful ResDAG sequence layers."""

from torch import nn

from resdag.layers.cells.torch_recurrent_cell import TorchRecurrentCell

from ._cell_layer import CellReservoirLayer


class TorchRecurrentLayer(CellReservoirLayer):
    """Wrap a copy of a standard GRUCell/LSTMCell, frozen by default.

    Inherits device/dtype and parameter values from `cell`. See
    `TorchRecurrentCell` for feedback/driver ordering and packed LSTM
    state. Sequence output contains only hidden states, not LSTM carry states.
    Set `trainable=True` to include this layer in a larger gradient pipeline.
    """

    def __init__(
        self,
        cell: nn.GRUCell | nn.LSTMCell,
        feedback_size: int,
        input_size: int | None = None,
        *,
        trainable: bool = False,
    ) -> None:
        super().__init__(TorchRecurrentCell(cell, feedback_size, input_size, trainable=trainable))
