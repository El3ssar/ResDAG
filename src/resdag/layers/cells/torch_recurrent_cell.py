"""Adapt standard PyTorch GRU/LSTM cells to the reservoir state protocol."""

from copy import deepcopy

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ._projected import positive_size, validate_inputs
from .base_cell import ReservoirCell


class TorchRecurrentCell(ReservoirCell):
    """Use an existing `nn.GRUCell` or `nn.LSTMCell` in a ResDAG graph.

    Deep-copies the supplied standard cell, preserving its parameters, device,
    and dtype. `trainable=False` freezes the copy; the caller's cell is never
    changed. Its `input_size` must equal `feedback_size + (input_size or 0)`.
    Inputs are ordered feedback then optional driver. LSTM state is packed
    `cat([h, c], -1)` of width `2*hidden_size`; GRU state is `h`. Both
    expose only `h` as output. This permits imported/pretrained cells and
    ordinary gradient training without requiring a separate RNN implementation.

    Input gate projections are batched across the entire sequence; recurrent
    updates implement the documented PyTorch equations, including GRU's reset
    gate placement around the hidden new-gate projection and its bias. A frozen
    GRU/LSTM is not guaranteed to have the echo state property.

    References: https://docs.pytorch.org/docs/stable/generated/torch.nn.GRUCell.html
    and https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html
    """

    def __init__(
        self,
        cell: nn.GRUCell | nn.LSTMCell,
        feedback_size: int,
        input_size: int | None = None,
        *,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        if type(cell) not in (nn.GRUCell, nn.LSTMCell):
            raise TypeError("cell must be a standard torch.nn.GRUCell or torch.nn.LSTMCell")
        self.feedback_size = positive_size("feedback_size", feedback_size)
        self.input_size = None if input_size is None else positive_size("input_size", input_size)
        total_width = feedback_size + (input_size or 0)
        if cell.input_size != total_width:
            raise ValueError(
                f"Native cell input_size must equal feedback plus driver width ({total_width})"
            )
        self.recurrent_cell = deepcopy(cell)
        self.reservoir_size = cell.hidden_size
        self._is_lstm = isinstance(cell, nn.LSTMCell)
        self.trainable = trainable

    @property
    def state_size(self) -> int:
        """Carried width: hidden state for GRU, packed hidden/carry state for LSTM."""
        return self.reservoir_size * (2 if self._is_lstm else 1)

    @property
    def output_size(self) -> int:
        """Hidden-state width, equal to the supplied native cell's `hidden_size`."""
        return self.reservoir_size

    @property
    def trainable(self) -> bool:
        """Whether the copied native cell's parameters receive gradients."""
        return self._trainable

    @trainable.setter
    def trainable(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("trainable must be a bool")
        self._trainable = value
        for parameter in self.parameters():
            parameter.requires_grad_(value)

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Return zero initial hidden state, including LSTM carry when applicable."""
        return torch.zeros(batch_size, self.state_size, device=device, dtype=dtype)

    def project_inputs(self, inputs: list[Tensor]) -> Tensor:
        """Project feedback and optional driver into the native cell's input gates."""
        validate_inputs(inputs, self.feedback_size, self.input_size)
        combined = inputs[0] if self.input_size is None else torch.cat(inputs, dim=-1)
        cell = self.recurrent_cell
        return F.linear(combined, cell.weight_ih, cell.bias_ih)

    def step(self, projected_t: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        """Advance one projected frame and return hidden output and carried state."""
        self.validate_state(state)
        gates = 4 if self._is_lstm else 3
        if projected_t.ndim != 2 or projected_t.shape != (
            state.shape[0],
            gates * self.reservoir_size,
        ):
            raise ValueError(
                f"Projected timestep must have shape (batch, {gates * self.reservoir_size})"
            )
        cell = self.recurrent_cell
        h = state[:, : self.reservoir_size]
        recurrent = F.linear(h, cell.weight_hh, cell.bias_hh)
        if self._is_lstm:
            i, f, g, o = (projected_t + recurrent).chunk(4, dim=-1)
            c = state[:, self.reservoir_size :]
            new_c = f.sigmoid() * c + i.sigmoid() * g.tanh()
            new_h = o.sigmoid() * new_c.tanh()
            return new_h, torch.cat((new_h, new_c), dim=-1)
        ir, iz, inn = projected_t.chunk(3, dim=-1)
        hr, hz, hn = recurrent.chunk(3, dim=-1)
        reset = (ir + hr).sigmoid()
        update = (iz + hz).sigmoid()
        candidate = (inn + reset * hn).tanh()
        # Match native arithmetic ordering, including promotion under autocast.
        new_h = candidate + update * (h - candidate)
        return new_h, new_h

    def forward(self, inputs: list[Tensor], state: Tensor) -> tuple[Tensor, Tensor]:
        """Advance one raw feedback/driver frame using the native GRU/LSTM equations."""
        return self.step(self.project_inputs(inputs), state)

    def extra_repr(self) -> str:
        return (
            f"kind={type(self.recurrent_cell).__name__}, reservoir_size={self.reservoir_size}, "
            f"feedback_size={self.feedback_size}, input_size={self.input_size}, trainable={self.trainable}"
        )
