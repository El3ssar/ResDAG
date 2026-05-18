# Add a reservoir cell

1. Implement `ReservoirCell` (single timestep).
2. Wrap with `BaseReservoirLayer` (sequence + state API).

```python
import torch
import torch.nn as nn
from resdag.layers.cells import ReservoirCell
from resdag.layers.reservoirs.base_reservoir import BaseReservoirLayer


class LinearCell(ReservoirCell):
  @property
  def state_size(self) -> int:
      return self.hidden

  @property
  def output_size(self) -> int:
      return self.hidden

  def __init__(self, hidden: int, input_dim: int):
      super().__init__()
      self.hidden = hidden
      self.W = nn.Linear(input_dim, hidden, bias=False)

  def init_state(self, batch_size, device=None, dtype=None):
      return torch.zeros(batch_size, self.hidden, device=device, dtype=dtype)

  def forward(self, inputs, state):
      x = inputs[0]  # feedback slice at this step
      if len(inputs) > 1:
          x = torch.cat([x, inputs[1]], dim=-1)
      new_state = self.W(x)
      return new_state, new_state


class LinearReservoir(BaseReservoirLayer):
    def __init__(self, hidden: int, feedback_size: int, input_size: int = 0):
        in_dim = feedback_size + input_size
        super().__init__(LinearCell(hidden, in_dim))
```

Override `set_state` on the layer if your state tensor rank differs from `(batch, state_size)` (see `NGReservoir`).

## See also

- [Reservoir layers](../reference/layers/reservoirs.md)
- [`ReservoirCell`](../reference/layers/cells.md)
