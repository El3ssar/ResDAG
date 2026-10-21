---
description: Reuse native PyTorch GRUCell and LSTMCell weights inside reservoir graphs, with frozen or trainable parameters.
---

# Reuse a PyTorch GRU or LSTM

Wrap an existing `torch.nn.GRUCell` or `torch.nn.LSTMCell` to use it in a
ResDAG graph. Its weights and gate equations are preserved; the wrapper adds
the normal sequence, state, training and forecasting interfaces.

```python
from torch import nn
from resdag import TorchRecurrentLayer

native = nn.LSTMCell(input_size=3, hidden_size=32)
reservoir = TorchRecurrentLayer(native, feedback_size=2, input_size=1)
# reservoir(feedback, driver): (batch, time, 32)
```

The native cell's input size must equal `feedback_size + input_size`. Feedback
comes first in the concatenated input. `input_size=None` means feedback only.
The wrapper copies the supplied cell so freezing it does not unexpectedly
freeze the caller's original module. It inherits the original dtype/device.

Weights are frozen by default, making this useful for readout experiments on
existing recurrent features. Pass `trainable=True` to train the recurrent
weights with a PyTorch optimizer. Freezing a random GRU/LSTM does not guarantee
a useful reservoir; compare its validation performance with a plain ESN.

GRU state has width `hidden_size`. LSTM state packs `[hidden | cell]` with width
`2 * hidden_size`; the readout sees only the hidden part. If you use explicit
states, keep that packed width. The input projection is computed over the full
sequence before the recurrence loop, and the gate updates match the native
PyTorch cells.

```python
import torch

feedback = torch.randn(4, 20, 2)
driver = torch.randn(4, 20, 1)
initial = reservoir.cell.init_state(4, feedback.device, feedback.dtype)
features, state = reservoir.forward_stateless([feedback, driver], initial)
```

Use the standard [composition workflow](../../start/end-to-end.md) to attach
readouts and embed the complete trained graph in a larger model. See the
[API reference](../../reference/layers.md) for state methods and the official
[GRUCell](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRUCell.html)
and [LSTMCell](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html)
documentation for native gate definitions.
