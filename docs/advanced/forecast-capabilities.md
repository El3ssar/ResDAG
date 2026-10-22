# Custom layers in forecasting

Ordinary graph execution accepts standard `torch.nn.Module` layers, including
whole-sequence transformations. Forecasting must also decide how those layers
receive future inputs.

The default `execution="fast"` processes the observed history once, then consumes
one frame per future tick. Built-in feature transforms, pointwise PyTorch layers,
readouts, and visible `BaseReservoirLayer` nodes support this execution. Explicit
reservoir states carry their temporal memory between ticks.

## Declare a framewise custom module

Set `supports_framewise = True` when splitting along the time axis preserves the
module's behavior. For inputs shaped `(batch, time, features)`, the module must
preserve the batch and time axes and apply the same operation independently at
each time. Feature widths may change inside the graph. The declaration also
applies to a custom coupled aggregator's time axis, immediately before its final
feature axis.

```python
import torch
from torch import nn
import resdag as rd

class SignedSquare(nn.Module):
    supports_framewise = True

    def forward(self, values):
        return values.sign() * values.square()

source = rd.reservoir_input(1, dtype=torch.float64)
graph = rd.ESNModel(source, SignedSquare()(source))
observed = torch.tensor([[[0.5], [1.0], [2.0]]], dtype=torch.float64)
prediction = graph.forecast(observed, horizon=2)
torch.testing.assert_close(prediction, torch.tensor([[[4.0], [16.0]]]).double())
```

An unchanged inherited implementation retains its declaration. A subclass that
overrides `forward` must declare the capability again. An existing module instance
can instead set its own `supports_framewise = True` after its behavior has been
checked. A declaration is a semantic promise; ResDAG does not infer a custom
function's temporal behavior by trying sample inputs.

Known scalar symbolic arithmetic such as `source + 1`, `source ** 2`, and
feature-only slices remains fast. Generic lambdas, arbitrary tensor methods,
and unknown module classes require a declaration or explicit history execution.
ResDAG validates the graph before warmup, hooks, or recurrent-state changes.

## Replay the complete observed prefix

A cumulative sum depends on every earlier input. Giving it isolated future
frames would discard that information. Wrap the graph with `rd.Model` and select
`execution="history"` to rerun its complete input prefix at each future tick.

```python
import torch
from torch import nn
import resdag as rd

class PrefixSum(nn.Module):
    def forward(self, values):
        return values.cumsum(dim=1)

source = rd.reservoir_input(1, dtype=torch.float64)
features = PrefixSum()(source)
head = rd.RidgeReadoutLayer(1, 1, alpha=1e-6).double()
model = rd.Model(
    inputs={"observed": source},
    outputs={"prediction": head(features)},
    preprocessing="off",
    washout=0,
)
observed = torch.arange(1, 21, dtype=torch.float64)[:, None]
model.fit(observed)
prediction = model.forecast(history=observed, horizon=3, execution="history")
assert prediction.shape == (3, 1)
assert torch.isfinite(prediction).all()
```

Each replay starts visible reservoirs from their configured reset state and
recomputes the full supplied history plus every subsequently consumed forecast
frame. It keeps the final prediction unconsumed, matching normal forecast timing.
History execution costs more computation and memory as the prefix grows, and
requires `compile=False`.

Graphs that need history execution retain their complete observed fitting
history, so omitted `history` replays that prefix. Fast graphs retain a compact
observed tail. Supplying `history` explicitly starts from that observed prefix;
include every earlier sample the temporal operation needs. Within a forecast
call, replay never shortens the supplied prefix.

Both execution choices expect custom modules to behave as pure input functions
in evaluation mode. Temporal operations such as cumulative sums can compute
their memory from the supplied prefix. Arbitrary mutable Python state or
per-call temporal buffers cannot be reset automatically. Implement streaming
memory through a `ReservoirCell` wrapped in `BaseReservoirLayer`, exposing that
reservoir as its own functional graph node. Reservoirs hidden inside arbitrary
containers remain outside the explicit-state contract.

The low-level `ESNModel.forecast` provides fast execution. Use the common
`rd.Model(graph=...)` workflow for explicit history replay. Direct graph
`forward` and whole-sequence `forward_stateless` remain available for custom
operations without a framewise declaration.
