---
description: Small public recipes for named seeds, registered initializers, custom recurrent cells, transforms and solvers.
---

# Build your own components

Start with existing layers and replace only the part you need. A custom
initializer builds weights; a cell defines one recurrent update; a Torch
module transforms signals; a readout solver fits the final prediction map.
They all work inside the same functional `rd.Model`.

## Give branches independent, repeatable seeds

```python
import torch
from torch import nn
import resdag as rd

seeds = rd.SeedStreams(42)
fast_seed = seeds["fast"]
slow_seed = seeds["slow"]
assert fast_seed != slow_seed
assert rd.SeedStreams(42)["fast"] == fast_seed
```

Explicit names keep the same seeds when you reorder branches or add another
one. Reusing a name gives the same seed; it does not share parameters or
memory between layers. The fixed `blake2b-v1` algorithm uses Python's standard
`hashlib`, consumes no global random state and accepts names such as
`"encoder/fast"`. Keep those names stable in experiments.

This controls initialization, not numerical nondeterminism in accelerator
kernels or the current position of a noise generator. Pass the derived seeds
to your components; constructing `Model` never reinitializes supplied layers.

## Register a weight initializer

```python
@rd.init.register_recurrent_initializer("tutorial_cycle", prescaled=True)
def tutorial_cycle(size, gain=0.8, *, device=None, dtype=None):
    return gain * torch.eye(size, device=device, dtype=dtype).roll(1, dims=0)

reservoir = rd.ESNLayer(
    8, feedback_size=2, seed=fast_seed, dtype=torch.float64,
    recurrent_initializer=("tutorial_cycle", {"gain": 0.7}),
)
```

`prescaled=True` means the builder owns its recurrent scale. Otherwise the
layer applies its requested spectral radius. Random builders should accept
`seed` and use a local Torch or NumPy generator. Registration is per process:
import your registration module on each worker. Direct callables work too.
See [custom initialization](../build/initialization/custom.md) for input,
feedback, bias and reset-state builders.

## Define one recurrent update

The cell below keeps an exponentially weighted memory. It uses the existing
state and sequence machinery rather than writing another time loop:

```python
class MovingAverageCell(rd.ReservoirCell):
    def __init__(self, width, rate=0.3):
        super().__init__()
        self.width = width
        self.register_buffer("rate", torch.tensor(rate, dtype=torch.float64))

    @property
    def state_size(self):
        return self.width

    @property
    def output_size(self):
        return self.width

    def init_state(self, batch_size, device, dtype):
        return torch.zeros(batch_size, self.width, device=device, dtype=dtype)

    def forward(self, inputs, state):
        updated = (1 - self.rate) * state + self.rate * inputs[0]
        return updated, updated

memory = rd.BaseReservoirLayer(MovingAverageCell(2))
```

The layer handles carried state, explicit state, batch iteration and device
moves. Register learnable quantities as `nn.Parameter` and fixed tensors as
buffers. Keep state in the returned state tensor. For a faster cell, the
optional public `project_inputs` and `step` methods can separate the batched
input projection from recurrence.

## Add a framewise transform

```python
class SignedSquare(nn.Module):
    supports_framewise = True

    def forward(self, data):
        return data.sign() * data.square()
```

The declaration means that splitting time into individual frames gives the
same result. It is valid here because each value is transformed independently.
A temporal operation such as a running sum needs a recurrent cell or
[history execution](forecast-capabilities.md).

## Supply a readout solver

Implement the public `solve(states, targets)` method. The base class owns
shape checks, fitted flags and copying coefficients into its PyTorch layer:

```python
class LeastSquaresReadout(rd.ReadoutLayer):
    def solve(self, states, targets):
        coefficients = torch.linalg.lstsq(states, targets).solution
        return coefficients, None
```

Inputs are flattened `(samples, features)` matrices. Return coefficients of
shape `(input_features, output_features)` and an optional output intercept.
This example uses `bias=False`. A solver supporting bias should estimate and
return that intercept. Use `fit` as the training entrypoint; `solve` is its
extension hook and may update solver diagnostics or online fitting statistics.
For example, the incremental and RLS solvers update their accumulated or
recursive state. Use the supplied dtype/device and standard Torch linear
algebra, keeping transfers explicit.

## Connect them and use the common workflow

```python
x = rd.reservoir_input(2, dtype=torch.float64)
fast = reservoir(x)
slow = memory(x)
features = rd.Concatenate()(SignedSquare()(fast), slow)
y = LeastSquaresReadout(10, 2, bias=False).double()(features)
custom = rd.Model(inputs={"state": x}, outputs={"prediction": y}, washout=5)

t = torch.linspace(0, 8, 90, dtype=torch.float64)
training = torch.stack((t.sin(), t.cos()), dim=-1)
custom.fit(training)
prediction = custom.forecast(horizon=5)
assert prediction.shape == (5, 2)
assert torch.isfinite(prediction).all()
```

Expose each reservoir as a separate functional node so explicit-state and
forecast execution can thread its state. Add branches with new layer
instances; share weight tensors explicitly when desired. See the
[common model guide](../start/model-workflow.md) for PyTorch pipelines and
[the roadmap](../project/roadmap.md) for tuning and complete model artifacts.
