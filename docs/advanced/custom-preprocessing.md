---
description: Fit reversible per-port Torch preprocessing while preserving physical prediction units.
---

# Custom preprocessing

Use `input_transforms` and `output_transforms` to override preprocessing by
port name. Each transform preserves the signals and their feature count;
the workflow fits it on training rows, runs the graph in transformed units,
and returns predictions in the original units. Put temporal operations or
feature construction inside the functional graph instead.

## Reuse Standardize or disable one port

`rd.Standardize(num_features=D)` works directly. Use `nn.Identity()` to
disable preprocessing for one port while retaining the global fallback
for other ports.

```python
import io
import torch
from torch import nn
import resdag as rd

training = torch.arange(1, 21, dtype=torch.float64)[:, None]
x = rd.reservoir_input(1, dtype=torch.float64)
head = rd.RidgeReadoutLayer(1, 1, alpha=0).double()
incoming = rd.Standardize(num_features=1)
model = rd.Model(
    inputs={"state": x},
    outputs={"prediction": head(x)},
    washout=2,
    preprocessing="standard",
    input_transforms={"state": incoming},
    output_transforms={"prediction": nn.Identity()},
).fit(training)
assert model.input_scalers[0] is incoming
torch.testing.assert_close(
    model.forecast(horizon=3), torch.arange(21, 24).double()[:, None]
)
```

Keys match the names declared by `Model(inputs=..., outputs=...)`. The
`ESN` preset uses `"input"` and `"output"`. Explicit overrides also run when
`preprocessing="off"`; unspecified ports use the global preprocessing policy.

## Define a reversible Torch module

A custom transform is an ordinary `nn.Module` with:

- `fit(training)`, accepting `(batch, time, features)` and returning `self`;
- `forward(data)` and `inverse(data)`, preserving the exact shape, dtype and device;
- `supports_framewise = True`, declaring that splitting the time axis into
  individual frames gives the same result;
- all learned state in persistent, pre-registered buffers or parameters with
  fixed names and shapes.

Allocate state for the known feature width in the constructor. Fit by copying
values into it, rather than adding or resizing registered tensors:

```python
class CenterScale(nn.Module):
    supports_framewise = True

    def __init__(self, num_features):
        super().__init__()
        self.register_buffer("center", torch.zeros(num_features))
        self.register_buffer("scale", torch.ones(num_features))

    @torch.no_grad()
    def fit(self, training):
        self.center.copy_(training.mean(dim=(0, 1)))
        self.scale.copy_(training.std(dim=(0, 1), unbiased=False).clamp_min(0.01))
        return self

    def forward(self, data):
        return (data - self.center) / self.scale

    def inverse(self, data):
        return data * self.scale + self.center


class Asinh(nn.Module):
    supports_framewise = True

    def fit(self, training):
        return self

    def forward(self, data):
        return data.asinh()

    def inverse(self, data):
        return data.sinh()
```

A fixed unit converter can use the same no-op `fit` recipe as `Asinh`.
`nn.Identity()` is the standard exception that needs neither `fit` nor
`inverse` methods. The workflow owns fittedness; a custom `is_fitted`
property is unnecessary.

Fit and forecast apply transforms in evaluation mode, restoring every
module's previous mode afterward. Their evaluation behavior must be pure,
deterministic, framewise and reversible, without modifying inputs or learned
state. Keep temporal state in a visible reservoir node using
`BaseReservoirLayer`; see [custom components](custom-components.md) and
[forecast execution](forecast-capabilities.md).

## Train in transformed units and forecast in physical units

The output override fits the actual target rows. This example learns a
nonlinear physical map by making its transformed targets linear:

```python
def make_model():
    signal = rd.reservoir_input(1, dtype=torch.float64)
    readout = rd.RidgeReadoutLayer(1, 1, alpha=0).double()
    return rd.Model(
        inputs={"state": signal},
        outputs={"prediction": readout(signal)},
        washout=0,
        preprocessing="off",
        input_transforms={"state": CenterScale(1)},
        output_transforms={"prediction": Asinh()},
    )

values = torch.linspace(-2, 2, 40, dtype=torch.float64)[:, None]
custom = make_model().fit(values, targets=(0.2 * values + 0.05).sinh())
current = values[-1:]
expected = []
for _ in range(3):
    current = (0.2 * current + 0.05).sinh()
    expected.append(current)
torch.testing.assert_close(
    custom.forecast(horizon=3), torch.cat(expected), rtol=1e-12, atol=1e-12
)

source = values[:4].clone().requires_grad_()
custom(source).sum().backward()
torch.testing.assert_close(
    source.grad, 0.2 * (0.2 * source.detach() + 0.05).cosh()
)
```

Only supervised rows after washout fit each transform. Custom output
transforms use the aligned targets, including next-step targets when targets
are omitted. An unspecified output keeps the built-in fallback: for implicit
next-step training it fits the original input's supervised rows. It does not
inherit a custom input transform. Feedback always converts through physical
units: output `inverse`, then receiving input `forward`.

## State, refitting and checkpoints

Fitting stages independent copies, validates reversible behavior on a bounded
sample and checks that loading registered tensor state alone reproduces it.
It also checks finite transformed training data. A failed staged fit leaves
the fitted workflow unchanged. Successful fits preserve the supplied module
and parameter identities, so an optimizer created before fitting keeps its
references. Distinct ports need independent modules and tensor storage;
sharing state with another port or the raw graph is rejected. Expanded or
overlapping state tensors must be cloned into writable independent storage.

Standard `.to(...)` moves the registered state. Ordinary transforms can be
fitted inside `torch.inference_mode()` and used with autograd afterward;
construct custom parameters outside inference mode. At inference, shape,
dtype and device checks remain active, while NaN and infinity propagate
according to the transform's Torch operations.

Save and load a matching workflow's state with standard Torch tools:

```python
buffer = io.BytesIO()
torch.save(custom.state_dict(), buffer)
buffer.seek(0)
state = torch.load(buffer, weights_only=True, map_location="cpu")
restored = make_model()
restored.load_state_dict(state)
torch.testing.assert_close(
    restored.forecast(horizon=3), custom.forecast(horizon=3), rtol=0, atol=0
)
```

Reconstruct the same transform classes, feature widths and Python
configuration before loading. The checkpoint validates registered tensor
schema and sampled behavior; arbitrary Python settings remain your
responsibility. Store fitted quantities in registered tensors, including
flags that affect fitted behavior. Ordinary Torch `load_state_dict(assign=True)`
can replace parameters, so construct an optimizer after such a load.
