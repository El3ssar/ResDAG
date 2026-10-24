---
description: Save complete models, reload their DAGs, preserve streaming memory and export diagrams.
---

<span class="nb-kicker">Work · Save and load</span>

# Save, load and inspect a model

## The everyday workflow

```python
import torch
import resdag as rd

series = torch.sin(torch.linspace(0, 18, 220, dtype=torch.float64))[:, None]
model = rd.ESN(reservoir_size=24, washout=15, seed=7).double().fit(series)
expected = model.forecast(horizon=20)
model.save("run.pt")

restored = rd.load("run.pt", map_location="cpu")
actual = restored.forecast(horizon=20)
torch.testing.assert_close(actual, expected, rtol=0, atol=0)
```

You do not rebuild the graph. The file contains the model's construction,
weights, fitted preprocessing, named ports, feedback settings, prediction leads,
observed history and current reservoir memory. Presets and custom `rd.Model`
DAGs use the same commands. Parallel and coupled graphs keep their geometry and
member structure. Trajectory sampling metadata and signal names are retained;
a live TSDynamics system object is not bundled into portable files.

A loaded model remains an ordinary `nn.Module`, so `.to(device)`, `.float()`,
`model(x)` and integration into a larger PyTorch network work as usual.
`state_dict()` remains useful when the surrounding application already owns
its architecture. Complete artifacts do not store your optimizer, autograd
graph, compiled executors, active execution hooks or live Optuna study connection.
Built-in checkpoint integrity hooks remain available.

## Reservoir memory and forecasts

Saving includes current reservoir memory by default. After processing one
chunk, saving and loading, an ordinary `restored(next_chunk)` continues from
that memory. Use `model.reset_state()` for an unrelated sequence, or save with
`include_state=False` to start with fresh recurrent memory on reload. Fitted
weights, preprocessing and observed training history are retained either way.
Moving between device types can restart a backend-specific noise RNG stream;
the loader reports that limitation. Stored memory still moves with the tensors.

A forecast uses its saved or explicitly supplied history to synchronize and
construct delayed feedback queues. A checkpoint does not save an in-progress
forecast iterator or its clock. Do not treat ordinary reservoir state as an
autonomous forecast cursor. Temporal layers that need full history may produce
larger files because that history is part of their forecasting context.

## Diagrams and formats

```python
model.save("structure.dot")  # text; no visualization dependencies
model.save("structure.svg")
model.save("structure.png")
model.save("structure.pdf")
model.save("wide", format="svg", rankdir="LR", show_trainable=True)
source = model.plot_model()  # notebook-displayable Graphviz source
assert "pt" in model.save_formats
```

Install `pip install 'resdag[viz]'` and the Graphviz system executable to render
SVG, PNG or PDF (`brew install graphviz` on macOS; `apt install graphviz` on
Debian/Ubuntu). DOT needs neither. The suffix selects the format; a name without
a suffix defaults to `.pt`, or gains the explicit requested format. A format
that disagrees with the suffix raises an error. Exports write one file at the
returned path and never launch a system viewer.

Diagrams show branches, merges, layer names, feature widths, named input/output
ports and delayed feedback. Coupled members have separate clusters; parallel
graphs show periodic patch geometry and the shared reservoir/readout. Graph
inspection does not execute layers, advance reservoir state or change modes.

## Custom components: portable registration

A portable file contains configuration and tensors, with versioned component
identifiers. Built-in components are registered by ResDAG. For your own class,
register a small recipe that describes how to construct it. Learned parameters
and buffers are handled separately by the standard Torch state mechanism.

```python
from torch import nn

class Gain(nn.Module):
    supports_framewise = True

    def __init__(self, factor=2.0):
        super().__init__()
        self.factor = float(factor)

    def forward(self, x):
        return self.factor * x

rd.register_component(
    "myproject.Gain", Gain,
    encode=lambda layer: {"factor": layer.factor},
    decode=lambda config: Gain(**config),
    version=1,
)

x = rd.reservoir_input(1, dtype=torch.float64)
y = rd.SVDReadoutLayer(1, 1).double()(Gain(2.0)(x))
custom = rd.Model(inputs={"signal": x}, outputs={"prediction": y}, washout=5)
custom.fit(series)
custom.save("custom.pt")
loaded = rd.load("custom.pt")
assert loaded.input_names == ("signal",)
```

Import your registration module before loading in another process. Configuration
can contain tensors, primitive values and other registered components. Use a
stable project-specific name and increase the codec version when its contract
changes. Unknown classes or versions fail with a registration remedy; the file
does not import arbitrary Python paths. Custom subclasses require their own
registration so custom behavior cannot silently disappear. Initializer
registration remains the separate weight-generation extension point. A trained
raw graph can restore its weights without an initializer-only callback; a
callback without a portable codec may be omitted from that raw graph's recipe.
Register its codec or use a trusted bundle to retain that recipe for reuse.
Presets retain their factory recipe for tuning and reject unsupported recipe
components instead of silently dropping them.

You can provide a local `ComponentCodec` through
`rd.load(path, custom_objects={"myproject.Gain": codec})` instead of changing
the global registry. Its identifier, exact class and version must match.
The decoder is application code you have explicitly installed.

Portable loading restores realized weights without rerunning original initializer
callbacks. Some structured cells still perform constructor work, such as QR
factorization or Legendre discretization, before those weights are restored.
Treat loading as setup work when measuring performance.

## Notebook-defined Python code

For experiments that define classes or functions inside a notebook, an optional
bundle can carry that Python code too:

```python
# Requires pip install 'resdag[artifacts]'.
custom.save("notebook.pt", trusted=True)
notebook_model = rd.load("notebook.pt", trusted=True)
```

Both operations require the explicit flag. Only load a trusted bundle when
you trust its producer: restoring bundled Python executes code. Normal portable
checkpoints use restricted Torch loading and require no bundled Python.

### Low-level graph checkpoints

The raw graph, parallel and ensemble classes also provide weight checkpoints
through `save`/`load`. Their metadata and custom extra state must contain tensors
and primitive containers; unsupported Python objects are rejected before saving.
Loading uses Torch's restricted `weights_only=True` mode. For complete portable
models, prefer the common `Model.save`/`rd.load` workflow above.

Raw `save_full` files pickle Python. Their corresponding `load_full` methods
require `trusted=True` explicitly, before reading the file. Use this only for a
known producer; the portable artifact format is the default recommendation.
Bundles use the standard cloudpickle package and need a compatible Python and
dependency environment; keep experiment environments pinned when archiving.

## Metadata and failure behavior

```python
model.save("annotated.pt", metadata={"experiment": "sine", "seed": 7})
loaded, metadata = rd.load("annotated.pt", return_metadata=True)
assert metadata == {"experiment": "sine", "seed": 7}
```

Metadata uses primitive values and tensors. Saves write to a temporary file and
replace the destination only after successful serialization. Unsupported formats,
unknown artifact versions and missing components raise useful errors. Keep an
independent copy of valuable experiments and their environment specification.
