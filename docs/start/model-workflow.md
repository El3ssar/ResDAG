---
description: Fit a named functional graph, forecast in original units, and use the trained model inside PyTorch.
---

# The common model workflow

`rd.Model` adds training, scaling and forecasting around a functional graph.
It is an ordinary PyTorch module. You keep control of the layers, their
initializers and the readout solver.

Inputs are floating Torch tensors shaped `(time, features)` or
`(batch, time, features)`, on the same device and dtype as the graph. Named
multi-input data uses a dictionary of those tensors. `rd.ESN` provides the same
Torch workflow for a ready-made classic reservoir. Convert NumPy arrays with
`torch.as_tensor` at the boundary. `fit` and `forecast` also accept
[TSDynamics trajectories](trajectory-workflow.md), including
[named trajectories for several ports](named-trajectories.md).
Ordinary `model(x)` calls retain the tensor interface.

This is part of [phase two](../project/roadmap.md#phase-2-one-functional-workflow).
The common interface supports the `ESN` preset, symbolic, parallel and coupled
graphs. Raw `ESNModel` and factories remain useful building blocks. Unified tuning and complete
model/diagram saves are later roadmap phases.

## Build, fit and forecast

The inputs are named signals. The outputs are named predictions. Connect layers
with `layer(x)`, then give those connections to `rd.Model`.

```python
import torch
import resdag as rd

# A small example with two signals in different physical units.
t = torch.arange(400, dtype=torch.float64) * 0.05
series = torch.stack((10 + 2 * t.sin(), -3 + 0.5 * t.cos()), dim=-1)
train, truth = series[:300], series[300:340]

x = rd.reservoir_input(2, dtype=torch.float64)
h = rd.ESNLayer(64, feedback_size=2, seed=7, dtype=torch.float64)(x)
y = rd.SVDReadoutLayer(64, 2, alpha=1e-6).double()(h)
model = rd.Model(inputs={"state": x}, outputs={"prediction": y}, washout=30)

model.fit(train)
prediction = model.forecast(history=train[-60:], horizon=40)
assert prediction.shape == truth.shape
assert torch.isfinite(prediction).all()
```

For one input and one output of the same width, `fit(train)` learns to predict
the next sample. It uses the first `washout` samples to synchronize the
reservoir, then fits the readout. Pass only training data to `fit`; keep
validation and test samples separate.

Standard scaling is automatic. Its statistics come from the supervised
training inputs, excluding washout and the final shifted target. Forecasts are
returned in the signals' original units. The first forecast sample estimates
the sample immediately after the supplied history.

Omit `history` to continue from the saved tail of the training data. Supplying
history resynchronizes the reservoir on that history before forecasting.
Choose `preprocessing=None` to handle scaling yourself, or choose an existing
normalization method: `"standard"`, `"minmax"`, `"noncentered"` or
`"meanpreserving"`.

Override individual ports with `input_transforms={"state": your_module}` and
`output_transforms={"prediction": your_module}`. Each module preserves the
signals and provides a reversible change of units. Use a distinct instance
for each port; `torch.nn.Identity()` disables preprocessing on one port.
Explicit overrides still apply when the global policy is off. See
[custom preprocessing](../advanced/custom-preprocessing.md) for a complete
example, fitting rules and checkpoints. Operations that add features or memory
belong inside the DAG as ordinary layers.

## Use a preset graph

Wrap a factory-built graph with the same model interface:

```python
graph = rd.classic_esn(
    reservoir_size=64, feedback_size=2, output_size=2,
    readout="svd", seed=7, dtype=torch.float64,
)
preset = rd.Model(graph=graph, washout=30)
preset.fit(train)
preset_prediction = preset.forecast(history=train[-60:], horizon=40)
assert preset_prediction.shape == truth.shape
```

`model.graph` gives access to the exact underlying graph and its parameters.
It works in the graph's raw, preprocessed units. Use `model(...)` when your
pipeline supplies and expects the original units.

## Parallel and coupled models

The same wrapper works for a spatial field split into overlapping patches:

```python
parallel = rd.Model(
    graph=rd.parallel_esn(
        input_dim=2, core_width=1, halo=1, reservoir_size=32,
        readout="svd", seed=7, dtype=torch.float64,
    ),
    washout=30,
)
parallel.fit(train)
parallel_prediction = parallel.forecast(history=train[-60:], horizon=40)
assert parallel_prediction.shape == truth.shape
```

Inputs and returned predictions use the full field width. Scaling is learned
per field component, before the backend gathers local patches. Each patch
keeps its own recurrent state; reservoir and readout weights remain shared.

For an ensemble whose members share one aggregate feedback signal:

```python
coupled = rd.Model(
    graph=rd.coupled_ensemble_esn(
        n_models=3, model_factory=rd.classic_esn,
        reservoir_size=32, feedback_size=2, output_size=2,
        readout="svd", seed=7, dtype=torch.float64,
    ),
    washout=30,
)
coupled.fit(train)
coupled_prediction = coupled.forecast(history=train[-60:], horizon=40)
assert coupled_prediction.shape == truth.shape
```

The coupled backend retains its existing single aggregate output. Each
member fits its own head, even when internal readout names differ. Inspect
individual member forecasts through `coupled.graph`; the common workflow
returns the aggregate. Parallel and coupled `forecast` currently require
`compile=False`. Unsupported compiled forecasts fail before running the graph.

Both families use the same original-unit forward calls, carried state,
explicit-state calls, dtype/device moves and state dictionaries as symbolic
graphs. Rebuild the same architecture before loading a state dictionary,
including custom aggregator settings. Its schema checks structural facts,
not arbitrary Python configuration. Complete architecture reconstruction
from one saved file remains phase four.

## Give several heads explicit targets

Name each output and pass matching targets. Explicit targets line up with the
supplied input samples; `fit` does not shift them. Here we supply that shift
ourselves and add a second quantity to predict:

```python
state = rd.reservoir_input(2, dtype=torch.float64)
features = rd.ESNLayer(32, feedback_size=2, seed=11, dtype=torch.float64)(state)
position = rd.SVDReadoutLayer(32, 2, alpha=1e-6).double()(features)
energy = rd.SVDReadoutLayer(32, 1, alpha=1e-6).double()(features)
multi = rd.Model(
    inputs={"state": state},
    outputs={"position": position, "energy": energy},
    washout=30,
)
next_state = train[1:]
multi.fit(
    {"state": train[:-1]},
    targets={
        "position": next_state,
        "energy": next_state.square().sum(dim=-1, keepdim=True),
    },
)
named = multi({"state": train[-20:]})
assert named["position"].shape == (20, 2)
assert named["energy"].shape == (20, 1)
```

Each target gets its own training-only scaling. The user-facing output names
do not need to match the graph's generated readout names. Feedback converts
the selected prediction through original units before applying the receiving
input's scaling. See [feedback and timing](feedback.md) for named connections,
delays and predicting several samples ahead.

## Put the trained model inside PyTorch

`model(x)` returns a tensor for a single output, preserves gradients and
**carries reservoir state between calls**. Reset before starting an unrelated
sequence. `train()` and `eval()` do not reset that state.

```python
from torch import nn

pipeline = nn.Sequential(
    nn.Linear(2, 2, dtype=torch.float64),
    model,
    nn.Linear(2, 1, dtype=torch.float64),
)
model.reset_state()
batch = train[-20:].unsqueeze(0).detach().requires_grad_()
loss = pipeline(batch).square().mean()
loss.backward()
assert batch.grad is not None
assert torch.isfinite(batch.grad).all()
```

Frozen reservoir weights still pass gradients to the preceding layer. Stored
reservoir state follows the layers' existing detach policy at call boundaries.
For explicit streaming state and gradients across chunks, use
`prediction, state = model.forward_stateless(chunk, states=previous_state)`.
That path leaves stored reservoir state unchanged.

Move the complete model with ordinary `.to(device)` or `.double()` calls;
fitted scaling statistics move with it. Scaling and the graph are included in
`state_dict()`. As with ordinary PyTorch checkpoints, loading a state dict still
requires a compatible model architecture. Loading a complete model without
rebuilding its graph remains [phase four](../project/roadmap.md#phase-4-diagrams-and-complete-model-artifacts).

On Apple MPS, float32 fitting with `CGReadoutLayer(..., use_float64=False)` is
tested. MPS does not support float64 or the direct Cholesky-solve operation
used by some readouts. You can also fit on CPU, then move the trained model
with `model.float().to("mps")`. Automatic device-aware solver selection remains
in [phase three](../project/roadmap.md#phase-3-tuning-and-automatic-solver-policy).

## Capture the surrounding pipeline with FX

If a larger pipeline uses PyTorch FX, keep the reservoir as one module call.
Use FX's standard leaf-module hook:

```python
from torch import fx

class ReservoirTracer(fx.Tracer):
    def is_leaf_module(self, module, qualified_name):
        return isinstance(module, (rd.Model, rd.ESN)) or super().is_leaf_module(
            module, qualified_name
        )

captured = fx.GraphModule(pipeline, ReservoirTracer().trace(pipeline))
assert captured.get_submodule("1") is model
model.reset_state()
expected = pipeline(batch.detach())
model.reset_state()
actual = captured(batch.detach())
torch.testing.assert_close(actual, expected)
```

Capture preserves the same model, parameters and carried state. It does not
execute the reservoir during tracing. Direct tracing through `rd.Model` and
general deployment export are not supported; see the
[graph-tooling review](../project/graph-tooling.md) for the evaluated boundaries.

## Current boundaries

- The common wrapper supports symbolic `ESNModel`, `ParallelESNModel` and
  `CoupledEnsembleESNModel` graphs. Their original low-level APIs remain available.
- Automatic next-sample targets require one unambiguous input/output pair.
  Several heads need explicit named targets; their predictions are returned
  as a dictionary. Expose each algebraically fitted readout as a named output;
  an intermediate head can also feed a downstream reservoir.
- Default feedback connects the first output to the first input. Custom
  [named routes](feedback.md) support fanout and explicit delays. Inputs
  without a feedback route need future driver values.
- Reservoirs must be separate functional nodes for explicit-state and forecast
  execution. See [state management](concepts.md#2-reservoir-layers-are-stateful).
- Ordinary forward calls run module hooks. The native flat fast forecast path
  runs hooks during history warmup and bypasses them for generated steps.
  Named/delayed feedback and full-history execution use the graph executor;
  operation hooks also run when that executor evaluates generated steps.
- Fast forecasting requires framewise operations or explicit reservoir nodes.
  Custom sequence operations can use [history execution](../advanced/forecast-capabilities.md),
  which replays their full growing prefix. Delayed/named feedback uses explicit
  state; reservoir state alone cannot resume an in-progress delayed rollout.
- Fitting stages a copy of the graph before committing trained weights. This
  preserves the existing model after a failed fit and keeps parameter objects
  intact, but requires memory for that temporary copy and the solver workspace.

See the [core reference](../reference/core.md) for signatures and the
[phase-two design note](../project/phase-two-design.md) for agreed behavior and
remaining decisions.
