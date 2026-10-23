---
description: A migration guide for users coming from reservoirpy or ReservoirComputing.jl — side-by-side concept and API mapping tables, plus the differences in mental model that ResDAG asks you to keep.
---

<span class="nb-kicker">Migrating</span>

# Coming from reservoirpy or ReservoirComputing.jl

If you have built echo-state networks in
[reservoirpy](https://github.com/reservoirpy/reservoirpy) (Python/NumPy) or
[ReservoirComputing.jl](https://github.com/SciML/ReservoirComputing.jl)
(Julia/SciML), the building blocks of ResDAG will feel familiar — a reservoir,
a ridge readout, a warmup, a forecast. What differs is the *mental model*:

- **Models are PyTorch modules, not a bespoke graph runtime.** A ResDAG model
  is an ordinary `nn.Module`. It moves with `.to("cuda")`, serializes with
  `state_dict()`, and embeds inside a larger network. There is no separate
  "node" abstraction or compile step.
- **Composition is the API, not a feature.** Where reservoirpy connects nodes
  with `>>` and RC.jl wires fixed `ESN` slots, ResDAG builds an arbitrary DAG
  with the `pytorch_symbolic` functional API. State augmentation, parallel
  reservoirs, and multiple readout heads are a few lines of composition.
- **The common workflow accepts 2-D or batch-first 3-D tensors.** Use
  `(time, features)` for one trajectory or `(batch, time, features)` for a batch.
  Raw graph APIs use the 3-D layout.
- **Start with `model.fit(data)`.** `rd.ESN` handles shifted targets, washout and
  training-only scaling. A custom `rd.Model` uses the same commands. Raw
  `ESNTrainer` and graph factories remain available for explicit data windows.

The rest of this page maps concepts and API surface side by side, then walks
the same Lorenz forecast through all three libraries.

## Phase-five compact weights and PyTorch hooks

For `sparse=True`, recurrent weights now live only in compressed component
buffers. `cell.weight_hh` is a CSR Tensor view rather than a dense Parameter;
use `.to_dense()` for explicit inspection and `.values()` for existing edge
updates. State dictionaries store compressed arrays instead of a dense square
matrix. Dense models retain their existing representation. See
[memory and sparse models](workflows/sparse-reservoirs.md) for initialization,
device, numerical and trainability details.

Forecasting now honors native PyTorch hooks, including output-changing hooks on
layers and the outer model. Models with hooks use a compatible execution path;
hook-free models retain the lean path. `compile=True` with relevant hooks warns
and uses eager execution. Existing forecasts that relied on hooks being ignored
can change. Put recurrent memory in declared DAG nodes so explicit state and
checkpointing can track it; arbitrary Python state hidden in hooks is not a
model-state interface. See [execution capabilities](advanced/forecast-capabilities.md).

`readout="lsmr"` is a new explicit low-memory CPU solver. Existing solver defaults
remain unchanged; [its guide](reference/lsmr-readout.md) explains convergence and
accelerator transfers.

### Independent initialization streams

New ESN, continuous, Euler, edge-of-stability, residual ESN, gated reservoir and
LIF models now use separate random streams for feedback, external inputs,
recurrent weights and bias, including the gated cell's candidate and gate roles.
Previously, mixing a named or custom initializer with a default could
restart the same random sequence: a bias could repeat a feedback prefix, or
recurrent weights could reuse feedback draws. Changing one role no longer
shifts the other roles, even when an optional input is added or removed.

**New seeded models can have different weights and forecast results.** This
includes graph-based recurrent initializers, whose fallback seed now belongs
to the recurrent role. Repeat comparisons and tuning for newly constructed
models; do not treat older accuracy measurements as evidence for these new
initializations. Loading an existing checkpoint keeps its materialized weights
and reservoir state; it does not replace them with new seeded draws.

An integer seed fixes every local role without consuming the global RNG.
Passing a `torch.Generator` now uses its `initial_seed()` without advancing it,
matching the named-initializer convention. With `seed=None`, the cell draws
one master integer from Torch's CPU global RNG and derives its local roles
from that value; `torch.manual_seed` still reproduces construction. Explicit
initializer seeds, including an explicitly bound `seed=None`, and RNGs owned
by custom initializer objects or callbacks keep their own policies. The
standalone `create_torch_generator` utility still returns and advances an
explicitly supplied generator when callers use it for draws.
Advancing a generator does not change its `initial_seed()`. For distinct models
within one experiment, pass different integer seeds or use
[`SeedStreams`](advanced/custom-components.md#give-branches-independent-repeatable-seeds).

## Phase-four training and persistence changes

The common `rd.ESN` and `rd.Model` now support `model.save("run.pt")` and
`rd.load("run.pt")`, including the complete graph. Use `.svg`, `.png`, `.pdf`
or `.dot` for diagrams. Portable custom components need registration; optional
notebook-code bundles require explicit `trusted=True` at save and load.
See [save and load](workflows/save-load.md).

Batch fitting evaluates one continuous warmup-plus-training prefix. Readout
regression still excludes warmup, and downstream reservoirs use fitted upstream
outputs throughout the prefix. Temporal layers now see one sequence instead of
a separate warmup call and training call. Ordinary hooks consequently observe
one combined graph call. Stochastic layers, including reservoir noise and dropout, follow the new traversal order,
so old noisy-training results can change even for the same seed; new seeded
runs remain repeatable. Custom outer `ESNModel.forward` bodies are rejected
rather than bypassed; put custom behavior in graph operations or module hooks.

A readout object may fan out from one invocation but cannot occupy several
fitted graph positions. Use distinct readout instances there. Internal readouts
can receive `readout_targets={"name": values}` without becoming public outputs.
The raw trainer mutates in place; the common workflow commits a fit only after
it succeeds. `fit_stream` retains its independent incremental-readout contract.

Raw `plot_model(save_path=...)` now infers the suffix and returns the saved path;
without a path it returns a notebook-displayable source. It no longer opens a
system viewer or silently writes SVG when a PNG filename was requested.

## Forecast timing after v0.11

The default low-level forecast now starts at the **immediate unseen sample**
after all observations. It agrees with the common workflow and with next-sample
training. Previously, core and coupled forecasts skipped that first prediction.

```python
pred = model.forecast(observed, horizon=H)
truth = full_series[:, observed.shape[1]:observed.shape[1]+H]
```

`prepare_esn_data` now includes the final supervised target in forecast warmup.
Validation still starts after that target, and training/target/validation
windows are unchanged. Prepared predictions continue to align directly with
`val`; remove any manual shift added around older custom observed histories.

Other intentional interface changes:

- Future drivers begin after observed history and need `H-1` samples, since the
  final observed driver already produced prediction zero. For `H=1`, omit them.
- `initial_feedback` is a distinct additional input after full warmup. It needs
  `H` drivers. Do not repeat the final warmup observation as this seed.
- `return_warmup=True` has length `W-1+H`, since the boundary prediction occurs
  once. Explicit seeds and legacy alignment retain length `W+H`.
- ParallelESN already used immediate unseen timing. Its default future trajectory
  keeps that convention; its combined prefix now contains historical predictions
  instead of raw observations, with the same `W-1+H` length as other models.

To reproduce an older prepared-data core or ensemble experiment:

```python
warmup, train, target, f_warmup, val = rd.prepare_esn_data(
    data, warmup_steps=W, train_steps=L, val_steps=H, alignment="legacy",
)
rd.ESNTrainer(model).fit((warmup,), (train,), {"output": target})
pred = model.forecast(f_warmup, horizon=H, alignment="legacy")
```

Both legacy arguments matter: preparation restores the earlier history window,
and forecasting restores its skipped-boundary rollout. Golden fixtures for the
new default use version 2, with unchanged numerical tolerances and quality
rules; frozen version 1 fixtures additionally check legacy reproduction. The
migration is backed by exact integer and driver recurrence tests before fixture
regeneration. [Timing](theory/timing.md) explains the indices and saved-state
continuation.

## Coming from reservoirpy

reservoirpy composes `Node`s — `Reservoir`, `Ridge`, `Input` — with the `>>`
pipe operator, or wraps the common case in the `ESN` class. ResDAG composes
layers with the functional API and wraps them in named `rd.Model` inputs and
outputs. Use `rd.ESN` for a preset with the same fitting/forecasting interface.

### Concept mapping

| reservoirpy | ResDAG | Notes |
| --- | --- | --- |
| `Node` (`Reservoir`, `Ridge`, …) | `nn.Module` layer (`ESNLayer`, `CGReadoutLayer`, …) | ResDAG layers are plain PyTorch modules. |
| `reservoir >> readout` | `readout_layer(reservoir_layer(inp))` | Functional wiring; wrap with `rd.Model(inputs={"state": inp}, outputs={"prediction": out})`. |
| `Model` | `rd.Model` | Named functional graph with the common PyTorch workflow. |
| `ESN(...)` convenience class | `rd.ESN(...)` | PyTorch preset with training-only scaling and `fit`/`forecast`. |
| `node.fit(X, Y, warmup=…)` | `model.fit(X, targets=Y)` with constructor `washout=…` | Explicit targets are already aligned; omit them for automatic shifted targets. |
| `model.run(X)` | `model(X)` (one-step) / `model.forecast(...)` (autoregressive) | `run` is teacher-forced; `forecast` is the closed-loop rollout. |
| `readout.Wout`, `readout.bias` | `readout.weight`, `readout.bias` | Standard `nn.Linear`-style parameters. |
| `to_forecasting(X, forecast=1)` | `model.fit(observed)` or `rd.prepare_esn_data(data, …)` | Automatic shifted targets in the common workflow; explicit windows for raw trainers. |

### Parameter mapping

| reservoirpy (`Reservoir` / `Ridge`) | ResDAG (`ESNLayer` / `CGReadoutLayer`) |
| --- | --- |
| `units=100` | `reservoir_size=100` |
| `sr=0.9` (spectral radius) | `spectral_radius=0.9` |
| `lr=0.3` (leak rate) | `leak_rate=0.3` |
| `input_scaling=1.0` | `input_initializer=("random", {"input_scaling": 1.0})` |
| `rc_connectivity=0.1` | `topology=("erdos_renyi", {"p": 0.1})` |
| `seed=1234` | `seed=1234` |
| `Ridge(ridge=1e-7)` | `CGReadoutLayer(..., alpha=1e-7)` |
| `warmup=100` (a step count) | `washout=100` on `rd.ESN`/`rd.Model`; separate warmup tensors for raw trainers |

### Side by side

The following examples show the two APIs. The reservoirpy example evaluates
teacher-forced predictions; the ResDAG example rolls forward autonomously.
They illustrate usage, not an accuracy comparison. **reservoirpy:**

```python
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.datasets import lorenz, to_forecasting

X = lorenz(n_timesteps=5000)
x_train, x_test, y_train, y_test = to_forecasting(X, forecast=1, test_size=0.2)

reservoir = Reservoir(units=300, sr=0.9, lr=0.3, seed=42)
readout = Ridge(ridge=1e-6)
esn = reservoir >> readout

esn.fit(x_train, y_train, warmup=100)
predictions = esn.run(x_test)
```

**ResDAG** — fitting prepares the targets and scaling automatically:

<div class="nb-specimen" data-label="from_reservoirpy.py" markdown>

```python
import torch
import resdag as rd

data = rd.lorenz(5000, seed=42).double()  # (1, 5000, 3)
model = rd.ESN(
    reservoir_size=300, input_size=3, washout=100,
    spectral_radius=0.9, leak_rate=0.3, alpha=1e-6,
    seed=42, dtype=torch.float64,
)
model.fit(data[:, :4000])
forecast = model.forecast(horizon=200)    # estimates data[:, 4000:4200]
```

</div>

The closest one-call analogue to reservoirpy's `ESN` is `rd.ESN`. Build custom
branches and heads with [the same common model workflow](start/model-workflow.md).
For explicit low-level training, the [graph factories](build/architectures/index.md)
return raw `ESNModel` instances that work with `ESNTrainer`.

## Coming from ReservoirComputing.jl

ReservoirComputing.jl (current `LuxCore`-based API) builds an `ESN` from
initializer functions, runs `setup` to obtain parameters and state, trains with
`train!`, and forecasts with `predict` in a `Generative` rollout. ResDAG folds
the parameter/state handling into the stateful `nn.Module` and the trainer.

### Concept mapping

| ReservoirComputing.jl | ResDAG | Notes |
| --- | --- | --- |
| `ESN(in, res, out; …)` | `ESNLayer(res, feedback_size=in)` + `CGReadoutLayer(res, out)` | RC.jl bundles reservoir + readout in one type; ResDAG keeps them as composable layers. |
| `init_reservoir = rand_sparse(; radius, sparsity)` | `spectral_radius=…`, `topology=("erdos_renyi", {"p": …})` | Radius → spectral radius; sparsity → topology density. |
| `init_input = weighted_init(; scaling)` / `scaled_rand` | `input_initializer` / `feedback_initializer` | A named [initializer](build/initialization/initializers/index.md). |
| `state_modifiers = NLAT2` | a transform layer or premade model (e.g. `ott_esn`) | Nonlinear state augmentation is a [composed layer](build/architectures/index.md). |
| `ps, st = setup(rng, esn)` | *(implicit)* — the layer owns its parameters and state | No separate params/state objects; it is a stateful module. |
| `train!(esn, X, Y, ps, st, StandardRidge(λ))` | `ESNTrainer(model).fit(warmup_inputs=…, train_inputs=…, targets=…)` | `λ` of `StandardRidge` ↦ `alpha` on the readout. |
| `predict(esn, len, ps, st; initialdata=…)` | `model.forecast(warmup, horizon=len)` | `initialdata` ↦ the warmup tensor that re-syncs the reservoir. |
| `Generative` prediction mode | `model.forecast(...)` (closed loop) | The autoregressive rollout. |
| `Predictive` prediction mode | `model(inputs)` (teacher-forced one-step) | Run the model directly on the inputs. |

### Parameter mapping

| ReservoirComputing.jl | ResDAG |
| --- | --- |
| `res_size` (e.g. `300`) | `reservoir_size=300` |
| `rand_sparse(; radius=1.2)` | `spectral_radius=1.2` |
| `rand_sparse(; sparsity=6/300)` | `topology=("erdos_renyi", {"p": 6/300})` |
| `weighted_init(; scaling=0.1)` | `input_initializer=("random", {"input_scaling": 0.1})` |
| `StandardRidge(1e-6)` | `CGReadoutLayer(..., alpha=1e-6)` |
| `predict(esn, predict_len, …)` | `model.forecast(warmup, horizon=predict_len)` |

### Side by side

The same Lorenz forecast. **ReservoirComputing.jl** (current API):

```julia
using ReservoirComputing, Random

esn = ESN(3, 300, 3;                       # in_size, res_size, out_size
    init_reservoir = rand_sparse(; radius = 1.2, sparsity = 6 / 300),
    init_input = weighted_init(; scaling = 0.1),
    state_modifiers = NLAT2,
)
ps, st = setup(MersenneTwister(17), esn)
ps, st = train!(esn, input_data, target_data, ps, st, StandardRidge(1e-6))
output, st = predict(esn, predict_len, ps, st; initialdata = test_data[:, 1])
```

**ResDAG** — the reservoir and readout are separate composable layers, and the
parameter/state bookkeeping (`ps`, `st`, `setup`) is folded into the stateful
module:

<div class="nb-specimen" data-label="from_reservoircomputing_jl.py" markdown>

```python
import torch
import resdag as rd
from resdag import ESNLayer, ESNModel, reservoir_input, lorenz
from resdag.layers import CGReadoutLayer

data = lorenz(5000, seed=17)               # (1, 5000, 3)
warmup, train, target, f_warmup, val = rd.utils.prepare_esn_data(
    data, warmup_steps=100, train_steps=3000, val_steps=1000, normalize=True
)

inp = reservoir_input(3)
states = ESNLayer(
    300, feedback_size=3,
    spectral_radius=1.2,                                   # rand_sparse radius
    topology=("erdos_renyi", {"p": 6 / 300}),             # rand_sparse sparsity
    feedback_initializer=("random", {"input_scaling": 0.1}),  # weighted_init scaling
)(inp)
out = CGReadoutLayer(300, 3, name="output", alpha=1e-6)(states)   # StandardRidge(1e-6)
model = ESNModel(inp, out)

rd.ESNTrainer(model).fit(
    warmup_inputs=(warmup,),
    train_inputs=(train,),
    targets={"output": target},
)
output = model.forecast(f_warmup, horizon=200)            # predict(...; Generative)
```

</div>

The `NLAT2` nonlinear state-augmentation modifier has a direct analogue in
ResDAG: [`ott_esn`](build/architectures/ott-esn.md) squares the even-indexed reservoir
features and concatenates them back — the same trick used to break the
odd symmetry that hampers Lorenz forecasting — as a composed architecture
rather than a modifier flag.

## What to read next

- [Start · The mental model](start/concepts.md) — the DAG, the stateful reservoir, the solved readout
- [Build · Architectures](build/architectures/index.md) — composing reservoirs and heads into a DAG
- [Work · Train](workflows/train.md) and [Forecast](workflows/forecast.md) — the full fit/forecast cycle
- [Project · Related work](project/related-work.md) — where ResDAG sits in the landscape
