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
- **Tensors are batch-first 3-D.** Every signal is `(batch, time, features)`.
  Both libraries above are largely `(time, features)` single-trajectory; add a
  leading batch axis on the way in (`series.unsqueeze(0)`).
- **The readout is solved, not a node you `fit`.** `ESNTrainer.fit` warms the
  reservoir and solves every readout in one pass; the readout's `alpha` (ridge
  λ) lives on the layer, not the trainer.

The rest of this page maps concepts and API surface side by side, then walks
the same Lorenz forecast through all three libraries.

## Coming from reservoirpy

reservoirpy composes `Node`s — `Reservoir`, `Ridge`, `Input` — with the `>>`
pipe operator, or wraps the common case in the `ESN` class. ResDAG composes
layers with the functional API and wraps the model in `ESNModel`.

### Concept mapping

| reservoirpy | ResDAG | Notes |
| --- | --- | --- |
| `Node` (`Reservoir`, `Ridge`, …) | `nn.Module` layer (`ESNLayer`, `CGReadoutLayer`, …) | ResDAG layers are plain PyTorch modules. |
| `reservoir >> readout` | `readout_layer(reservoir_layer(inp))` | Functional wiring; the model is then `ESNModel(inp, out)`. |
| `Model` | `ESNModel` | Wraps a `pytorch_symbolic` graph; `nn.Module` subclass. |
| `ESN(...)` convenience class | a premade model, e.g. [`classic_esn(...)`](build/architectures/classic-esn.md) / [`ott_esn(...)`](build/architectures/ott-esn.md) | One-call factories returning a ready `ESNModel`. |
| `node.fit(X, Y, warmup=…)` | `ESNTrainer(model).fit(warmup_inputs=…, train_inputs=…, targets=…)` | Warmup is a separate tensor, not a step count. |
| `model.run(X)` | `model(X)` (one-step) / `model.forecast(...)` (autoregressive) | `run` is teacher-forced; `forecast` is the closed-loop rollout. |
| `readout.Wout`, `readout.bias` | `readout.weight`, `readout.bias` | Standard `nn.Linear`-style parameters. |
| `to_forecasting(X, forecast=1)` | `rd.utils.prepare_esn_data(data, …)` | Cuts warmup/train/target/val with the +1 shift built in. |

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
| `warmup=100` (a step count) | a warmup *tensor* passed to `fit`/`forecast` |

### Side by side

The same Lorenz one-step forecast. **reservoirpy:**

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

**ResDAG** — note the leading batch axis and the warmup *tensor*:

<div class="nb-specimen" data-label="from_reservoirpy.py" markdown>

```python
import torch
import resdag as rd
from resdag import ESNLayer, ESNModel, reservoir_input, lorenz
from resdag.layers import CGReadoutLayer

data = lorenz(5000, seed=42)               # (1, 5000, 3) — batch-first, already 3-D
warmup, train, target, f_warmup, val = rd.utils.prepare_esn_data(
    data, warmup_steps=100, train_steps=3000, val_steps=1000, normalize=True
)

inp = reservoir_input(3)
states = ESNLayer(300, feedback_size=3, spectral_radius=0.9, leak_rate=0.3, seed=42)(inp)
out = CGReadoutLayer(300, 3, name="output", alpha=1e-6)(states)
model = ESNModel(inp, out)

rd.ESNTrainer(model).fit(
    warmup_inputs=(warmup,),               # a tensor, not a step count
    train_inputs=(train,),
    targets={"output": target},            # keyed by the readout's name
)
forecast = model.forecast(f_warmup, horizon=200)   # closed-loop autoregression
```

</div>

The closest one-call analogue to reservoirpy's `ESN` wrapper is a
[premade model](build/architectures/index.md): `rd.classic_esn(reservoir_size=300,
feedback_size=3, output_size=3)` returns a wired `ESNModel`, leaving only the
`fit` / `forecast` calls.

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
