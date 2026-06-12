---
description: The pytorch_symbolic functional API — wire reservoirs, transforms, and readouts into arbitrary DAGs, or grab a premade factory.
---

<span class="rd-eyebrow">Learn · 04</span>

# Building models

This is the page where ResDAG stops looking like other RC libraries. Every
model — premade or yours — is three steps: declare symbolic inputs, call
layers on them like functions, wrap the result in `ESNModel`. Master that
pattern and the five architectures below are ten-line variations.

```python
import resdag as rd

inp = rd.reservoir_input(features)    # 1. symbolic input
out = layer_chain(inp)                # 2. functional composition
model = rd.ESNModel(inp, out)         # 3. trace the DAG into an nn.Module
```

`rd.reservoir_input(F)` is shorthand for `rd.core.Input((1, F))` — the time
axis is a tracing placeholder; real sequences of any length flow through at
call time.

## 1 — Minimal: one reservoir, one readout

<div class="rd-window" data-title="minimal.py" markdown>

```python
import resdag as rd

inp = rd.reservoir_input(1)
states = rd.ESNLayer(reservoir_size=100, feedback_size=1,
                     spectral_radius=0.9)(inp)
out = rd.CGReadoutLayer(in_features=100, out_features=1,
                        name="output")(states)

model = rd.ESNModel(inp, out)
```

</div>

<figure markdown>
![Minimal ESN architecture](../assets/figures/arch_minimal.svg)
<figcaption>The smallest useful DAG: one signal in, one reservoir, one
readout.</figcaption>
</figure>

Each layer call consumes a symbolic tensor and returns a new one — shapes
are checked at build time, so a mismatched `in_features` fails here, not
three functions deep at runtime.

## 2 — Input-driven: feedback plus exogenous driver

When the system has external forcing — control signals, weather, any
variable you *know* in advance — give the reservoir two inputs. The first
is always feedback (the channel the model predicts and feeds back during
autoregression); the rest are drivers riding alongside.

```python
feedback = rd.reservoir_input(3)
driver = rd.reservoir_input(2)

states = rd.ESNLayer(300, feedback_size=3, input_size=2,
                     spectral_radius=0.9)(feedback, driver)
out = rd.CGReadoutLayer(300, 3, name="output")(states)

model = rd.ESNModel([feedback, driver], out)
```

<figure markdown>
![Input-driven architecture](../assets/figures/arch_input_driven.svg)
<figcaption>Two inputs, one reservoir. Only <code>feedback</code> is
autoregressed; the driver is supplied at forecast time.</figcaption>
</figure>

At forecast time the future driver values go in through
`forecast_inputs` — [chapter 06](forecasting.md) covers the alignment.

## 3 — Ott-style state augmentation

The trick that made reservoir forecasts of chaos famous: square half the
reservoir units before the readout, so the linear readout can access
quadratic features of the state.

```python
inp = rd.reservoir_input(3)
states = rd.ESNLayer(500, feedback_size=3, spectral_radius=0.9)(inp)

augmented = rd.SelectiveExponentiation(index=0, exponent=2.0)(states)
merged = rd.Concatenate()(inp, augmented)
out = rd.CGReadoutLayer(3 + 500, 3, name="output")(merged)

model = rd.ESNModel(inp, out)
```

<figure markdown>
![Ott ESN architecture](../assets/figures/arch_ott_esn.svg)
<figcaption>Squared even-indexed units, concatenated with the raw input —
exactly what the <code>ott_esn</code> factory builds.</figcaption>
</figure>

`SelectiveExponentiation(index=0, exponent=2.0)` squares the even-indexed
features and leaves the rest alone; `Concatenate` joins streams along the
feature axis. This DAG *is* the `ott_esn` factory — the factory just saves
you the typing.

## 4 — Parallel reservoirs: two timescales, one readout

Nothing forces a model to contain one reservoir. A fast reservoir (low
spectral radius, no leak) and a slow one (high radius, heavy leak) capture
different timescales; the readout learns how to combine them.

```python
inp = rd.reservoir_input(3)
fast = rd.ESNLayer(200, feedback_size=3, spectral_radius=0.6, leak_rate=1.0)(inp)
slow = rd.ESNLayer(200, feedback_size=3, spectral_radius=0.95, leak_rate=0.3)(inp)

merged = rd.Concatenate()(fast, slow)
out = rd.CGReadoutLayer(400, 3, name="output")(merged)

model = rd.ESNModel(inp, out)
```

<figure markdown>
![Parallel reservoirs architecture](../assets/figures/arch_parallel_reservoirs.svg)
<figcaption>Both reservoirs see the same input; the readout sees a 400-dim
concatenation of their states.</figcaption>
</figure>

## 5 — Multi-readout: one reservoir, several heads

The reservoir runs once; any number of readouts regress against different
targets from the same states. Each head is fitted independently in a single
training pass ([chapter 05](training.md)).

```python
inp = rd.reservoir_input(3)
states = rd.ESNLayer(400, feedback_size=3, spectral_radius=0.9)(inp)

position = rd.CGReadoutLayer(400, 3, name="position")(states)
energy = rd.CGReadoutLayer(400, 1, name="energy")(states)

model = rd.ESNModel(inp, [position, energy])
```

<figure markdown>
![Multi-readout architecture](../assets/figures/arch_multi_readout.svg)
<figcaption>Shared dynamics, separate heads. Each readout's
<code>name</code> is its key in the training targets dict.</figcaption>
</figure>

!!! warning "Output order matters for forecasting"
    During autoregressive forecasting the **first** output is fed back as
    the next input, so its dimension must equal the feedback input's. Here
    `position` (3 features) matches `inp` (3 features) — listing `energy`
    first would break `forecast()`.

## The premade factories

When the architecture is standard, skip the wiring — every single-model
factory returns a regular `ESNModel` you can train, forecast, and extend
identically (the exception: `coupled_ensemble_esn` returns a
`CoupledEnsembleESNModel` with its own `fit()`/`forecast()` of the same
shape):

| Factory | Architecture |
|---|---|
| `rd.classic_esn` | Input → Reservoir → Concat(input, states) → CGReadout |
| `rd.ott_esn` | Input → Reservoir → SelectiveExponentiation → Concat(input, augmented) → CGReadout |
| `rd.power_augmented` | Like `ott_esn`, but `Power(exponent)` on all units instead of even-index squaring |
| `rd.linear_esn` | Input → Reservoir with `activation="identity"` — states out, no readout |
| `rd.headless_esn` | Input → Reservoir — raw states out, no readout |
| `rd.coupled_ensemble_esn` | N independently seeded sub-models (any factory) coupled through aggregated feedback |

The single-model factories take `reservoir_size`, `feedback_size`, and
(where a readout exists) `output_size`, plus the usual reservoir knobs and
`readout_alpha`, `readout_name`. `coupled_ensemble_esn` takes `n_models`
and forwards everything else to its sub-model factory — its own
[cookbook recipe](../cookbook/ensembles.md) has the details. For chaotic
systems, start with `ott_esn`.

## Next

[**05 · Training**](training.md) — one-pass algebraic fitting, and the two
gradient-descent paths beyond it.
