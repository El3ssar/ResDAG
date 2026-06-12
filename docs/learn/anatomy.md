---
description: The three moving parts of every ResDAG model — stateful reservoir layer, self-fitting readout, and the ESNModel wrapper — and the state that flows between them.
---

<span class="rd-eyebrow">Learn · 03</span>

# Anatomy of an ESN

By the end of this page you'll know exactly what was inside the model you
trained in the quickstart: a stateful reservoir layer, a linear readout
that fits itself algebraically, and an `ESNModel` wrapper orchestrating
both. Three parts, three responsibilities.

<figure markdown>
![Classic ESN architecture](../assets/figures/arch_classic_esn.svg)
<figcaption><code>classic_esn</code> as a DAG: the input drives the
reservoir, then input and states concatenate into the readout.</figcaption>
</figure>

## ESNLayer — the dynamics

<div class="rd-window" data-title="anatomy.py" markdown>

```python
import torch
from resdag.layers import ESNLayer

reservoir = ESNLayer(
    reservoir_size=500,      # neurons
    feedback_size=3,         # dim of the signal being forecast
    spectral_radius=0.9,     # memory vs stability
    leak_rate=1.0,           # 1.0 = standard update
    bias_scaling=1.0,        # random bias in U(-1, 1)
)

feedback = torch.randn(4, 100, 3)    # (batch, time, features)
states = reservoir(feedback)         # (4, 100, 500)

for name, p in reservoir.named_parameters():
    print(name, tuple(p.shape), p.requires_grad)
# cell.weight_feedback (500, 3)   False
# cell.weight_hh       (500, 500) False
# cell.bias_h          (500,)     False
```

</div>

Each timestep applies the leaky-ESN update

$$h_t = (1-\alpha)\,h_{t-1} + \alpha\,\tanh\!\big(W_{fb}\,x_t + W_{rec}\,h_{t-1} + b\big)$$

with $\alpha$ the `leak_rate`. The weights are real `nn.Parameter`s with
`requires_grad=False` — frozen, not fake: they move with `.to("cuda")`,
serialize in `state_dict()`, and unfreeze with `trainable=True` if you ever
want gradients ([chapter 05](training.md)).

Internally the work is split in two. An `ESNCell` owns the weights and the
single-step update $h_{t-1} \to h_t$; the `ESNLayer` owns the loop over
time and the state bookkeeping. You rarely touch the cell directly —
`ESNLayer` delegates unknown attributes to it, so `reservoir.weight_hh` and
`reservoir.reservoir_size` just work.

**The three knobs, at intuition level.** `spectral_radius` rescales
$W_{rec}$ so its largest eigenvalue magnitude hits the target: small means
fast-fading memory and rock-solid stability, near 1.0 means long memory at
the edge of instability. `leak_rate` blends the new state with the old one
— values below 1.0 slow the reservoir down to match slow signals. The bias
$b$ is drawn from $\mathcal{U}(-\beta, \beta)$ with $\beta$ =
`bias_scaling`: since $\tanh$ is odd, a bias-free reservoir maps a negated
input to an exactly negated trajectory, halving the diversity of states the
readout can draw from. The random bias breaks that symmetry. Full equations
[under the hood](../under-the-hood/reservoir-equations.md).

!!! note "Raw layer vs factories"
    `ESNLayer` itself defaults to `spectral_radius=None` (no rescaling).
    The premade factories all pass `0.9`. If you build by hand, set it
    explicitly.

## State: carried, lazy, detached

The reservoir is **stateful** — that is the whole point. Its
`(batch, reservoir_size)` state initializes lazily to zeros on the first
forward pass and then carries across calls:

```python
out_a = reservoir(seq_a)     # state: zeros -> evolved through seq_a
out_b = reservoir(seq_b)     # continues exactly where seq_a left off

saved = reservoir.get_state()        # clone of the live state (None if uninitialized)
reservoir.reset_state()              # back to None; re-zeroed on next forward
reservoir.set_state(saved)           # restore the checkpoint
reservoir.set_random_state(batch_size=4)   # standard-normal state
```

Between calls the stored state is *detached* from the autograd graph —
truncated BPTT at call boundaries. Gradients still flow within a call, but
batch N's backward can never reach into batch N−1's graph, which is exactly
what makes the SGD recipes in [chapter 05](training.md) work without
ceremony.

## CGReadoutLayer — the only thing that learns

A `CGReadoutLayer` is literally an `nn.Linear` with two additions: it
applies itself per timestep — `(batch, time, F)` is flattened to
`(batch*time, F)`, transformed, and reshaped back — and it has a `fit()`
method that solves ridge regression by conjugate gradient instead of
waiting for gradient descent:

```python
from resdag.layers import CGReadoutLayer

readout = CGReadoutLayer(in_features=500, out_features=3,
                         alpha=1e-6, name="output")
readout.fit(states, targets)     # one algebraic solve, no gradients
prediction = readout(states)     # (4, 100, 3)
```

`alpha` is the L2 penalty — the single most important regularizer in the
library. The solve runs in float64 internally for stability, then casts
back. With `bias=True` (default) the data is centered and an unpenalized
intercept recovered; with `bias=False` the uncentered ridge problem is
solved directly. The `name` is how training data finds this layer:
`targets={"output": ...}` in the trainer keys on it.

## ESNModel — the orchestrator

`ESNModel` extends `pytorch_symbolic.SymbolicModel`: you hand it symbolic
inputs and outputs ([chapter 04](building-models.md) is all about this) and
it traces the DAG into a regular `nn.Module`, plus the ESN-specific verbs:

```python
model.reset_reservoirs()              # zero every reservoir in the DAG
model.warmup(feedback)                # teacher-forced state sync, no output
model.forecast(feedback, horizon=300) # warmup + autoregression
model.save("model.pt")                # weights (+ states with include_states=True)
model.load("model.pt")
```

`get_reservoir_states()` / `set_reservoir_states()` snapshot and restore
every reservoir at once — handy for branching forecasts off one
synchronized state. Note that `save`/`load` handle weights only: you
re-create the architecture in code, then load.

## Next

[**04 · Building models**](building-models.md) — wire these parts into
DAGs: parallel reservoirs, augmented states, multiple readouts.
