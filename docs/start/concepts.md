---
description: The four ideas behind ResDAG — frozen dynamics, the state, the algebraic readout, and the two-phase forecast.
---

<span class="nb-kicker">Start · 03</span>

# The mental model

Four ideas. Everything in the library is one of them wearing different
clothes.

## 1 — The dynamics are designed, not learned

A reservoir is a recurrent network whose weights are *chosen* and then
frozen: a connectivity structure (random, a graph, any
[matrix-building function](../build/initialization.md)), scaled to a target
spectral radius, driven by your signal. It is a fixed nonlinear dynamical
system that unfolds your input's history into hundreds of feature
trajectories. Because nothing inside is learned, everything inside is an
experimental knob — that is what makes reservoir computing a *design*
discipline, and why ResDAG treats structure as a first-class, pluggable
function.

The same slot also holds reservoirs with no randomness at all:
[`NGReservoir`](../build/layers.md) builds features from delayed inputs and
polynomial combinations — next-generation reservoir computing.

## 2 — The state is yours

Reservoir layers are stateful. The hidden state `(batch, reservoir_size)`
persists across `forward` calls; that persistence is what lets a warmup
pass hand a synchronized reservoir to a forecast loop. The full API:

```python
layer.reset_state()            # forget everything (lazy re-init)
layer.get_state()              # clone the live state
layer.set_state(saved)         # restore a checkpoint
layer.set_random_state(batch_size=4)
model.reset_reservoirs()       # all reservoirs in a model at once
```

Two contracts worth memorizing: the state silently re-initializes when the
incoming batch size, device, or dtype changes; and gradients never cross
`forward`-call boundaries (the stored state is detached — SGD over
consecutive batches just works).

## 3 — Training is a solve, not a search

The readout is a linear layer fitted by ridge regression on collected
states — a closed-form problem solved by conjugate gradient in
milliseconds, in one pass, with no learning rate. `ESNTrainer.fit` does
exactly three things: reset, teacher-forced warmup, and one forward pass
during which each readout fits itself at the moment it executes (so
multi-readout DAGs fit in dependency order for free).

Because readouts are ordinary `nn.Linear` layers underneath, the gradient
world stays open: freeze the reservoir and train a deep head with Adam, or
set `trainable=True` and backpropagate through everything. The
[Train](../workflows/train.md) page maps all three paths.

## 4 — Forecasting is feedback

A trained model maps *signal now* → *signal one step ahead*. Forecasting
closes the loop: after a teacher-forced warmup, each prediction is fed back
as the next input, `horizon` times. Exogenous drivers slot in alongside the
feedback with a pinned [timing convention](../theory/timing.md) — drivers
for the forecast window start exactly where the warmup drivers ended.

---

That is the whole model: design dynamics, manage state, solve the readout,
close the loop. The [Build](../build/index.md) track turns idea 1 into an
architecture handbook; [Work](../workflows/index.md) turns ideas 2–4 into
daily workflows.

## See also

- [Theory](../theory/index.md) — the equations behind all four ideas
- [Build](../build/index.md) — the composition handbook
