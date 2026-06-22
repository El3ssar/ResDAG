---
description: The four ideas behind ResDAG — frozen dynamics, the state, the algebraic readout, and the two-phase forecast.
---

<span class="nb-kicker">Start · 03</span>

# The mental model

ResDAG rests on four ideas. Every component in the library implements one
of them.

## 1 — The dynamics are designed, not learned

A reservoir is a recurrent network whose weights are *chosen* and then
frozen: a connectivity structure (random, a graph, any
[matrix-building function](../build/initialization/index.md)), scaled to a target
spectral radius, driven by your signal. It is a fixed nonlinear dynamical
system that unfolds your input's history into hundreds of feature
trajectories. Because nothing inside is learned, every structural choice
is a hyperparameter; ResDAG therefore treats connectivity structure as a
pluggable function rather than a fixed implementation detail.

Reservoir families are an open set. The same interface also holds
reservoirs with no randomness at all:
[`NGReservoir`](../build/layers/ng-reservoir.md) builds features from delayed inputs and
polynomial combinations — next-generation reservoir computing.

## 2 — Reservoir layers are stateful

The hidden state persists across `forward` calls — each reservoir family
defines its own shape, such as `(batch, reservoir_size)` for an
`ESNLayer`. That persistence is what lets a warmup pass hand a
synchronized reservoir to a forecast loop. The state-management API:

```python
layer.reset_state()            # forget everything (lazy re-init)
layer.get_state()              # clone the live state
layer.set_state(saved)         # restore a checkpoint
layer.set_random_state(batch_size=4)
model.reset_reservoirs()       # all reservoirs in a model at once
```

Two contracts: the state silently re-initializes when the incoming batch
size, device, or dtype changes; and gradients never cross `forward`-call
boundaries, because the stored state is detached — gradient training over
consecutive batches does not accumulate graph history across calls.

## 3 — Readout training is a closed-form solve

Readouts are fitted algebraically on collected states rather than trained
by gradient descent. The current readout,
[`CGReadoutLayer`](../build/readouts/cg-readout.md), is a linear layer
fitted by ridge regression — a closed-form problem solved by conjugate
gradient in a single pass, with no learning rate or epoch schedule.
`ESNTrainer.fit` does exactly three things: reset, teacher-forced warmup,
and one forward pass during which each readout fits itself at the moment
it executes, so multi-readout DAGs fit in dependency order.

Because `CGReadoutLayer` is an ordinary `nn.Linear` underneath,
gradient-based training remains available: freeze the reservoir and train
a deep head with Adam, or set `trainable=True` and backpropagate through
everything. The [Train](../workflows/train.md) page covers all three
approaches.

!!! note "The pure-PyTorch quickstart"
    Prefer a normal optimizer loop? `ReservoirFeatureExtractor` wraps a
    reservoir as a plain `nn.Module` that drops into `nn.Sequential` ahead
    of any head, frozen by default — so you compute its features once under
    `torch.no_grad()` and train the head with `Adam` like any other model.
    The runnable snippet is
    [Train · Path 2 — frozen reservoir, gradient head](../workflows/train.md#path-2-frozen-reservoir-gradient-head),
    and [Scale & deploy](../workflows/deploy.md#inside-a-larger-pipeline)
    embeds the same frozen backbone inside a classifier `nn.Module`.

## 4 — Forecasting is autoregressive feedback

A trained model maps *signal now* → *signal one step ahead*. Forecasting
closes the loop: after a teacher-forced warmup, each prediction is fed back
as the next input, `horizon` times. Exogenous drivers slot in alongside the
feedback with a fixed [timing convention](../theory/timing.md): drivers
for the forecast window start exactly where the warmup drivers ended.

!!! note "For scikit-learn users"
    A reservoir forecaster has no single `fit`/`predict` because training
    and rollout are two phases — but they map cleanly onto the estimator
    workflow you already know:

    | scikit-learn | ResDAG |
    | --- | --- |
    | `est.fit(X, y)` | synchronize the state on a warmup window, then `ESNTrainer.fit(...)` solves the readout in one algebraic pass |
    | `est.predict(X)` | `model.forecast(warmup, horizon=...)` re-synchronizes, then rolls the prediction back in as the next input |

    The `ESN` facade collapses both phases into one object —
    `ESN(...).fit(series).forecast(horizon=...)`, numpy in and numpy out —
    for when you want the estimator shape without wiring the graph yourself:

    ```python
    import numpy as np
    from resdag import ESN

    series = np.cumsum(np.random.randn(2000, 3), axis=0)   # (time, features)
    prediction = ESN(reservoir_size=300).fit(series).forecast(horizon=200)
    ```

    `esn.model` exposes the full composable graph the moment you outgrow
    the facade.

---

The [Build](../build/index.md) section expands on idea 1 — layers,
topologies, initializers, and architectures; the
[Workflows](../workflows/index.md) section covers ideas 2–4 in practice —
training, forecasting, tuning, and deployment.

## See also

- [Theory](../theory/index.md) — the equations behind all four ideas
- [Build](../build/index.md) — composing models from components
