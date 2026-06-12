---
description: Three ways to train a ResDAG model — one-pass algebraic fitting, SGD on a frozen-reservoir head, and full BPTT — and when to combine them.
---

<span class="rd-eyebrow">Learn · 05</span>

# Training

Three paths share one `ESNModel`; what changes is which parameters move and
what moves them. By the end of this page you'll know the mechanics of the
algebraic default, how to bolt an SGD-trained head onto a frozen reservoir,
and when full backprop through the reservoir earns its cost.

| Path | Reservoir | Readout | Optimizer |
|---|---|---|---|
| Algebraic (default) | frozen | `CGReadoutLayer`, fitted in one solve | `ESNTrainer.fit` |
| Frozen features + SGD head | frozen | any `nn.Module` | `torch.optim` |
| Full BPTT | `trainable=True` | `trainable=True` | `torch.optim` |

## Path 1 — algebraic fitting

<div class="rd-window" data-title="train.py" markdown>

```python
import resdag as rd

trainer = rd.ESNTrainer(model)
trainer.fit(
    warmup_inputs=(warmup,),          # (batch, warmup_steps, features)
    train_inputs=(train,),            # (batch, train_steps, features)
    targets={"output": target},       # keyed by each readout's name
)
```

</div>

What `fit()` actually does, in order:

1. **Validate** — every readout in the model must have a matching key in
   `targets` (a missing one raises, an extra one warns).
2. **Warmup** — `model.warmup(*warmup_inputs)`: reservoirs reset to zero,
   then a teacher-forced pass washes out the arbitrary initial state.
3. **Hook** — a forward *pre-hook* is registered on each readout. When the
   readout's forward fires, the hook first calls
   `readout.fit(arriving_states, target)` — one conjugate-gradient ridge
   solve, weights copied in place.
4. **One pass** — a single `model(*train_inputs)` under `torch.no_grad()`.
   Hooks fire as the graph executes, then are removed.

Because hooks fire at execution time, topological order comes for free: a
readout deeper in the DAG is fitted only after upstream readouts already
emit *fitted* outputs. That's the whole trainer — no epochs, no learning
rate, no convergence to babysit.

Multi-readout models train in the same single pass — one target per name:

```python
trainer.fit(
    warmup_inputs=(warmup,),
    train_inputs=(train,),
    targets={"position": pos_target, "energy": energy_target},
)
```

For input-driven models the tuples grow in parallel:
`warmup_inputs=(warmup_fb, warmup_drv)`, `train_inputs=(train_fb,
train_drv)` — feedback first, always.

**The one hyperparameter: `alpha`.** Each readout solves
$\min_W \lVert XW - Y\rVert^2 + \alpha \lVert W\rVert^2$, with `alpha` set
at construction (`CGReadoutLayer(..., alpha=1e-6)`). Small `alpha` fits the
training data harder; larger values trade training accuracy for forecasts
that stay stable over long horizons. Sweep it on a log scale before
touching anything else. Solver details in
[readout fitting](../under-the-hood/readout-fitting.md); why the targets
are shifted one step lives in
[timing and alignment](../under-the-hood/timing-and-alignment.md).

## Path 2 — frozen reservoir, SGD head

Need a nonlinear head, a classification loss, or a probabilistic decoder?
Use the reservoir as a feature extractor and train any head with a standard
loop:

```python
import torch
import torch.nn as nn
import resdag as rd

features = rd.headless_esn(reservoir_size=500, feedback_size=3)
head = nn.Sequential(nn.Linear(500, 64), nn.Tanh(), nn.Linear(64, 3))

optim = torch.optim.Adam(head.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for x, y in loader:                  # x: (batch, time, 3), y: (batch, time, 3)
    features.reset_reservoirs()
    states = features(x)             # (batch, time, 500), no grads into reservoir
    loss = loss_fn(head(states), y)
    optim.zero_grad()
    loss.backward()
    optim.step()
```

This works out of the box for two reasons: the reservoir's parameters have
`requires_grad=False`, so no gradients flow into them; and the stored state
is detached between forward calls, so consecutive batches never share an
autograd graph — no "backward through the graph a second time" surprises.

## Path 3 — full BPTT

Set `trainable=True` on the reservoir (and the readout) and PyTorch tracks
gradients through every timestep of the recurrent update:

```python
inp = rd.reservoir_input(3)
states = rd.ESNLayer(200, feedback_size=3, spectral_radius=0.9,
                     trainable=True)(inp)
out = rd.CGReadoutLayer(200, 3, name="output", trainable=True)(states)
model = rd.ESNModel(inp, out)

optim = torch.optim.Adam(model.parameters(), lr=5e-3)
for epoch in range(n_epochs):
    model.reset_reservoirs()
    loss = loss_fn(model(x), y)
    optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optim.step()
```

!!! warning "BPTT caveats"
    Memory grows linearly with sequence length, gradient clipping is
    effectively mandatory, and training moves the spectral radius — the
    echo state property is no longer guaranteed, so long-horizon stability
    can degrade even as training loss falls.

Worth it only when the random reservoir genuinely lacks the features your
task needs and a bigger reservoir or better [tuning](tuning.md) didn't fix
it. Try paths 1 and 2 first; they're orders of magnitude cheaper.

**The paths combine.** Build the readout with `trainable=True`, call
`trainer.fit()` for an algebraic warm start (the solver runs under
`no_grad` and writes weights regardless of the flag), then fine-tune with
SGD from a solution that's already good.

A quick sanity check on which path you've actually configured:

```python
print([n for n, p in model.named_parameters() if p.requires_grad])
# path 1: []   path 2: head params only   path 3: everything
```

## Next

[**06 · Forecasting**](forecasting.md) — warmup, autoregression, drivers,
and how far ahead you can honestly predict.
