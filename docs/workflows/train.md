---
description: Three ways to train a reservoir model — the one-pass algebraic solve, frozen features with a gradient head, and full BPTT.
---

<span class="nb-kicker">Work · Train</span>

# Train

Three training paths coexist because ResDAG models are ordinary
`torch.nn.Module`s. The algebraic solve is the standard
reservoir-computing approach and the usual starting point; use
gradient-based training when the problem is not a least-squares fit.

## Path 1 — the algebraic solve

`ESNTrainer.fit` fits every readout in the model algebraically, all
inside a single forward pass; for a `CGReadoutLayer` each fit is one
ridge regression:

<div class="nb-specimen" data-label="train_algebraic.py" markdown>

```python
import torch
import resdag as rd
from resdag import ESNLayer, ESNModel, reservoir_input
from resdag.layers import CGReadoutLayer

data = torch.load("series.pt")        # (1, 7500, 3) — (batch, time, features)
warmup, train, target, f_warmup, val = rd.utils.prepare_esn_data(
    data, warmup_steps=300, train_steps=5000, val_steps=2000, normalize=True
)

inp = reservoir_input(3)
states = ESNLayer(500, feedback_size=3, spectral_radius=0.9)(inp)
out = CGReadoutLayer(500, 3, name="output", alpha=1e-6)(states)
model = ESNModel(inp, out)

rd.ESNTrainer(model).fit(
    warmup_inputs=(warmup,),          # tuple: (feedback, driver1, ...)
    train_inputs=(train,),            # same number of tensors as warmup
    targets={"output": target},       # keyed by readout name
)
```

</div>

**What `fit` does**, in order:

1. Validates — warmup and train tuples must hold the same number of
   tensors, targets must match train in sequence length, and every
   readout needs a key in `targets` (a missing key raises; an extra one
   warns).
2. Resets all reservoir states, then runs one teacher-forced pass over
   `warmup_inputs` to synchronize them.
3. Registers a pre-hook on every readout in the model.
4. Runs one forward pass over `train_inputs`. When each readout
   executes, its hook fits it on the tensor entering it (by conjugate
   gradient, for `CGReadoutLayer`). In multi-readout DAGs, downstream
   layers therefore receive outputs from already-fitted readouts, in
   dependency order.

**Multi-readout models** need one key per readout `name`:

```python
rd.ESNTrainer(model).fit(
    warmup_inputs=(warmup,),
    train_inputs=(train,),
    targets={"position": pos_target, "velocity": vel_target},
)
```

**What `alpha` does.** Each `CGReadoutLayer` solves
$\min_W \lVert XW - Y \rVert^2 + \alpha \lVert W \rVert^2$; `alpha` is
its only fitting hyperparameter and it lives on the layer, not the
trainer. Larger values shrink the weights, giving smoother, more stable
forecasts at the cost of one-step accuracy. It acts on a log scale:
sweep $10^{-8}$ to $10^{-2}$ in decade steps, not linearly.

### Preparing the data

`prepare_esn_data` cuts one series into the five tensors used above:

```text
[ discard ][ warmup ][———————————— train ————————————][ val ]
                      [—— target = train shifted +1 ——]
                                      [—— f_warmup ——]
```

The contract is *target equals train shifted forward one step*: each pair
`(train[:, t], target[:, t] == train[:, t+1])` teaches next-step
prediction, which is exactly what the forecast loop replays. `f_warmup`
is the tail of `train` (last `warmup_steps` samples), ready to
re-synchronize the reservoir right before forecasting over `val`. If you
slice by hand, keep the shift: an off-by-one here raises no error during
training but degrades forecasts.

---

## Path 2 — frozen reservoir, gradient head

When the head must be nonlinear or the loss is not least-squares, use the
reservoir as a fixed feature extractor and train any PyTorch head on top:

```python
import torch.nn as nn

inp = reservoir_input(3)
states = ESNLayer(500, feedback_size=3, spectral_radius=0.9)(inp)
extractor = ESNModel(inp, states)            # headless: (batch, time, 500)

head = nn.Sequential(nn.Linear(500, 64), nn.Tanh(), nn.Linear(64, 3))
opt = torch.optim.Adam(head.parameters(), lr=1e-3)

with torch.no_grad():                        # frozen features: compute once
    extractor.reset_reservoirs()
    feats = extractor(torch.cat([warmup, train], dim=1))[:, warmup.shape[1]:]

for step in range(300):
    loss = nn.functional.mse_loss(head(feats), target)
    opt.zero_grad()
    loss.backward()
    opt.step()
```

Precomputing `feats` once avoids re-running the reservoir on every
optimization step; this is the main efficiency advantage of a frozen
base. Streaming also works: push fresh batches through the reservoir
inside the loop and consecutive `backward()` calls succeed without
resets, because the stored state is detached at every forward-call
boundary (`detach_state_between_calls=True`) and no autograd graph
survives to trigger "backward through the graph a second time".

## Path 3 — full BPTT

`trainable=True` unfreezes the recurrent and input weights; gradients
then flow through the recurrence itself:

```python
reservoir = ESNLayer(200, feedback_size=3, spectral_radius=0.9, trainable=True)
head = nn.Linear(200, 3)
opt = torch.optim.Adam([*reservoir.parameters(), *head.parameters()], lr=1e-3)

for epoch in range(30):
    reservoir.reset_state()                      # fresh state each epoch
    pred = head(reservoir(train[:, :400]))       # truncate: BPTT cost grows with T
    loss = nn.functional.mse_loss(pred, target[:, :400])
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(reservoir.parameters(), 1.0)
    opt.step()
```

Clip gradients (backpropagation through hundreds of recurrent steps is
prone to exploding gradients) and keep the windows short. Use this path
only when a frozen random reservoir is demonstrably insufficient: it is
orders of magnitude slower than path 1 on the same task, and training
can move the recurrent matrix away from the spectral radius you
configured.

!!! note "Combining paths"
    The paths compose. Build the readout with `trainable=True`, run
    `ESNTrainer.fit` to obtain the ridge solution (the solve writes the
    weights directly, regardless of `trainable`), then fine-tune from
    there with a small learning rate. The algebraic solution is a better
    initialization than random weights.

## Next

- [**Forecast**](forecast.md) — autoregressive prediction with the trained model
- [Theory · Readout solvers](../theory/readout.md) — derivation of the ridge regression solve
