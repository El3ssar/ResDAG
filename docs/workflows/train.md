---
description: Three ways to train a reservoir model — the one-pass algebraic solve, frozen features with a gradient head, and full BPTT.
---

<span class="nb-kicker">Work · Train</span>

# Train

Three training paths coexist because ResDAG models are ordinary
`torch.nn.Module`s. Start with the algebraic solve — it fits in
milliseconds and is the reservoir-computing classic; reach for gradients
only when the problem stops being least-squares.

## Path 1 — the algebraic solve

`ESNTrainer.fit` trains every readout in the model, each with one ridge
regression, all inside a single forward pass:

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
4. Runs one forward pass over `train_inputs`. The moment each readout
   executes, its hook fits it by conjugate gradient on the exact tensor
   entering it — so in multi-readout DAGs, downstream layers receive
   outputs from already-fitted readouts, in dependency order, for free.

**Multi-readout models** just need more keys, one per readout `name`:

```python
rd.ESNTrainer(model).fit(
    warmup_inputs=(warmup,),
    train_inputs=(train,),
    targets={"position": pos_target, "velocity": vel_target},
)
```

**What `alpha` does.** Each readout solves
$\min_W \lVert XW - Y \rVert^2 + \alpha \lVert W \rVert^2$; `alpha` is
the only fitting hyperparameter and it lives on the layer, not the
trainer. Larger values shrink the weights — smoother, more stable
forecasts at the cost of one-step accuracy. It works on a log scale:
sweep $10^{-8}$ to $10^{-2}$ in decade steps, never linearly.

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
slice by hand, keep the shift — an off-by-one here trains without
complaint and quietly ruins forecasts.

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

Precomputing `feats` once is the big speed win of a frozen base. But
streaming works too: push fresh batches through the reservoir inside the
loop and consecutive `backward()` calls just work, without resets — the
stored state is detached at every forward-call boundary
(`detach_state_between_calls=True`), so no autograd graph survives to
trigger "backward through the graph a second time".

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

Clip gradients — backprop through hundreds of recurrent steps is exactly
where they explode — and keep the windows short. This path is worth it
only when a frozen random reservoir demonstrably is not good enough: it
runs orders of magnitude slower than path 1 on the same task, and
training can drag the recurrent matrix away from the spectral radius you
designed.

!!! note "Combining paths"
    The paths compose. Build the readout with `trainable=True`, run
    `ESNTrainer.fit` to land on the ridge solution (the solve writes the
    weights directly, regardless of `trainable`), then fine-tune from
    there with a small learning rate. The algebraic answer is a far
    better initialization than random.

## Next

- [**Forecast**](forecast.md) — close the loop on what you just trained
- [Theory · Readout solvers](../theory/readout.md) — the ridge solve, derived
