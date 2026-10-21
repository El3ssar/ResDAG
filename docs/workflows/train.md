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
[ discard ][ warmup ][———————————— train ————————————][seam][ val ]
                      [—— target = train shifted +1 ——]
                                      [—— f_warmup ——]
```

Each training input is paired with the next sample in the original series:
``target[:, t] = data[:, discard_steps + warmup_steps + t + 1]``.
The final target is the seam sample, immediately after the training inputs.
With ``train_end = warmup_steps + train_steps`` measured after discarding,
validation starts at ``data[:, train_end + 1]``. The seam is therefore used for
training and excluded from held-out scoring.

`f_warmup` contains the last `warmup_steps` samples before `train_end`; when
training is shorter than warmup, it includes part of the earlier warmup history.
Its last output predicts the seam, which seeds the next autoregressive step.
The required source length is `discard_steps + warmup_steps + train_steps + 1 + val_steps`.
With `normalize=True`, statistics are fitted on training inputs only and applied
to every split; `return_stats=True` also returns those statistics for inverting
predictions with `rd.data.denormalize_data`.

---

## Path 2 — frozen reservoir, gradient head

When the head must be nonlinear or the loss is not least-squares, use the
reservoir as a fixed feature extractor and train any PyTorch head on top.
`ReservoirFeatureExtractor` packages a reservoir as a plain
`nn.Module` that drops straight into `nn.Sequential` ahead of the head —
single positional input, frozen by default, one optimizer over
`model.parameters()`:

<div class="nb-specimen" data-label="train_frozen_head.py" markdown>

```python
import torch
import torch.nn as nn
from resdag import ReservoirFeatureExtractor

model = nn.Sequential(
    ReservoirFeatureExtractor(500, feedback_size=3, spectral_radius=0.9),
    nn.Linear(500, 64),
    nn.Tanh(),
    nn.Linear(64, 3),
)
extractor, head = model[0], model[1:]
opt = torch.optim.Adam(head.parameters(), lr=1e-3)  # head only — reservoir is frozen

with torch.no_grad():                        # frozen features: compute once
    extractor.on_epoch_start()               # epoch-reset hook (alias of reset_state)
    feats = extractor(torch.cat([warmup, train], dim=1))[:, warmup.shape[1]:]

for step in range(300):
    loss = nn.functional.mse_loss(head(feats), target)
    opt.zero_grad()
    loss.backward()
    opt.step()
```

</div>

The reservoir is **frozen by default** (`extractor.is_frozen` is `True`), so
the optimizer only ever sees the head's parameters. Precomputing `feats`
once avoids re-running the reservoir on every optimization step; this is the
main efficiency advantage of a frozen base. Streaming also works: push fresh
batches through the reservoir inside the loop and consecutive `backward()`
calls succeed without resets, because the stored state is detached at every
forward-call boundary (`detach_state_between_calls=True`) and no autograd
graph survives to trigger "backward through the graph a second time".

**The epoch hook.** The reservoir is stateful, so re-zero it between epochs
with `extractor.on_epoch_start()` (a readable alias of `reset_state()`) at
the top of each epoch — otherwise a trajectory from the previous epoch
bleeds into the next one:

```python
for epoch in range(num_epochs):
    extractor.on_epoch_start()               # re-zero the reservoir state
    for batch_feedback, batch_target in loader:
        loss = nn.functional.mse_loss(head(extractor(batch_feedback)), batch_target)
        opt.zero_grad(); loss.backward(); opt.step()
```

**A non-regression head.** Because the extractor is just an `nn.Module`, any
torch head works. Feeding the last-timestep feature vector into a linear
classifier and a cross-entropy loss turns the same frozen reservoir into a
sequence classifier:

```python
extractor = ReservoirFeatureExtractor(500, feedback_size=3, spectral_radius=0.9)
classifier = nn.Linear(500, num_classes)

with torch.no_grad():
    extractor.reset_state()
    summary = extractor(sequences)[:, -1, :]   # (batch, 500) — last-step features

opt = torch.optim.Adam(classifier.parameters(), lr=1e-2)
for step in range(150):
    loss = nn.functional.cross_entropy(classifier(summary), labels)
    opt.zero_grad(); loss.backward(); opt.step()
```

**Reusing an existing model.** `ReservoirFeatureExtractor.from_model(esn)`
wraps the reservoir layers of an already-built `ESNModel` *by reference* —
the parameters are shared, not copied — so an algebraically-fitted model can
double as a feature source without rebuilding the reservoir:

```python
from resdag import ESNModel, ESNLayer, reservoir_input

inp = reservoir_input(3)
states = ESNLayer(500, feedback_size=3, spectral_radius=0.9)(inp)
esn = ESNModel(inp, states)
extractor = ReservoirFeatureExtractor.from_model(esn)  # shares esn's reservoir
```

To backpropagate through the recurrence as well (full BPTT, Path 3), pass
`trainable=True` or call `extractor.unfreeze()`. The runnable end-to-end
recipe — regression head, classification head, and `from_model` reuse — is
[`examples/12_feature_extractor.py`](https://github.com/El3ssar/ResDAG/blob/main/examples/12_feature_extractor.py).

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

## Path 4 — differentiable forecast (BPTT through the rollout)

Paths 1–3 train on a *teacher-forced* pass — every step sees ground truth.
But the quantity you usually care about is *free-running* forecast quality,
where each prediction is fed back as the next input and errors compound. To
optimize that directly, roll the forecast out **differentiably** and
backpropagate the multi-step loss through the autoregressive loop.

`forecast(no_grad=False)` does exactly this: instead of running under
`torch.no_grad()` (the default), it builds an autograd graph through the warmup
*and* every autoregressive step, so `loss.backward()` reaches the trainable
reservoir and readout parameters:

```python
inp = reservoir_input(3)
states = ESNLayer(200, feedback_size=3, spectral_radius=0.9, trainable=True)(inp)
model = ESNModel(inp, nn.Linear(200, 3)(states))     # any differentiable head
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(50):
    preds = model.forecast(f_warmup, horizon=60, no_grad=False)  # differentiable rollout
    loss = nn.functional.mse_loss(preds, val[:, :60])            # multi-step forecast loss
    opt.zero_grad()
    loss.backward()                                              # BPTT through the loop
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
```

The returned tensor carries `requires_grad=True`, and gradients flow back
through every fed-back step — this is BPTT through the *closed-loop* forecast,
not just a teacher-forced window. Keep the horizon short (backprop through a
long autoregressive chain is memory-heavy and prone to exploding gradients —
clip), and use this only when free-running skill genuinely matters more than
the one-step fit the algebraic solve optimizes.

Two constraints:

- `no_grad=False` is **mutually exclusive with `compile=True`** — the compiled
  flat engine is inference-only, so requesting both raises `ValueError`.
- Something in the loop must be trainable (`trainable=True` on the reservoir,
  and/or a gradient head); with everything frozen the backward pass has nothing
  to update.

## Next

- [**Forecast**](forecast.md) — autoregressive prediction with the trained model
- [Theory · Readout solvers](../theory/readout.md) — derivation of the ridge regression solve
