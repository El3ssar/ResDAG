---
description: Fit several readout heads on one reservoir in a single pass — separate targets per head, one shared dynamical state.
---

<span class="rd-eyebrow">Cookbook</span>

# Multi-readout models

One reservoir, several heads: a single set of reservoir states regressed
against as many target series as you like, all fitted in one forward pass.
Below, one ESN predicts a signal *and* its derivative from the same state.

<figure markdown>
![Multi-readout architecture](../assets/figures/arch_multi_readout.svg)
<figcaption>One reservoir feeding two independent readouts.</figcaption>
</figure>

## The whole thing

<div class="rd-window" data-title="multi_readout.py" markdown>

```python
import torch
import resdag as rd

# A signal and its derivative, both (batch, time, features)
t = torch.linspace(0, 60, 3000)
pos = torch.sin(t).reshape(1, -1, 1)
vel = torch.cos(t).reshape(1, -1, 1)

warmup, train, target_pos, f_warmup, val = rd.utils.data.prepare_esn_data(
    pos, warmup_steps=100, train_steps=2000, val_steps=300
)
# Velocity targets on the same rows as target_pos: one step ahead of train
target_vel = vel[:, 101:2101, :]

# One reservoir, two named heads
inp = rd.reservoir_input(1)
states = rd.ESNLayer(300, feedback_size=1, spectral_radius=0.9)(inp)
pos_head = rd.CGReadoutLayer(300, 1, name="position")(states)
vel_head = rd.CGReadoutLayer(300, 1, name="velocity")(states)
model = rd.ESNModel(inp, [pos_head, vel_head])

# One fit call, one key per head
rd.ESNTrainer(model).fit(
    warmup_inputs=(warmup,),
    train_inputs=(train,),
    targets={"position": target_pos, "velocity": target_vel},
)

# Multi-output forecast returns one tensor per head
pred_pos, pred_vel = model.forecast(f_warmup, horizon=300)
print(pred_pos.shape, pred_vel.shape)        # (1, 300, 1) (1, 300, 1)
print(torch.mean((pred_pos - val) ** 2))     # ~1e-6
```

</div>

The wiring is the `name=` argument: each `CGReadoutLayer` is fitted against
`targets[<its name>]`, nothing else. Each head solves its own ridge problem
on the same reservoir states — adding a head costs one extra solve, not a
second reservoir run.

## Forecast: the first output drives the loop

In `forecast()`, the **first** output passed to `ESNModel` becomes the
autoregressive feedback — here `pos_head`, whose dimension (1) matches
`feedback_size` (1). That match is required; the remaining heads are
computed along for the ride and returned in order. Put the head that
predicts the next feedback first, always.

!!! note "Readouts can feed downstream layers"
    A readout's output can be wired into further layers — another reservoir,
    a transform, even another readout. `ESNTrainer` fits each readout via a
    pre-hook that fires at the moment that readout would execute, so
    downstream layers always consume *already-fitted* outputs. Deep readout
    chains train correctly in the same single pass, no ordering work on
    your side.

## Related

- [Building models](../learn/building-models.md) — the symbolic API behind the wiring.
- [Readout fitting](../under-the-hood/readout-fitting.md) — the ridge solve each head runs.
- [Coupled ensembles](ensembles.md) — many models instead of many heads.
