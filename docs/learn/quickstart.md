---
description: Install ResDAG, train an Echo State Network, and forecast a signal — sixty seconds of code, then a guided replay.
---

<span class="rd-eyebrow">Learn · 02</span>

# Quickstart: forecast in sixty seconds

By the end of this page you'll have trained an Echo State Network and
generated a 300-step forecast — and you'll know what every line did.

## Install

=== "uv"

    ```bash
    uv add resdag
    ```

=== "pip"

    ```bash
    pip install resdag
    ```

## The whole thing

<div class="rd-window" data-title="quickstart.py" markdown>

```python
import torch
import resdag as rd

# A signal to forecast: a sine wave, shaped (batch, time, features)
t = torch.linspace(0, 60, 3000)
data = torch.sin(t).reshape(1, -1, 1)

# Split it the ESN way: warmup / train / target / forecast-warmup / validation
warmup, train, target, f_warmup, val = rd.utils.prepare_esn_data(
    data, warmup_steps=100, train_steps=2000, val_steps=300
)

# A classic ESN: 300-neuron reservoir, ridge readout on [input ‖ states]
model = rd.models.classic_esn(reservoir_size=300, feedback_size=1, output_size=1)

# Train: one teacher-forced pass, one algebraic solve. No epochs.
rd.ESNTrainer(model).fit(
    warmup_inputs=(warmup,),
    train_inputs=(train,),
    targets={"output": target},
)

# Forecast: re-synchronize on f_warmup, then run free for 300 steps
prediction = model.forecast(f_warmup, horizon=300)

print(torch.mean((prediction - val) ** 2))   # ~1e-6
```

</div>

<figure markdown>
![Sine forecast](../assets/figures/predict_sine.png)
<figcaption>300 autoregressive steps. The model is feeding on its own
predictions the whole time — no ground truth after warmup.</figcaption>
</figure>

## The replay, line by line

**The split.** `prepare_esn_data` cuts one timeline into the five pieces
every ESN workflow needs:

```text
[ warmup │ train ───────────────── │ val ]
                       └ f_warmup ┘
```

`target` is `train` shifted one step forward — the model learns *given the
signal now, emit the signal one step ahead*. `f_warmup` is the tail of
`train`: the drive that re-synchronizes the reservoir right before the
held-out `val` window. Details in the
[data recipe](../cookbook/data.md).

**The model.** `classic_esn` is a factory wiring the smallest useful DAG:
a 300-unit [`ESNLayer`](../reference/layers/reservoirs.md), its states
concatenated with the raw input, feeding a
[`CGReadoutLayer`](../reference/layers/readouts.md) named `"output"`. The
reservoir's weights are random and frozen; only the readout will change.

**The fit.** `ESNTrainer.fit` resets reservoir state, runs the warmup to
wash out the zero initial state, then makes a single forward pass over
`train` during which the readout is fitted by conjugate-gradient ridge
regression against `target`. The key `"output"` in `targets` matches the
readout's `name` — that's the entire wiring between data and head.

**The forecast.** `model.forecast(f_warmup, horizon=300)` runs two phases:
teacher-forced warmup on `f_warmup`, then 300 steps where each output is
fed back as the next input. Returned shape: `(1, 300, 1)`, aligned exactly
with `val`.

!!! tip "The two knobs that matter first"
    If a forecast misbehaves, reach for `spectral_radius` (dynamics memory,
    try 0.8–1.2) and the readout's `alpha` (regularization, log-scale
    1e-8–1e-2) before anything else. Chapter
    [07 · Tuning](tuning.md) covers the full panel.

## Next

[**03 · Anatomy of an ESN**](anatomy.md) — what's actually inside the model
you just trained.
