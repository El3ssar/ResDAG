# Your first ESN

Sine wave → `prepare_esn_data` → `classic_esn` → train → forecast on the
validation segment.

The input is a 6 000-step sine with 60 cycles. We split it into warm-up,
training, and validation regions:

<figure markdown>
  ![Sine wave used in the example](../assets/figures/signal_sine.png){ width="700" }
  <figcaption>One short window of the sine signal — the full series the
  script uses is 6 000 steps.</figcaption>
</figure>

```python
import torch
from resdag import classic_esn
from resdag.training import ESNTrainer
from resdag.utils.data import prepare_esn_data

torch.manual_seed(0)

# 6 000-step sine, 60 full cycles. Already in [-1, 1] so no normalization needed.
t = torch.linspace(0, 60 * 2 * torch.pi, 6_000)
data = torch.sin(t).view(1, -1, 1)

warmup, train, target, f_warmup, val = prepare_esn_data(
    data,
    warmup_steps=500,
    train_steps=4_500,
    val_steps=800,
    normalize=False,
)

model = classic_esn(
    reservoir_size=300,
    feedback_size=1,
    output_size=1,
    spectral_radius=0.99,   # near-edge memory
    leak_rate=0.3,          # smooth dynamics match the sine smoothness
    readout_alpha=1e-8,     # clean data → low regularization
)

ESNTrainer(model).fit(
    warmup_inputs=(warmup,),
    train_inputs=(train,),
    targets={"output": target},
)

model.reset_reservoirs()
pred = model.forecast(f_warmup, horizon=val.shape[1])
print("val MSE:", float(((pred - val) ** 2).mean()))   # ≈ 6e-11
```

`f_warmup` is the last `warmup_steps` rows of `train` (see
[`prepare_esn_data`](../reference/utils/data.md)). `val` is the unseen tail
used only for evaluation.

## What the reservoir sees

Before training, you can look at the raw reservoir trajectory — the
internal state of a few neurons as the sine drives them:

<figure markdown>
  ![Reservoir activations for a sine input](../assets/figures/activations_sine.png){ width="700" }
  <figcaption>20 reservoir neurons (out of 200) reacting to the input
  signal at the top. Each line is a different lagged, filtered version of
  the input — what the readout regresses against.</figcaption>
</figure>

## How the forecast looks

After training, `model.forecast` runs autoregressively (no teacher
forcing) for the full validation horizon:

<figure markdown>
  ![Sine prediction overlay](../assets/figures/predict_sine.png){ width="720" }
  <figcaption>800-step autoregressive forecast. Truth and prediction are
  visually indistinguishable; the MSE is ~6×10⁻¹¹.</figcaption>
</figure>

## Why these hyperparameters

| Knob | Value | Why |
|------|-------|-----|
| `reservoir_size=300` | medium | More than enough for a single-frequency oscillator. |
| `spectral_radius=0.99` | near 1 | Long memory — the reservoir must remember a full period (~100 steps) to predict the next sample. |
| `leak_rate=0.3` | slow | The dynamics of a sine are smooth; a low leak rate matches that timescale. |
| `readout_alpha=1e-8` | tiny | The data is clean; we want the readout to fit it precisely. |
| `normalize=False` | — | The sine is already in [−1, 1]; min-max would be a no-op. |

Cranking `spectral_radius` lower (e.g. `0.5`) breaks long-term memory and
the forecast drifts. Cranking `readout_alpha` higher (e.g. `1e-4`) blurs
the amplitude. Both are worth trying — the failure modes are instructive.

## Next

[Lorenz walkthrough](lorenz-walkthrough.md) — same workflow on a
non-periodic, chaotic 3-D system.
