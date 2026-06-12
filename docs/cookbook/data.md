---
description: Load time series from file and split them into the five tensors every ESN workflow needs — with the alignment rules that keep forecasts honest.
---

<span class="rd-eyebrow">Cookbook</span>

# Data preparation

By the end of this page you'll have a file on disk turned into five
correctly-aligned tensors — warmup, train, target, forecast-warmup,
validation — and you'll know exactly which timestep each one starts on.

## Load

`load_file` detects the format from the extension and always returns
`(batch, time, features)` — 1D input becomes `(1, T, 1)`, 2D becomes
`(1, T, D)`:

```python
from resdag.utils.data import load_file

data = load_file("lorenz.csv")                 # headerless numeric CSV
data = load_file("lorenz.npy")                 # NumPy array
data = load_file("runs.npz", key="data")       # .npz archive, pick the array
data = load_file("field.nc")                   # NetCDF (requires xarray)
```

## Split

<div class="rd-window" data-title="prepare.py" markdown>

```python
from resdag.utils.data import prepare_esn_data

warmup, train, target, f_warmup, val = prepare_esn_data(
    data,                  # (B, T, D)
    warmup_steps=2_000,
    train_steps=15_000,
    val_steps=5_000,
    discard_steps=1_000,   # drop initial transients before splitting
    normalize=True,
    norm_method="minmax",  # "minmax" | "standard" | "noncentered" | "meanpreserving"
)
```

</div>

The split, on the timeline after `discard_steps` are dropped:

```text
[ discard │ warmup │ train ─────────────────────── │ val ]
                      → target = train shifted +1 →
                                  └─── f_warmup ───┘
```

What each tensor is, exactly:

| Tensor | Shape | Meaning |
|---|---|---|
| `warmup` | `(B, warmup_steps, D)` | Teacher-forcing drive that washes out the zero initial state before training. |
| `train` | `(B, train_steps, D)` | Inputs for the readout fit. |
| `target` | `(B, train_steps, D)` | `train` shifted one step forward — the one-step-ahead contract. |
| `f_warmup` | `(B, warmup_steps, D)` | The **last** `warmup_steps` of `train` — the drive that re-synchronizes the reservoir immediately before `val`. |
| `val` | `(B, val_steps, D)` | Held out; compare against `forecast(f_warmup, horizon=val.shape[1])`. |

`f_warmup` is not a separate region of the file — it's the suffix of
`train`, chosen so that the reservoir state at the start of the forecast is
the state it had at the end of the training window, which is exactly where
`val` begins.

!!! warning "Common mistake: f_warmup from anywhere else"
    Warming up on any other window — the original `warmup`, a random slice,
    yesterday's data — leaves the reservoir synchronized to the *wrong
    point in time*, and `forecast()` output no longer lines up with `val`.
    The forecast may even look plausible while being shifted or detuned.
    Always warm up on the data immediately preceding the window you predict.

Two knobs worth a sentence each:

- **`discard_steps`** trims the head of the series before anything is
  split — use it when trajectories start off-attractor (numerical
  transients, burn-in of a simulation).
- **`normalize=True`** computes statistics from `train` only and applies
  them to all five splits, so no information leaks from `val` into the fit.
  `"minmax"` maps to [-1, 1]; `"standard"` is z-scoring; `"noncentered"`
  divides by the max absolute value (preserves zero); `"meanpreserving"`
  scales deviations but restores the mean.

## One-shot

`load_and_prepare` fuses both steps; multiple paths concatenate along the
batch dimension:

```python
from resdag.utils.data import load_and_prepare

warmup, train, target, f_warmup, val = load_and_prepare(
    ["run_a.npy", "run_b.npy"],     # -> batch of 2 trajectories
    warmup_steps=2_000, train_steps=15_000, val_steps=5_000,
    normalize=True,
)
```

## Related

- [Quickstart](../learn/quickstart.md) — the split in action on a sine wave.
- [Timing & alignment](../under-the-hood/timing-and-alignment.md) — the full index-by-index contract.
- [Coupled ensembles](ensembles.md) — the same five tensors feeding N models at once.
