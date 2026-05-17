# Data preparation and loading

Use `resdag.utils.data` for file I/O and the standard train/forecast split.

## Load

```python
from resdag.utils.data import load_file

data = load_file("series.csv")  # shape (1, T, D) or (T, D) depending on file
if data.dim() == 2:
    data = data.unsqueeze(0)
```

## Split with `prepare_esn_data`

```python
from resdag.utils.data import prepare_esn_data

warmup, train, target, f_warmup, val = prepare_esn_data(
    data,
    warmup_steps=2_000,
    train_steps=20_000,
    val_steps=5_000,
    discard_steps=1_000,
    normalize=True,
    norm_method="minmax",
)
```

Layout on the trimmed series:

```text
| discard | warmup | train | val |
```

| Tensor | Shape | Role |
|--------|-------|------|
| `warmup` | `(B, warmup_steps, D)` | Teacher-forced state sync before training |
| `train` | `(B, train_steps, D)` | Training inputs for readout fit |
| `target` | `(B, train_steps, D)` | One-step-ahead targets (`train` shifted by +1) |
| **`f_warmup`** | `(B, warmup_steps, D)` | **`train[:, -warmup_steps:, :]`** — last warmup window of train; use as `forecast()` input drive immediately before `val` |
| `val` | `(B, val_steps, D)` | Held-out segment; compare `forecast(f_warmup, horizon=val.shape[1])` to this |

`f_warmup` is **not** a separate region of the raw file. It is the suffix of `train`
adjacent to `val`, so the reservoir state at the start of the forecast matches the
end of the training drive.

## One-shot

```python
from resdag.utils.data import load_and_prepare

warmup, train, target, f_warmup, val = load_and_prepare(
    "lorenz.csv",
    warmup_steps=2_000,
    train_steps=20_000,
    val_steps=5_000,
    normalize=True,
)
```

## HPO `data_loader`

Return the same five keys:

```python
def data_loader(trial):
    data = ...
    warmup, train, target, f_warmup, val = prepare_esn_data(...)
    return {
        "warmup": warmup,
        "train": train,
        "target": target,
        "f_warmup": f_warmup,
        "val": val,
    }
```

## See also

[`prepare_esn_data`](../reference/utils/data.md)
