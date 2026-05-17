# Data preparation & loading

## Load from disk

```python
from resdag.utils.data import load_file

data = load_file("series.csv")  # auto-detect csv / npy / npz / nc
# shape: (batch or 1, time, features) depending on file
```

## Split for ESN workflows

```python
from resdag.utils.data import prepare_esn_data

warmup, train, target, f_warmup, val = prepare_esn_data(
    data,
    warmup_steps=200,
    train_steps=1000,
    val_steps=300,
    discard_steps=500,      # burn-in on raw series
    normalize=True,
    norm_method="minmax",
)
```

Layout after optional `discard_steps`:

```text
[warmup][-------- train --------][-- val --]
        ^ target = train shifted +1
```

`f_warmup` = last `warmup_steps` of the train segment — use as `forecast()` warmup.

## One-shot helper

```python
from resdag.utils.data import load_and_prepare

splits = load_and_prepare(
    "lorenz.csv",
    warmup_steps=200,
    train_steps=1000,
    val_steps=300,
    normalize="minmax",
)
warmup, train, target, f_warmup, val = splits
```

## Normalization methods

| `norm_method` | Effect |
|---------------|--------|
| `minmax` | Scale to $[-1, 1]$ |
| `standard` | Zero mean, unit variance |
| `noncentered` | Divide by max absolute value |
| `meanpreserving` | Scale fluctuations, keep mean |

Stats are computed on the training portion and applied globally to all splits.

## Gotchas

- Ensure `warmup_steps + train_steps + val_steps` fits after `discard_steps`.
- For HPO, return the five tensors from `data_loader` with shapes `(B, T, D)`.

## See also

- [`prepare_esn_data`](../reference/utils/data.md)
- [Chaotic systems guide](chaotic-systems.md)
