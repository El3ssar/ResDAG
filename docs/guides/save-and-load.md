# Save, load, and checkpoint

`ESNModel.save` / `load` store **weights only** — rebuild the same graph before loading.

## Weights

```python
from resdag import classic_esn

model = classic_esn(200, feedback_size=1, output_size=1)
# ... train with ESNTrainer ...

model.save("weights.pt")

rebuilt = classic_esn(200, feedback_size=1, output_size=1)
rebuilt.load("weights.pt")
```

## Checkpoint with reservoir states

```python
model.save(
    "checkpoint.pt",
    include_states=True,
    epoch=10,
    val_mse=0.042,
)
rebuilt.load("checkpoint.pt", load_states=True)
```

## Class method

```python
from resdag.core import ESNModel

model = ESNModel.load_from_file("weights.pt", model=rebuilt)
```

## Gotchas

- Architecture (layer sizes, topology names) is **not** in the file — version-control your
  build script.
- Readout must be fitted before save if you rely on `W_out` at inference.
- `headless_esn` saves reservoir weights only (no readout).

## See also

- [`ESNModel` persistence](../reference/core.md)
- [Example 07](https://github.com/El3ssar/resdag/blob/main/examples/07_save_load_models.py)
