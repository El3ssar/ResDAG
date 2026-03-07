# Model Persistence

Save and load trained ESN models using `ESNModel`'s built-in save/load interface.

---

## Important: Architecture is Not Saved

resdag uses a **state-dict-only** save format. This means:

- `save()` writes the weights (and optionally states + metadata) to a `.pt` file
- `load()` reads weights back into an **already-constructed** model
- You must re-build the same model architecture in code before loading

This is intentional — it's the PyTorch native pattern and keeps the format simple and forward-compatible.

---

## Saving

```python
# Basic save — weights only
model.save("model.pt")

# Save with reservoir states
model.save("model_with_states.pt", include_states=True)

# Save with arbitrary metadata
model.save(
    "checkpoint.pt",
    include_states=True,
    epoch=50,
    train_loss=0.0023,
    val_mse=0.0045,
    spectral_radius=0.9,
)
```

The file is a standard PyTorch `.pt` file (pickle-based). The checkpoint dictionary has at least:
- `"state_dict"` — the model's `nn.Module` state dict
- `"reservoir_states"` — dict of state tensors (if `include_states=True`)
- Any extra keyword arguments are stored directly

---

## Loading

```python
# 1. Re-build the same architecture
from resdag.models import ott_esn

model = ott_esn(reservoir_size=500, feedback_size=3, output_size=3)

# 2. Load weights
model.load("model.pt")

# 3. Load with reservoir states (e.g., to continue from a checkpoint)
model.load("checkpoint.pt", load_states=True)
```

### Class Method: `load_from_file`

```python
pre_built = ott_esn(reservoir_size=500, feedback_size=3, output_size=3)
model = ESNModel.load_from_file("model.pt", model=pre_built)
```

---

## Checkpointing Workflows

### Save After Training

```python
trainer = ESNTrainer(model)
trainer.fit(
    warmup_inputs=(warmup,),
    train_inputs=(train,),
    targets={"output": targets},
)

model.save(
    "lorenz_ott_v1.pt",
    include_states=True,   # save the post-training reservoir state
    epoch=1,
    train_mse=train_mse,
)
```

### Resume from Checkpoint

```python
# Re-build exact same architecture
model = ott_esn(reservoir_size=500, feedback_size=3, output_size=3)
model.load("lorenz_ott_v1.pt", load_states=True)

# Model is now in trained state, ready to forecast
preds = model.forecast(warmup_data, horizon=1000)
```

### Read Checkpoint Metadata

```python
import torch

checkpoint = torch.load("lorenz_ott_v1.pt", weights_only=False)
print(checkpoint["epoch"])      # 1
print(checkpoint["train_mse"])  # 0.00012
```

---

## Saving Reservoir States Separately

If you want to save the reservoir state independently (e.g., to try multiple forecasts from the same warmup):

```python
# Warmup and save state
model.warmup(warmup_data)
states = model.get_reservoir_states()
torch.save(states, "reservoir_state.pt")

# Later: restore and forecast
states = torch.load("reservoir_state.pt")
model.set_reservoir_states(states)
preds = model.forecast(warmup_data, horizon=1000)
```

---

## GPU ↔ CPU Transfer

When loading a model saved on GPU onto a CPU machine (or vice versa):

```python
import torch

# Load to CPU regardless of where it was saved
checkpoint = torch.load("model.pt", map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint["state_dict"])

# Or move model after loading
model.load("model.pt")
model = model.to("cpu")
```

---

## Best Practices

1. **Always save after training** — readout weights are random until `fit()` is called
2. **Include states if using warmup-then-forecast workflows** — saves having to re-warmup
3. **Version your filenames** — `model_v1_sr09.pt` beats `model.pt`
4. **Store hyperparameters in the checkpoint** — makes it easy to reconstruct the architecture
5. **Use `weights_only=False`** for resdag checkpoints — they include metadata beyond just tensors

```python
# Full reproducibility pattern
hparams = {
    "reservoir_size": 500,
    "feedback_size": 3,
    "output_size": 3,
    "spectral_radius": 0.9,
    "topology": "watts_strogatz",
    "alpha": 1e-6,
}
model = ott_esn(**hparams)
# ... train ...
model.save("run_001.pt", include_states=True, **hparams)

# Reconstruct
checkpoint = torch.load("run_001.pt", weights_only=False)
hparams = {k: checkpoint[k] for k in ["reservoir_size", "feedback_size", "output_size",
                                        "spectral_radius", "topology", "alpha"]}
model = ott_esn(**hparams)
model.load("run_001.pt", load_states=True)
```
