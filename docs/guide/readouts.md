# Readout Layer

The readout is the **only trained component** in a standard ESN. resdag provides `CGReadoutLayer`, which uses Conjugate Gradient ridge regression for exact, efficient fitting.

---

## Design Philosophy

Traditional neural networks train all weights via backpropagation. Reservoir computing takes a fundamentally different approach:

- **Reservoir weights**: random, fixed, never trained
- **Readout weights**: fitted analytically via ridge regression

This means:
- Training is **exact** (no learning rate, no epochs, no convergence worries)
- Training is **fast** — a single linear solve
- No risk of overfitting the dynamics learning phase

---

## CGReadoutLayer

```python
from resdag.layers.readouts import CGReadoutLayer

readout = CGReadoutLayer(
    in_features=500,   # reservoir state dimension
    out_features=3,    # output dimension
    alpha=1e-6,        # L2 regularization (ridge penalty)
    bias=True,         # include bias column
    name="output",     # key used in targets dict during training
    max_iter=100,      # max Conjugate Gradient iterations
    tol=1e-5,          # CG convergence tolerance
)
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `in_features` | `int` | — | Input feature dimension (reservoir size or concat size) |
| `out_features` | `int` | — | Output dimension |
| `alpha` | `float` | `1e-6` | Ridge regularization strength. Larger = more regularization |
| `bias` | `bool` | `True` | Prepend ones column for affine fit |
| `name` | `str \| None` | `None` | Layer name used as key in `targets` dict |
| `max_iter` | `int` | `100` | Maximum CG iterations |
| `tol` | `float` | `1e-5` | CG convergence tolerance |

### The `name` Parameter

!!! important "Name must match targets key"
    The `name` you give `CGReadoutLayer` is the key you must use in `targets` when calling `ESNTrainer.fit()`.

    ```python
    readout = CGReadoutLayer(500, 3, name="output")   # name = "output"

    trainer.fit(
        ...,
        targets={"output": train_targets},             # must match!
    )
    ```

    If `name=None`, the layer gets an auto-generated name like `"CGReadoutLayer_1"`.

---

## Training

Readouts are **never trained manually** — `ESNTrainer` handles everything. But you can call the fitting methods directly if needed:

```python
# Direct fit (reservoir_states shape: (batch*time, in_features))
readout.fit(reservoir_states, targets)

# Check if fitted
print(readout.is_fitted)   # True after fitting

# Manual forward pass
output = readout(reservoir_states)
```

### Numerical Stability

Internally, `CGReadoutLayer` casts everything to `float64` before solving, then copies back to the original dtype. This ensures numerical stability for ill-conditioned systems even when operating in `float32`.

---

## Choosing `alpha`

The regularization parameter `alpha` controls the bias-variance tradeoff:

| `alpha` | Effect | Use when |
|---|---|---|
| `1e-8` – `1e-6` | Light regularization | Large clean datasets, expressive reservoirs |
| `1e-4` – `1e-2` | Moderate regularization | Noisy data, smaller reservoirs |
| `> 0.1` | Heavy regularization | Very small datasets |

!!! tip "Tune `alpha` with HPO"
    `alpha` is one of the most important hyperparameters. Use [HPO](../hpo/overview.md) to find
    the best value automatically:
    ```python
    alpha = trial.suggest_float("alpha", 1e-8, 1e-2, log=True)
    ```

---

## Multi-Output Readouts

A single readout can map to multiple output dimensions:

```python
# One readout for 3D output
readout = CGReadoutLayer(500, 3, name="output")

# Or separate readouts for different targets
readout_pos = CGReadoutLayer(500, 3, name="position")
readout_vel = CGReadoutLayer(500, 3, name="velocity")
```

With multiple readouts, train them all at once:

```python
trainer.fit(
    warmup_inputs=(warmup,),
    train_inputs=(train,),
    targets={
        "position": position_targets,
        "velocity": velocity_targets,
    },
)
```

---

## Readout Architecture Variants

### Standard: Reservoir → Readout

```python
inp      = ps.Input((100, 3))
res      = ESNLayer(500, feedback_size=3)(inp)
readout  = CGReadoutLayer(500, 3, name="output")(res)
model    = ESNModel(inp, readout)
```

### Augmented: Reservoir + Input → Readout (Ott-style)

```python
inp      = ps.Input((100, 3))
res      = ESNLayer(500, feedback_size=3)(inp)
aug      = SelectiveExponentiation(index=0, exponent=2.0)(res)
cat      = Concatenate()(inp, aug)                      # concat input + augmented states
readout  = CGReadoutLayer(3 + 500, 3, name="output")(cat)
model    = ESNModel(inp, readout)
```

### Multi-Scale: One Reservoir → Multiple Readouts

```python
inp      = ps.Input((100, 3))
res      = ESNLayer(1000, feedback_size=3)(inp)
r1step   = CGReadoutLayer(1000, 3, name="1step")(res)
r10step  = CGReadoutLayer(1000, 3, name="10step")(res)
model    = ESNModel(inp, [r1step, r10step])
```

---

## API Reference

::: resdag.layers.CGReadoutLayer
    options:
      show_root_heading: true
      show_source: false
