---
description: Adapt reservoir activity using training inputs before fitting a readout, with intrinsic plasticity or flow control.
---

# Calibrate a reservoir

Calibration changes the reservoir using **training inputs only**, before the
readout is fitted. It needs no targets. It is optional: start with a normal
ESN, then compare calibration on held-out data if its neurons saturate or its
activity is poorly scaled.

```python
import torch
from resdag import ESNLayer
from resdag.training import calibrate_intrinsic_plasticity

training = torch.randn(1, 800, 2)
reservoir = ESNLayer(64, feedback_size=2, spectral_radius=0.8, seed=5)
report = calibrate_intrinsic_plasticity(
    reservoir, training,
    epochs=2, warmup=50, learning_rate=1e-3,
    target_mean=0.0, target_std=0.2,
)
```

Inputs may be `(time, features)` or `(batch, time, features)`. Pass
`(feedback, driver)` for a driven reservoir. Every batch slot represents an
independent trajectory; time order is preserved. Calibration replays the
training sequence for each epoch and resets state between replays. A successful
calibration resets the layer's carried state, ready for a fresh readout warmup.
It restores the layer's training/evaluation mode. Invalid inputs fail before
committing adapted values.

## Intrinsic plasticity

Gaussian-target intrinsic plasticity adapts each neuron's gain and offset in
`tanh(a * drive + b)`. `target_mean` and `target_std` describe the desired
activity distribution. The learning rate controls the adaptation speed; an
aggressive value can make the procedure unstable. The result is not guaranteed
to become exactly Gaussian or to improve forecast accuracy.

This implementation supports the standard tanh ESN and its documented leak
behavior. It records adaptations in registered buffers, so `.to()`,
`state_dict()` and model checkpoints carry them with the reservoir. It leaves
all ordinary weights frozen unless you explicitly enable training.

## Flow control

Flow control adapts per-neuron gains on the **recurrent contribution**. It
balances recurrent input power against the previous activity's power:

```python
from resdag.training import calibrate_flow_control

reservoir = ESNLayer(64, feedback_size=2, leak_rate=1.0, seed=5)
report = calibrate_flow_control(
    reservoir, training,
    target_radius=0.9, learning_rate=1e-3,
    warmup=50, epochs=2, mode="local",
)
```

`mode="local"` uses each neuron's own activity; `"global"` averages the
control signal across neurons. This is a data-driven adaptation rule, not an
exact eigenvalue rescale. Strong correlations in the inputs can move the
achieved radius away from its target. The report includes the measured
before/after effective radius and activity statistics for inspection. For example, inspect
`report.activity_std_before`, `report.activity_std_after`, and
`report.spectral_radius_after`.

Flow control requires unit leak and does not combine with an existing
intrinsic-plasticity adaptation. Unsupported cell types and parameter combinations are rejected explicitly;
frozen dense and frozen sparse ESNs use the same adaptation math; consult the
[function reference](../reference/training.md) for the exact supported contract.

## Fit after calibration

Calibrate the layer, then use that same instance in the normal model graph:

```python
from resdag import ESNModel, ESNTrainer, SVDReadoutLayer, reservoir_input

inp = reservoir_input(2)
model = ESNModel(
    inp, SVDReadoutLayer(64, 2, name="output")(reservoir(inp)),
)
ESNTrainer(model).fit(training[:, :50], training[:, 50:-1], training[:, 51:])
model.eval()
```

After calibration, refit any existing readout: its old weights were fitted to
the previous reservoir features. Never calibrate on validation/test samples,
and fit normalization on the training period too. Use chronological validation
to decide whether the extra calibration helps your task.

The update rules follow the original
[Gaussian-tanh adaptation paper](https://backoffice.biblio.ugent.be/download/416663/447989)
and [local flow-control paper](https://arxiv.org/abs/2101.10665).
