---
description: API reference for resdag.training — the ESNTrainer that fits readouts algebraically in one pass.
---

<span class="nb-kicker">Reference</span>

# Training

The trainer that warms up reservoir states, then fits every readout in the
model algebraically — ridge regression, not gradient descent — in a single
forward pass.

::: resdag.training
    options:
      members:
        - ESNTrainer
