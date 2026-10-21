---
description: API reference for resdag.training — the ESNTrainer that fits readouts algebraically in one pass.
---

<span class="nb-kicker">Reference</span>

# Training

The trainer that warms up reservoir states, then fits every readout in the
model with its configured readout solver in a single forward pass. Online RLS
and FORCE fitting are also available through `fit_online` and `fit_force`.

::: resdag.training
    options:
      members:
        - ESNTrainer
