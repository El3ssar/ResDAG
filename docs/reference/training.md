---
description: API reference for resdag.training — algebraic fitting with dependency-aware stages.
---

<span class="nb-kicker">Reference</span>

# Training

The trainer that warms up reservoir states, then fits every readout in the
model with its configured readout solver. Ordinary models use one training
pass. A readout feeding a downstream reservoir requires another stage so that
the downstream state is warmed up using the fitted upstream readout. Online RLS
and FORCE fitting are also available through `fit_online` and `fit_force`.

::: resdag.training
    options:
      members:
        - ESNTrainer
        - CalibrationReport
        - calibrate_intrinsic_plasticity
        - calibrate_flow_control
