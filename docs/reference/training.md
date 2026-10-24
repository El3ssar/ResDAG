---
description: API reference for resdag.training — algebraic fitting with dependency-aware stages.
---

<span class="nb-kicker">Reference</span>

# Training

`ESNTrainer` fits raw `ESNModel` graphs with each readout's configured solver.
Its `fit` method processes the complete warmup-plus-training prefix in dependency
order: a readout is fitted before its output reaches a downstream reservoir.
Each reservoir consumes the continuous prefix once. Use `Model.fit` or
`ESN.fit` when you also need training-only scaling, named signals and original
units; see the [common workflow](../start/model-workflow.md).

For long streams, `fit_stream` accepts chunks and incremental ridge readouts,
then finalizes their accumulated statistics. `fit_force` trains a single RLS
head inside the autonomous feedback loop. Direct online fitting is available
on [`RLSReadout.fit_online`][resdag.layers.readouts.RLSReadout.fit_online],
not as another trainer method. See [streaming](../workflows/streaming.md) and
[training](../workflows/train.md) for these distinct lifecycles.

Calibration functions change reservoir operating points using observed
training inputs and return a `CalibrationReport`. They are explicit steps,
separate from readout fitting and default model construction.

::: resdag.training
    options:
      members:
        - ESNTrainer
        - CalibrationReport
        - calibrate_intrinsic_plasticity
        - calibrate_flow_control
