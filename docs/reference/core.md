---
description: API reference for resdag.core — Model, ESNModel, ReservoirFeatureExtractor, and symbolic input helpers.
---

<span class="nb-kicker">Reference</span>

# Core

`Model` adds named inputs/outputs, fitted scaling and fit/forecast operations
around a functional graph. Start with the
[common model guide](../start/model-workflow.md). `ESNModel` remains the
underlying symbolic graph with lower-level training and state interfaces.
For timestamped runs, see [TSDynamics trajectories](../start/trajectory-workflow.md).
Several ports use [named trajectories](../start/named-trajectories.md);
per-port unit conversions use [custom preprocessing](../advanced/custom-preprocessing.md).

::: resdag.core
    options:
      members:
        - Model
        - ESNModel
        - ReservoirFeatureExtractor
        - Input
        - reservoir_input
