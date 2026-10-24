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

Use `Feedback(output="prediction", input="state", delay=1)` to declare a
named recurrent connection. Prediction leads and feedback delays have separate
meanings; see [feedback and clocks](../start/feedback.md).

`forward_stateless` returns predictions and a `ReservoirStates` mapping without
changing stored recurrent state. Reuse the returned mapping for differentiable
chunks. Raw graph `as_module()` returns a `ReservoirModelModule` that shares
the complete graph and its parameters; `ReservoirFeatureExtractor` selects
reservoir features instead. These adapters serve different pipeline needs.

Standard inherited PyTorch operations, including `parameters`, `state_dict`,
`load_state_dict`, `train`, `eval` and `to`, remain available. Their inherited
reference entries and checkpoint protocol hooks are omitted here.

::: resdag.core
    options:
      members:
        - Model
        - Feedback
        - ESNModel
        - ReservoirModelModule
        - ReservoirFeatureExtractor
        - Input
        - reservoir_input

## Recurrent state type

The exported `ReservoirStates` type is `dict[str, torch.Tensor]`. Keys identify
reservoirs in the owning graph; each value follows that cell's state layout.
The mapping returned by a model's state methods is accepted by its
`forward_stateless` method. State keys are specific to that model's wiring.

::: resdag.core.ReservoirStates
    options:
      show_if_no_docstring: true
