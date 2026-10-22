---
description: Implemented behavior, agreed construction and boundaries for the common model workflow.
---

# Phase two: the common model workflow

**Status: Done; awaiting merge and major release.** See verification in the
[roadmap](roadmap.md#completion-verification).
This page records current decisions and implementation boundaries. Start with
the runnable [common model guide](../start/model-workflow.md) or the shorter
[end-to-end guide](../start/end-to-end.md).

## Agreed behavior

Basic ESNs and custom graphs share the same `fit`, `forecast` and state workflow.
Unified `tune` and complete-artifact `save` commands remain phases three and four.
Layers connect through the existing functional
construction API: `hidden = reservoir(input)` and `output = readout(hidden)`.

The common model is a PyTorch `nn.Module`:

- `model(x)` applies scaling learned from training data and returns predictions
  in the original units. `model.graph(x)` exposes the raw underlying graph
  for pipelines that manage their own scaling.
- Calls **carry reservoir state by default**. An explicit reset starts
  an unrelated sequence. Evaluation mode controls noise and other training
  behavior without implicitly resetting state.
- One output returns a tensor. Multiple outputs return a dictionary
  using the names declared when constructing the model.
- Ordinary forward calls retain PyTorch gradients and module hooks.
  Freezing reservoir weights still permits gradients to upstream modules.
- Scaling statistics are registered buffers, so device/dtype conversions
  and checkpoints include them. Each input and target has its own
  preprocessing where their units or dimensions differ.

These decisions apply directly to the new major API. The maintainer chose
no legacy wrapper or compatibility layer. Raw graph classes and trainers
remain useful public building blocks for advanced workflows.

## Approved construction

The maintainer approved `rd.Model(inputs={...}, outputs={...})`. It wraps the
existing symbolic graph in a common module with one registered `graph` child,
reusing the current graph construction and execution infrastructure.

The following illustrates the implemented interface:

```text
x = rd.reservoir_input(3)
h = reservoir(x)
y = readout(h)

model = rd.Model(inputs={"state": x}, outputs={"prediction": y})
model.fit(train)
prediction = model.forecast(history=observed, horizon=500)
```

The `graph=` form wraps an existing factory-built graph. The PyTorch `ESN`
preset constructs one common workflow without requiring users to wire layers.
Use `input_size=` or `build()` before attaching optimizers; an untouched lazy
preset can instead infer width, dtype and device on its first fit.

Each graph is registered once, preserving parameter identities and avoiding
duplicate checkpoint entries. See [the ESN API changes](esn-migration.md).

## Current workflow

`rd.Model` accepts named symbolic inputs/outputs or an existing graph through
`graph=`. It fits declared readout heads, returns original-unit forward and
forecast predictions, and supports named targets and explicit recurrent state.
Its state dictionary includes named-port/feedback metadata, fitted scaling and
observed history. Loading validates their consistency before using them.

Declare every algebraically fitted readout as a named output and supply its
target. An intermediate head can also feed a downstream reservoir; staged
training fits it before collecting that reservoir's features. Unexposed
internal fitted heads require the lower-level trainer's explicit targets.

`Feedback` connects named output/input ports with an explicit graph-step delay.
`prediction_lead` separately records each head's target distance. Default
self-feedback uses matching lead/delay, and forecasts always start at the first
unseen physical sample. Pending warmup outputs prevent gaps for multi-step
heads. Differing output/input scales are bridged through original units on the
model device. See [feedback and timing](../start/feedback.md).

Custom framewise modules can declare `supports_framewise=True`. Pure temporal
sequence operations can use `execution="history"`: fitting and forecasting
preserve the complete prefix. Fast models retain the optimized existing
backend and compact warmup tail. Delayed rollout queues are reconstructed from
history; reservoir state alone cannot resume such a rollout. See
[execution capabilities](../advanced/forecast-capabilities.md).

## Existing backend families

`Model(graph=...)` now also wraps `ParallelESNModel` and
`CoupledEnsembleESNModel`. Small private adapters reuse their existing
execution paths; the raw graph remains the sole registered graph child.
Parallel ports represent full fields, and scaling precedes patch gathering.
Coupled ports represent the shared member inputs and single aggregate output;
each member's terminal readout name is resolved independently during fitting.

State, gradients, training-only scaling, checkpoint metadata and trajectory
conversion use the same workflow. Parallel/coupled compiled forecasting is
rejected before state changes because those raw forecast paths do not support
the compile option. Their explicit-state paths keep their existing support.
A reservoir hidden inside a custom coupled aggregator cannot participate in
explicit-state or forecast execution; it is rejected before operations run.
Ordinary stateful calls remain available.

Non-symbolic checkpoint schemas include patch geometry or ordered member
interfaces, the aggregate selector/class, and persistent tensor name/shape
signatures. They reject structural mismatches before copying tensors. Custom
Python settings still belong to the compatible architecture supplied by the
caller. Generic architecture reconstruction remains phase four.

The [graph-tooling review](graph-tooling.md) records the evaluated standard
tools. Functional construction stays in place; FX can capture a surrounding
PyTorch pipeline using the documented leaf-module hook.

## Trajectory integration

The maintainer approved automatic, configurable resampling for uneven time
axes, with uniform sampling as the normal case. A model fitted on a bare
TSDynamics trajectory now returns a trajectory by default. The optional
`return_type="tensor"` keeps forecasts on the model device; ordinary forward
calls retain the Torch interface.

The trajectory's time axis defines sampling. Default interpolation preserves
both endpoints and sample count; an explicit `resample_dt` creates an
in-support grid ending at the latest observation. Later trajectory histories
use the fitted interval and the same component names/order. Predictions begin
at the immediate unseen time; optional warmup predictions start at
`history[max(prediction_leads.values()):]`, so every head shares the same
physical timeline. New tensor histories require tensor output because
they carry no absolute clock.

Named trajectory inputs align once on a shared grid inside their common
observed range. The first declared model input supplies the default interval;
`resample_dt` overrides it. Per-head trajectory targets are physical signals:
an origin at `t` uses that head's label at `t + prediction_lead * dt`. Labels
beyond the common observed endpoint are held out before interpolation.
Fitting preserves the full input prefix and uses only jointly valid supervised
rows after washout. Fast models retain a compact tail ending at the common
observed endpoint; temporal models retain the full prefix.

Clock, per-port components and primitive source metadata survive current
state-dictionary checkpoints without requiring TSDynamics at load time. Live
producing-system objects remain transient. One output returns a trajectory;
several return a dictionary with each target's own signal names and one shared
future timeline. See the [trajectory guide](../start/trajectory-workflow.md)
and [named trajectories](../start/named-trajectories.md).

The common workflow packages the existing four normalization methods in an
internal fitted PyTorch module. Statistics are learned only by an explicit
fit operation, remain detached from training gradients, and do not update
during inference. Transform and inverse operations preserve upstream gradients.
Tests cover constant channels, refitting, checkpoint restoration, invalid data,
and device/dtype behavior, including native MPS. The module is used internally
by every supported `Model` family. Named `input_transforms` and
`output_transforms` override individual ports with reversible Torch modules,
including `Standardize` and `nn.Identity`. Fitting stages independent copies
and preserves the original modules and parameters on successful commit.
See [custom preprocessing](../advanced/custom-preprocessing.md).
Existing data-preparation defaults
are unchanged. Minmax scaling now divides before multiplying, avoiding an
intermediate overflow for large finite training ranges.

The workflow also guards explicit-state execution against reservoirs
hidden inside containers or custom modules. Previously, such a call could
change stored state and omit it from the returned state dictionary. The guard
checks the entire execution plan before running any operation and provides a
remedy: expose each reservoir as a separate functional graph node. Ordinary
stateful graph calls and supported pointwise containers remain usable. See
[state management](../start/concepts.md#2-reservoir-layers-are-stateful).

## Approved extensions

The maintainer approved these choices on 22 October 2026:

- Custom preprocessing uses reversible changes of units that preserve signals.
  Temporal operations and transforms that add features belong inside the DAG.
- Named TSDynamics inputs align automatically to a common sampling grid within
  their shared observed range. Resampling never extrapolates beyond observations.
  Targets use physical timestamps and each head's declared prediction lead.

Both extensions are implemented. Complete integrated and hosted verification
passes; see the roadmap's completion record. The new major API uses current version-2 workflow/preset
checkpoint metadata. Older common-workflow checkpoints require reconstruction
and refitting; no compatibility converter is provided.

The [public component recipes](../advanced/custom-components.md) cover stable
named seeds, registered initializers, recurrent cells, framewise transforms and
the public `ReadoutLayer.solve` hook. User modules remain ordinary PyTorch
components.

Tuning remains phase three. Complete model loading and diagram exports remain
phase four. They will use the agreed common interface, with separate acceptance
tests rather than placeholder methods.
