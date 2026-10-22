---
description: Agreed behavior, construction proposal and implementation boundaries for the common model workflow.
---

# Phase two: the common model workflow

**Status: WIP.** Track delivery in the [roadmap](roadmap.md#phase-2-one-functional-workflow).
This page records design decisions and implementation boundaries. The first
symbolic-DAG increment is implemented; see the runnable
[common model guide](../start/model-workflow.md). The
[established end-to-end guide](../start/end-to-end.md) covers existing APIs.

## Agreed behavior

Basic ESNs and custom graphs will share the same `fit`, `forecast`, `tune` and
`save` commands. Layers will still connect through the existing functional
construction API: `hidden = reservoir(input)` and `output = readout(hidden)`.

The common model will be a PyTorch `nn.Module`:

- `model(x)` will apply scaling learned from training data and return predictions
  in the original units. `model.graph(x)` will expose the raw underlying graph
  for pipelines that manage their own scaling.
- Calls will **carry reservoir state by default**. An explicit reset will start
  an unrelated sequence. Evaluation mode will control noise and other training
  behavior without implicitly resetting state.
- One output will return a tensor. Multiple outputs will return a dictionary
  using the names declared when constructing the model.
- Ordinary forward calls will retain PyTorch gradients and module hooks.
  Freezing reservoir weights will still permit gradients to upstream modules.
- Scaling statistics will be registered buffers, so device/dtype conversions
  and checkpoints include them. Each input and target will have its own
  preprocessing where their units or dimensions differ.

These decisions apply to the new common workflow. Existing graph classes,
`as_module()` defaults, forecast alignment and checkpoint meanings remain the
documented legacy interfaces during migration.

## Approved construction

The maintainer approved `rd.Model(inputs={...}, outputs={...})`. It wraps the
existing symbolic graph in a common module with one registered `graph` child,
reusing the current graph construction and execution infrastructure.

The following illustrates the agreed interface under implementation:

```text
x = rd.reservoir_input(3)
h = reservoir(x)
y = readout(h)

model = rd.Model(inputs={"state": x}, outputs={"prediction": y})
model.fit(train)
prediction = model.forecast(history=observed, horizon=500)
```

An explicit `graph=` form will wrap an existing factory-built graph. Factory
return types and the low-level `ESNModel` class would remain available during
migration. A preset would construct the same common model without requiring
users to wire its layers.

This composition avoids changing the inheritance of existing saved graph
objects. A graph must be registered once; compatibility accessors must refer
to the same object without duplicating parameters or checkpoint entries.

## The first increment

`rd.Model` now accepts named symbolic inputs/outputs or an existing `ESNModel`
through `graph=`. It supports fitting terminal readout heads, original-unit
forward/forecast operations, named multi-head targets and explicit recurrent
state. Ordinary calls carry state by default and retain upstream gradients.
Its state dictionary includes named-port metadata, fitted scaling and the
observed history tail. Loading validates their consistency before using them.

This increment retains the existing first-input/first-output forecast route.
Implicit next-step targets share their input's training statistics, so feedback
units agree. Explicit targets receive separate statistics; autonomous
forecasting rejects differing feedback scales until a conversion bridge is
implemented. Parallel/coupled adapters, the `ESN` facade migration and arbitrary
feedback routes remain unfinished.

The first increment packages the existing four normalization methods in an
internal fitted PyTorch module. Statistics are learned only by an explicit
fit operation, remain detached from training gradients, and do not update
during inference. Transform and inverse operations preserve upstream gradients.
Tests cover constant channels, refitting, checkpoint restoration, invalid data,
and device/dtype behavior, including native MPS. The module is used internally
by `Model`; a custom-transform extension contract and integration into the
remaining model families are still pending. Existing data-preparation defaults
are unchanged. Minmax scaling now divides before multiplying, avoiding an
intermediate overflow for large finite training ranges.

The first increment also guards explicit-state execution against reservoirs
hidden inside containers or custom modules. Previously, such a call could
change stored state and omit it from the returned state dictionary. The guard
checks the entire execution plan before running any operation and provides a
remedy: expose each reservoir as a separate functional graph node. Ordinary
stateful graph calls and supported pointwise containers remain usable. See
[state management](../start/concepts.md#2-reservoir-layers-are-stateful).

## Decisions before the next implementation increments

- Define a clear way to build lazy presets before attaching an optimizer.
- Define what a feedback delay counts in physical sample time. A one-step
  prediction cannot simply be fed through a longer queue without defining how
  training targets, history and external drivers align.
- Define a small capability interface for custom temporal modules, including
  explicit state, causal stepping and compilation support. A module accepting
  a sequence does not establish that it can process one time step at a time.
- Define trajectory time/metadata behavior and an explicit irregular-sampling
  policy for the optional TSDynamics integration.

Tuning presets remain in phase three. Complete model loading and diagram
exports remain in phase four. They will use the agreed common interface, with
separate acceptance tests rather than placeholder methods.
