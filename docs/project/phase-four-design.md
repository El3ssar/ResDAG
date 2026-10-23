---
description: Approved artifact, diagram and single-pass training contracts for phase four.
---

# Phase four: artifacts and efficient training

Status: **Implemented and verified**, awaiting merge in
[PR #463](https://github.com/El3ssar/ResDAG/pull/463). The [roadmap](roadmap.md) tracks acceptance and delivery.
Phase four includes the maintainer-requested SQ-01 training optimization as
P4-06. Documentation and tests ship alongside each implementation.

## Agreed interfaces

- `model.save("run.pt")` saves the complete model. `rd.load("run.pt")`
  reconstructs its graph without the user remembering its architecture.
- Use standard Torch serialization for tensors and device mapping, versioned
  declarative component recipes for portable construction, and an explicit
  registry for custom code installed by the application.
- Offer an optional cloudpickle bundle for notebook-local classes/functions.
  Saving and loading bundled Python both require explicit `trusted=True`.
- Preserve ordinary-forward reservoir memory by default, with
  `include_state=False` available. Preserve fitted preprocessing and observed
  forecasting history. Reservoir memory does not include delayed forecast
  queues, an iterator cursor or its clock.
- `model.save("run.svg")`, `.png`, `.pdf` and `.dot` export diagrams through
  standard Graphviz/DOT. Inspection does not execute the graph or open viewers.
- `model.fit(data, targets=final_values,
  readout_targets={"stage1": intermediate_values})` trains hidden readouts
  without adding outputs. Internal target tensors use the complete input clock
  and the desired internal output units. Public preprocessing stays unchanged.

## One traversal for multiple readouts

The trainer owns a warmup-plus-training prefix and resets recurrence once.
It executes operations in topological order. At a readout, it fits on the
training suffix only, then forwards the full prefix using those fitted weights.
Downstream reservoirs receive those actual predictions during warmup and
training. Shared branches are reused, and cached values are released after
their last consumer.

Normal PyTorch module calls preserve graph, container and leaf hooks. Standard
`nn.Sequential` expresses nested ordering. Ambiguous custom containers and a
readout object invoked at multiple graph positions are rejected before fitting;
fanout from one readout invocation remains valid. Fitting a shared mutable graph
concurrently is unsupported. Fit before wrapping a model with `torch.compile`. Custom outer forward bodies
are rejected rather than bypassed; customize declared operations or hooks.

Temporal operations now see an uninterrupted prefix. Stochastic layers, including reservoir noise, consume
random draws in a different order from the old replaying implementation;
repeatability is tested, but old noisy fit results can change. The common
`Model.fit` stages and commits on success; the raw trainer mutates in place.

## Verification boundaries

Independent regression/recurrence oracles check fitted coefficients, downstream
features, final states and forecasts. Structural tests count calls and samples,
exercise branching/merge liveness and verify hooks and failure cleanup.
The [controlled CPU benchmark](single-pass-training-benchmark.md) reports fixed
workload measurements, propagation/readout timing, variability and process peak
memory. It is not a cross-library or GPU speed claim.

Artifact acceptance includes fresh-process reconstruction, ordinary-stream
continuation, preprocessing/history/trajectory metadata, graph families, custom
codecs, trusted notebook bundles, storage aliases, device mapping and clear
version errors. Loading does not restore an optimizer, autograd history, live
HPO database connection or an autonomous rollout cursor.
