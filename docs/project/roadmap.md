---
description: Delivery tracker for dependable forecasting, functional DAG composition, PyTorch integration, tuning, persistence and competitive performance.
---

# Roadmap

Last updated: **22 October 2026**. Baseline: **ResDAG 1.1.0**.

This is the canonical delivery tracker. Update the status, evidence and next
step in the same pull request as the implementation. A task is **Done** only
after its acceptance checks pass and its public behavior is documented.

| Status | Meaning |
| --- | --- |
| Pending | Agreed work that has not started, or depends on an earlier task. |
| WIP | Implementation or verification is in progress; it is not a release claim. |
| Done | Implementation, regression tests, documentation and required CI checks are complete. |

## Goals and decisions

- Autonomous dynamical-system forecasting is the easiest default workflow.
- Presets and custom DAGs use the same fit, forecast, tune and save lifecycle.
- Keep the expressive **functional construction API**, similar to building a
  Keras functional model. `pytorch_symbolic` remains supported while its
  limitations and available standard PyTorch alternatives are evaluated.
  Changing the backend must preserve that construction experience.
- Prefer existing PyTorch, Optuna, Graphviz, SciPy and TSDynamics capabilities.
  Add a ResDAG abstraction only when a demonstrated requirement needs it.
- Keep every component replaceable through public interfaces: initializers,
  cells, state, transforms, readouts, solvers, objectives and search spaces.
- Automatic defaults must be dependable; users can force the solver,
  precision, device, preprocessing and execution choices.
- Learn preprocessing from training data and return forecasts in original
  units. Keep preprocessing configurable and optional.
- Provide explicit delayed feedback connections and named inputs/outputs.
- Keep TSDynamics optional, with first-class trajectory input/output.
- Tuning has quick, balanced and thorough presets. Balanced is the default;
  the winning model is fitted and ready to forecast afterward.
- Compare forecast accuracy, speed and memory together. Publish reproducible
  workload-specific results and distinguish measured hardware from skipped checks.
- Breaking simplifications belong in a major release with migration guidance.
  Consult the maintainer before major implementation choices.

## Delivery overview

| Phase | Status | Deliverable | Exit condition |
| --- | --- | --- | --- |
| 0. Baseline audit | Done | Review all 173 runtime modules, 62 documentation pages and 17 examples; reproduce high-impact defects. | Explicit coverage inventory, independent numerical/state probes and chronological HPO pilot recorded. |
| 1. Correctness and verification | WIP | Repair confirmed training, numerical, initialization, diagnostic, HPO and integration failures. | Independent regression oracles pass; full supported CI, optional integration and strict docs are green. |
| 2. Unified functional model workflow | Pending | One PyTorch model lifecycle for presets and custom graphs. | The same documented commands work for both; functional construction and PyTorch semantics are preserved. |
| 3. Reliable automatic tuning | Pending | Robust tuning with little setup and complete winner provenance. | Quick/balanced/thorough studies reproduce, respect budgets and leave a fitted winner. |
| 4. Model artifacts and diagrams | Pending | Save a complete model, load it without rebuilding its DAG, and export readable graph diagrams. | Fresh-process load/forecast parity and diagram correctness for every supported graph family. |
| 5. Performance and competitive evidence | Pending | Improve measured bottlenecks; publish fair speed–accuracy–memory comparisons. | Reproducible matched-workload and equal-budget results across systems, seeds and scale regimes. |
| 6. Experiments and release readiness | Pending | Stable experiment recipes, executable documentation and migration tools. | End-to-end experiments run from published packages with recorded environments and artifacts. |

Phase 4 can begin once the graph/state artifact contract in phase 2 is agreed;
it does not wait for every tuning or performance feature. Documentation and CI
are part of every phase, not deferred cleanup.

## Phase 1: correctness and verification

Preserve current public workflows while fixing incorrect behavior. Numerical
corrections may change forecasts; explain those changes and review affected
golden fixtures against independent references.

| ID | Status | Work | Acceptance checks / evidence |
| --- | --- | --- | --- |
| P1-01 | WIP | Stable batch and streaming readout statistics. | Offset, rank-deficient and ill-conditioned fixtures agree with independently centered SVD/SciPy references; chunked fitting agrees with batch fitting. |
| P1-02 | WIP | Preserve working precision in incremental/RLS readouts; correct fresh RLS prior and FISTA convergence. | `.float()` and factory construction retain the configured accumulation policy; fresh streaming parity; adversarial FISTA optimality; legacy checkpoints migrate explicitly. |
| P1-03 | WIP | Correct staged fitting when an upstream fitted readout drives downstream recurrent state. | Fresh cascaded DAG agrees with a closed-form/stagewise oracle; ordinary one-stage fitting retains its fast path. |
| P1-04 | WIP | Correct inference mode, multi-head dtype preservation and divergent ensemble behavior. | Repeatable noise-free inference; mode restoration; mixed float/integer output schemas preserved; finite-member and all-invalid policies tested. |
| P1-05 | WIP | Coherent sparse weight edits and live versus fixed cell settings. | In-place weight changes affect actual recurrence; accepted configuration writes affect dynamics; fixed settings reject misleading edits. |
| P1-06 | WIP | Consistent initializer arguments, RNGs, precision, discovery and spectral scaling. | Direct/name/tuple/object forms agree; required arguments work; local seeds are independent of unrelated draws; custom float64 is preserved; analytic spectral counterexamples pass. |
| P1-07 | WIP | Correct calibrated/vector-leak diagnostics and held-out memory-capacity evaluation. | Actual-cell Jacobian/reference comparisons; caller state/modes preserved; held-out estimator validated with stated finite-sample limitations. |
| P1-08 | WIP | Repair current TSDynamics analysis integration and test it explicitly. | Current supported TSDynamics analyses work; dependency-free tests run without the optional package; optional CI lane runs genuine integration tests. |
| P1-09 | WIP | Stop failed HPO evaluations masquerading as winners; validate transferred objective directions. | Invalid evaluations are not selected as successful models; all-invalid studies fail clearly; incompatible transfer is rejected; error details remain accessible. |
| P1-10 | WIP | Repair generated initializer recipes and stale correctness/solver guidance. | All 51 catalog recipes execute; defaults match implementations; examples teach held-out evaluation and training-only scaling. |
| P1-11 | WIP | Strengthen CI coverage and documentation selection. | Docs-only changes select executable doc tests; optional integrations have a lane; full coverage remains near the established baseline; no performance thresholds weakened. |
| P1-12 | WIP | Review historical benchmark claims and record release verification. | Historical results are clearly separated from current claims; full tests, strict docs, CPU gates, wheel/install checks and hosted CI recorded before release. |

The regular graph now honors the requested degree, including odd degrees when
the node count permits them. Cycle/jump overlaps add both contributions. These
are intentional mathematical corrections approved for phase 1: regenerated
weights and experiment results can change even with the same seed. Migration
notes accompany both initializers.

Phase 1 does not silently change dataset normalization defaults, deep-stack
seed schedules, feedback routing or the meaning of existing `save()` files.
Those require the documented major-release migration below.

## Phase 2: one functional workflow

Current boundary: phase 1 orders algebraic fitting inside standard
`nn.Sequential`, but does not extend the explicit-state or compiled backend to
reservoirs hidden inside arbitrary containers. Expose reservoirs as separate
functional graph nodes for those execution paths. The capability and state
contracts below address that remaining limitation.

| ID | Status | Work | Acceptance checks |
| --- | --- | --- | --- |
| P2-01 | Pending | Agree a minimal common model lifecycle and capability contract. | Preset, branched DAG, parallel and coupled examples use the same lifecycle; clear unsupported-operation errors. |
| P2-02 | Pending | Preserve functional construction while evaluating existing graph tooling. | Compare `pytorch_symbolic`, standard PyTorch graph/shape tooling and a minimal adapter against actual requirements before replacing infrastructure. No private graph API exposed to users. |
| P2-03 | Pending | Named ports, branches/merges, multiple heads and explicit delayed feedback. | Instantaneous graph is acyclic; delayed loops, several feedback streams and driver alignment have independent timing oracles. |
| P2-04 | Pending | Coherent cell/state and execution semantics. | Pointwise, stateful and sequence-only capabilities are explicit; gradients/hooks, reset/chunks and output schemas agree across ordinary and accelerated paths. |
| P2-05 | Pending | Training-only preprocessing owned by the convenient workflow. | Automatic original-unit forecasts; custom/off switches; no validation leakage; preprocessing is included in model artifacts. |
| P2-06 | Pending | First-class TSDynamics trajectories through the optional extra. | Preserve sampling times, component names and provenance; forecasts have correct future times; irregular sampling requires an explicit policy. |
| P2-07 | Pending | Independent per-node RNG streams and public customization templates. | Reproducible heterogeneous graphs; explicit tied/per-node seeds; tiny custom initializer, cell, transform and solver examples without private APIs. |
| P2-08 | Pending | Standard PyTorch composition and migration. | Single-output models fit ordinary neural pipelines; frozen reservoirs pass upstream gradients; train/eval, devices, parameter registration and checkpoints behave conventionally. |

## Phase 3: tuning and automatic solver policy

| ID | Status | Work | Acceptance checks |
| --- | --- | --- | --- |
| P3-01 | Pending | Unified `model.tune(...)` with quick/balanced/thorough presets. | The model is fitted with the winner and ready to forecast; advanced Optuna study/report access is retained. |
| P3-02 | Pending | Several seeds/origins with transparent resource budgets. | Screening and robust promotion respect trial/time/memory budgets; untouched final tests remain separate; sampling-time units are explicit. |
| P3-03 | Pending | Complete trial provenance and safe study transfer/resume. | Persist resolved parameters, actual seeds, preprocessing, split identity, objective units and environment; re-evaluate incompatible prior configurations. |
| P3-04 | Pending | Pruning during rollout and device-local scoring. | Stopped trials skip forecast computation; tensor/NumPy metric parity; no hidden device transfers in the supported fast path. |
| P3-05 | Pending | Stable automatic readout selection with forced overrides. | Standard primal/dual/QR/SVD choices are validated; report solver, working precision, conditioning/residual diagnostics and fallback; never hide changed regularization. |

## Phase 4: diagrams and complete model artifacts

The following is the **target API**, not a claim about current `save()` behavior:

```python
model.save("experiment.pt")       # complete reloadable model artifact
restored = rd.load("experiment.pt")  # no DAG reconstruction by the caller
model.save("experiment.svg")      # diagram export
model.save("experiment.png")
```

A diagram is a visualization, not a reloadable model. Preserve explicit
low-level weights/state checkpoint operations for PyTorch users. Resolve the
complete-artifact format and extension policy before changing legacy saves.

| ID | Status | Work | Acceptance checks |
| --- | --- | --- | --- |
| P4-01 | Pending | Versioned complete artifact: graph, component configuration, weights, preprocessing, metadata and optional streaming state. | Use standard PyTorch serialization and existing schema/packaging conventions where suitable; load in a fresh process without a manually built DAG. |
| P4-02 | Pending | Generic `rd.load(...)`, version migration and custom-component reconstruction. | Presets, heterogeneous custom DAGs and registered components round-trip; custom Python code requirements are explicit; unknown versions/components fail with useful remedies. |
| P4-03 | Pending | Save diagram as SVG, PNG, PDF or DOT with clear labels and shapes. | Reuse existing Graphviz/DOT rendering; show branches, merges, named ports and delayed feedback; export does not mutate model state/modes or open a viewer unexpectedly. |
| P4-04 | Pending | Unified save dispatch and user documentation. | Supported suffixes are discoverable; unsupported formats fail clearly; diagram exports and model artifacts cannot be confused. |
| P4-05 | Pending | Fresh-process, cross-device and legacy artifact verification. | Forecast and preprocessing parity after reload; CPU map-location works; documented legacy files remain loadable or have a tested converter. |

## Phase 5: speed, memory and comparisons

| ID | Status | Work | Acceptance checks |
| --- | --- | --- | --- |
| P5-01 | Pending | Correct held-out comparison windows and common prediction alignment. | Shared data/splits and one-step timing oracles across adapters; historical tables are not treated as new measurements. |
| P5-02 | Pending | Controlled numerical and idiomatic equal-budget benchmark tracks. | Several systems, trajectories, seeds and scale regimes; record uncertainty, initialization/compile cost, fit/forecast time and memory. |
| P5-03 | Pending | Actual sparse/structured operators and appropriate matrix-free readouts. | Reuse Torch/SciPy operator/sparse/solver capabilities; memory scales as advertised; no mandatory hidden dense copy; save/load and numerical contracts pass. |
| P5-04 | Pending | Profile-driven batching and compilation improvements. | Reference/fast value, state and gradient parity; measured improvement in specified workloads; setup and steady state reported separately. |
| P5-05 | Pending | Publish speed–accuracy–memory tradeoffs against current competitors. | Reproducible commands, versions, hardware and equal search budgets; no universal unsupported superiority or GPU claim. |

## Phase 6: experiment readiness and ongoing documentation

| ID | Status | Work | Acceptance checks |
| --- | --- | --- | --- |
| P6-01 | Pending | Canonical generate → split → tune/fit → forecast → evaluate → save → PyTorch integration guide. | Runs from a clean published-package environment; includes a complete TSDynamics example and original-unit predictions. |
| P6-02 | Pending | Progressive customization guides and generated capability facts. | Every public extension point has a small runnable recipe; examples/defaults are driven by shared metadata where possible. |
| P6-03 | Pending | Bounded executable docs/examples in CI, optional compatibility and release artifact checks. | Run example bodies, not only imports; requirements/skips are explicit; actual package install and reload are tested. |
| P6-04 | Pending | Stable experiment configurations, manifests and migration notes. | Reproduce saved models and metrics with pinned dependencies; users can resume or compare experiments without reconstructing hidden settings. |

## Completion log

| Date | Task | Status | Evidence / next step |
| --- | --- | --- | --- |
| 2026-10-22 | Baseline audit | Done | 173 runtime modules reviewed; standard suite 3,809 passed, 91.63% coverage; optional TSDynamics and catalog failures independently reproduced. |
| 2026-10-22 | Roadmap and maintainer decisions | Done | Canonical tracker is in the documentation navigation; strict MkDocs build and links pass. |
| 2026-10-22 | Phase 1 implementation | WIP | Fixes and regression oracles implemented; strict docs, whole-package typing, CPU gates and wheel smoke checks pass. Final full-suite and hosted CI verification are in progress. |

## Standard tools to reuse

- [PyTorch serialization](https://docs.pytorch.org/docs/stable/notes/serialization.html)
  for tensor/state storage, device mapping and established checkpoint conventions.
- [Graphviz](https://graphviz.readthedocs.io/en/stable/manual.html) for DOT layouts
  and diagram formats; ResDAG already has optional Graphviz integration.
- Existing `pytorch_symbolic` functional composition while graph requirements
  are evaluated; ordinary `nn.Module` and standard Torch tooling for execution.
- Optuna for studies, samplers, pruners and storage; Torch/SciPy for numerical
  solvers; TSDynamics for dynamical-system generation and analysis.
