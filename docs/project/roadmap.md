---
description: Delivery tracker for dependable forecasting, functional DAG composition, PyTorch integration, tuning, persistence and competitive performance.
---

# Roadmap

Last updated: **22 October 2026**. Audit baseline: **ResDAG 1.1.0**;
phase-one corrections released in **1.1.1**.

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
- The common PyTorch workflow applies fitted preprocessing in `model(x)` and
  returns original units; `model.graph(x)` exposes the raw graph. Calls carry
  reservoir state by default. One output returns a tensor; several outputs
  return a dictionary keyed by their declared names.
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
| 1. Correctness and verification | Done | Repair confirmed training, numerical, initialization, diagnostic, HPO and integration failures. | Independent regression oracles pass; full supported CI, optional integration and strict docs are green. |
| 2. Unified functional model workflow | WIP | One PyTorch model lifecycle for presets and custom graphs. | The same documented commands work for both; functional construction and PyTorch semantics are preserved. |
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
| P1-01 | Done | Stable batch and streaming readout statistics. | Offset, rank-deficient and ill-conditioned fixtures agree with independently centered SVD/SciPy references; chunked fitting agrees with batch fitting. |
| P1-02 | Done | Preserve working precision in incremental/RLS readouts; correct fresh RLS prior and FISTA convergence. | `.float()` and factory construction retain the configured accumulation policy; fresh streaming parity; adversarial FISTA optimality; legacy checkpoints migrate explicitly. |
| P1-03 | Done | Correct staged fitting when an upstream fitted readout drives downstream recurrent state. | Fresh cascaded DAG agrees with a closed-form/stagewise oracle; ordinary one-stage fitting retains its fast path. |
| P1-04 | Done | Correct inference mode, multi-head dtype preservation and divergent ensemble behavior. | Repeatable noise-free inference; mode restoration; mixed float/integer output schemas preserved; finite-member and all-invalid policies tested. |
| P1-05 | Done | Coherent sparse weight edits and live versus fixed cell settings. | In-place weight changes affect actual recurrence; accepted configuration writes affect dynamics; fixed settings reject misleading edits. |
| P1-06 | Done | Consistent initializer arguments, RNGs, precision, discovery and spectral scaling. | Direct/name/tuple/object forms agree; required arguments work; local seeds are independent of unrelated draws; custom float64 is preserved; analytic spectral counterexamples pass. |
| P1-07 | Done | Correct calibrated/vector-leak diagnostics and held-out memory-capacity evaluation. | Actual-cell Jacobian/reference comparisons; caller state/modes preserved; held-out estimator validated with stated finite-sample limitations. |
| P1-08 | Done | Repair current TSDynamics analysis integration and test it explicitly. | Current supported TSDynamics analyses work; dependency-free tests run without the optional package; optional CI lane runs genuine integration tests. |
| P1-09 | Done | Stop failed HPO evaluations masquerading as winners; validate transferred objective directions. | Invalid evaluations are not selected as successful models; all-invalid studies fail clearly; incompatible transfer is rejected; error details remain accessible. |
| P1-10 | Done | Repair generated initializer recipes and stale correctness/solver guidance. | All 51 catalog recipes execute; defaults match implementations; examples teach held-out evaluation and training-only scaling. |
| P1-11 | Done | Strengthen CI coverage and documentation selection. | Docs-only changes select executable doc tests; optional integrations have a lane; full coverage remains near the established baseline; no performance thresholds weakened. |
| P1-12 | Done | Review historical benchmark claims and record release verification. | Historical results are clearly separated from current claims; full tests, strict docs, CPU gates, wheel/install checks and hosted CI recorded before release. |

The regular graph now honors the requested degree, including odd degrees when
the node count permits them. Cycle/jump overlaps add both contributions. These
are intentional mathematical corrections approved for phase 1: regenerated
weights and experiment results can change even with the same seed. Migration
notes accompany both initializers.

Phase 1 does not silently change dataset normalization defaults, deep-stack
seed schedules, feedback routing or the meaning of existing `save()` files.
Those require the documented major-release migration below.

### Verification record

Verified **22 October 2026**, runtime source commit `453f34a` in
[PR #458](https://github.com/El3ssar/ResDAG/pull/458). The final tracker update
changes documentation only. Merged into `main` on 22 October 2026 as
`41a1744`.

| Check | Result |
| --- | --- |
| Complete local suite | **4,084 passed**, 234 skipped, 13 benchmark cases deselected; **93.06% coverage**. Forecast golden fixtures are unchanged. |
| Locked hosted CI | **Passed** on Python 3.11–3.14, including lint/types, CPU gates and installed TSDynamics lanes on 3.12–3.14. [Run](https://github.com/El3ssar/ResDAG/actions/runs/39789171340). |
| Latest-dependency hosted CI | **Passed** on Python 3.11–3.14, optional TSDynamics lanes and strict docs. The 3.13 lane tested Torch 2.14.1 and NumPy 2.5.4. [Run](https://github.com/El3ssar/ResDAG/actions/runs/39789185722). |
| Documentation | Strict MkDocs build passes; all **51** catalog recipes and three primary architecture guides execute; rendered roadmap and phase navigation inspected. |
| Optional integration | Real TSDynamics 6.0.2: **32 passed**, one intentional missing-dependency check skipped. |
| CPU performance gates | **4 passed**, four unavailable-device cases skipped, one deselected; no ratio thresholds weakened. |
| Package and compatibility | Wheel/sdist build, clean wheel install, fit/forecast/full-model-load smoke; legacy readout checkpoints and old full-model pickle compatibility tests pass. |

Native MPS correctness checks ran on this Mac. CUDA throughput was not measured.
These checks establish the documented correctness and regression contracts;
they do not establish comparative forecast superiority. The aligned
speed–accuracy study remains in phase 5. For a runnable workflow today, see
[the end-to-end guide](../start/end-to-end.md) and
[the TSDynamics example](ecosystem.md#generate-train-and-forecast).

## Phase 2: one functional workflow

The [phase-two design note](phase-two-design.md) distinguishes agreed behavior,
the agreed construction API, and implementation progress. The maintainer has
approved `rd.Model(inputs={...}, outputs={...})`; later major choices still
require consultation before implementation.

Current boundary: phase 1 orders algebraic fitting inside standard
`nn.Sequential`, but the explicit-state and compiled backends cannot thread
state through reservoirs hidden inside containers. Phase two rejects that case
before running the graph, rather than changing stored state during a supposedly
stateless call. Expose reservoirs as separate functional graph nodes for those
execution paths. Ordinary stateful graph calls remain available.

| ID | Status | Work | Acceptance checks |
| --- | --- | --- | --- |
| P2-01 | WIP | Agree a minimal common model lifecycle and capability contract. | `rd.Model` named functional constructor, original-unit fit/forecast, carried state and named multi-output results implemented for symbolic DAGs, parallel local-state graphs and coupled ensembles. Remaining: facade migration and public capability contract. |
| P2-02 | WIP | Preserve functional construction while evaluating existing graph tooling. | Functional construction retained; [standard-tooling review](graph-tooling.md) records FX/export/module/Graphviz boundaries. Executable FX leaf tracing preserves ownership, state, values and upstream gradients for tensor and named multi-head pipelines. Required hosted checks pending. No private graph API exposed to users. |
| P2-03 | WIP | Named ports, branches/merges, multiple heads and explicit delayed feedback. | Named feedback, fanout, delays and separate per-head prediction leads implemented with independent physical-time and driver-count oracles. Includes lead-aware targets, scaling bridges and metadata. Required full/hosted verification pending. |
| P2-04 | WIP | Coherent cell/state and execution semantics. | Public framewise declaration and explicit full-history execution implemented. Temporal fitting preserves the complete prefix; default fast path remains. Explicit-state visibility, final state, batch switching, hooks and gradients covered across model families. Required full/hosted verification pending. |
| P2-05 | WIP | Training-only preprocessing owned by the convenient workflow. | `Model` owns fitted Torch statistics, inverse transforms and off/method switches across symbolic/parallel/coupled families. Independent training-window, gradient, native MPS and checkpoint oracles pass. Public `Standardize.fit` adopts native precision/device. Remaining: custom transforms, feedback conversion and complete artifacts. |
| P2-06 | WIP | First-class TSDynamics trajectories through the optional extra. | Bare trajectory fitting/forecasting preserves component names, future times and checkpoint-safe provenance. Configurable automatic resampling, fitted-step history alignment and tensor-return option implemented. Remaining: named trajectory mappings and custom-target timing. |
| P2-07 | WIP | Independent per-node RNG streams and public customization templates. | Public `SeedStreams` gives stable named initialization seeds; executable initializer/cell/transform/solver recipes use public APIs, including `ReadoutLayer.solve`. Seeded presets/ensembles preserve caller RNG streams; fresh-process MPS and CPU bit-pattern oracles pass. Required full/hosted verification pending. |
| P2-08 | WIP | Standard PyTorch composition and migration. | Direct new `ESN` module uses the common workflow, eager/build/lazy construction and raw `.graph`; no legacy wrapper. Symbolic/parallel/coupled state, gradients/checkpoints and native MPS covered; FX leaf composition supports presets/custom models. Required full/hosted verification pending. |

### First-increment verification record

Local verification on **22 October 2026**, commit `665b04a`, covers the symbolic-DAG increment,
not completion of phase two. The [runnable guide](../start/model-workflow.md)
shows functional construction, factory wrapping, named targets and PyTorch
composition; its Python examples execute in CI, including on later docs-only
changes.

| Check | Evidence |
| --- | --- |
| Complete local suite | **4,264 passed**, 234 skipped, 13 benchmark cases deselected; **93.04% coverage**. Existing staged-fitting and forecast golden tests pass unchanged. |
| Independent compatibility probe | **120/120** CPU float64 combinations: 12 reservoir configurations, including delayed and GRU/LSTM states, across ten readout/solver configurations. Forecasts agree with manually normalized training; deepcopy and checkpoint parity pass, without convergence warnings in the final configuration. |
| Native MPS | Recurrent float32 fitting with a compatible CG solver, original-unit forecasting, CPU/checkpoint transfer and upstream gradients pass; CPU-fit-then-MPS inference also passes. This is correctness evidence, not a GPU throughput claim. |
| CPU performance gates | **4 passed**, four unavailable-device cases skipped, one deselected; existing thresholds unchanged. |
| PyTorch and checkpoint behavior | Named mappings, state carry/chunks, hooks, gradient preservation, parameter identity, failed-refit isolation, inference-mode fitting/loading and malformed schema rejection have permanent regression tests. |
| Package smoke | Wheel/sdist build and isolated installed-wheel fit/forecast/checkpoint/PyTorch smoke pass, including Torch 2.14.1. |
| Documentation | Strict MkDocs build passes; the common-model guide executes, and rendered guide/roadmap navigation was inspected. |

The first increment deliberately retains the established flat forecast engine
and its single feedback route. Custom preprocessing, feedback conversion,
named delayed routes, trajectory mappings, other backend families and preset
migration remain open. Required hosted CI and strict documentation checks must
pass on the implementation PR before merge.

### Trajectory increment

The [trajectory guide](../start/trajectory-workflow.md) covers direct
`fit(trajectory)`, default trajectory forecasts, the explicit tensor option,
configurable resampling and portable checkpoints. Uniform time axes are the
normal case; irregular axes are resampled automatically with linear, PCHIP or
cubic interpolation. Explicit sampling intervals preserve the last observed
time and never extrapolate. Future histories use the fitted interval.

The integration adds independent physical-time oracles, source/clock/schema
validation, real TSDynamics simulation, failed-refit isolation, legacy
checkpoint and optional-dependency tests. Its runnable guide is included in
the installed-dynamics CI lanes on Python 3.12–3.14, so missing optional
packages cannot silently skip integration verification. Phase two remains WIP.

The adapter, workflow and guide comprise **81 checks**, verified from a clean
installed wheel against the minimum supported **TSDynamics 6.0.2**, and with
the locked **6.0.3** locally. Native MPS fitting, device-local tensor forecasts,
portable metadata, bfloat16 output conversion and strict documentation builds
are covered. Complete-suite and hosted verification are tracked on
[PR #459](https://github.com/El3ssar/ResDAG/pull/459), merged as `d63905d`.
The post-merge [release workflow](https://github.com/El3ssar/ResDAG/actions/runs/39826799461)
passed the full Python 3.11–3.14 test matrix, installed dynamics lanes,
lint/types and package publication. The initial post-merge
[documentation run](https://github.com/El3ssar/ResDAG/actions/runs/39826799223)
also passed. This increment is released in
[1.2.0](https://github.com/El3ssar/ResDAG/releases/tag/v1.2.0).

### Completion increment in progress

The completion increment adds parallel/coupled adapters, direct PyTorch `ESN`,
named delayed feedback and independent prediction leads, full-history temporal
execution/fitting, stable named seeds and public customization recipes. It also
repairs `Standardize` precision/device construction and fitting, scripted inverse,
accelerator RNG isolation, and MPS solve diagnostics. These public changes target
the next **major release**: the maintainer approved the new API directly, without
legacy wrappers or compatibility constraints. Custom preprocessing and named
trajectory alignment still await the corresponding design approvals.

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
| 2026-10-22 | Phase 1 implementation | Done | 4,084 local tests pass at 93.06% coverage; all supported locked/latest CI lanes, strict docs, optional integration, CPU gates and wheel checks pass. See the phase 1 verification record above. |
| 2026-10-22 | Phase 1 merge | Done | PR #458 squash-merged into `main` as `41a1744`; phase-two branch starts from that commit. |
| 2026-10-22 | Phase 1 release | Done | Post-merge full Python 3.11–3.14 matrix, optional dynamics lanes and release job passed; [v1.1.1](https://github.com/El3ssar/ResDAG/releases/tag/v1.1.1) published. Documentation deployment passed. |
| 2026-10-22 | Phase 2 foundations | WIP | Approved `rd.Model` constructor implemented for symbolic DAGs; fitted Torch preprocessing, safe explicit-state validation and executable guide added. Complete common lifecycle is not yet delivered; see the first-increment verification record. |

## Standard tools to reuse

- [PyTorch serialization](https://docs.pytorch.org/docs/stable/notes/serialization.html)
  for tensor/state storage, device mapping and established checkpoint conventions.
- [Graphviz](https://graphviz.readthedocs.io/en/stable/manual.html) for DOT layouts
  and diagram formats; ResDAG already has optional Graphviz integration.
- Existing `pytorch_symbolic` functional composition while graph requirements
  are evaluated; ordinary `nn.Module` and standard Torch tooling for execution.
- Optuna for studies, samplers, pruners and storage; Torch/SciPy for numerical
  solvers; TSDynamics for dynamical-system generation and analysis.
