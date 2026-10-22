---
description: Delivery tracker for dependable forecasting, functional DAG composition, PyTorch integration, tuning, persistence and competitive performance.
---

# Roadmap

Last updated: **23 October 2026**. Audit baseline: **ResDAG 1.1.0**;
phase-one corrections released in **1.1.1**, and the initial trajectory
workflow in **1.2.0**. Phase-two completion was merged through
[PR #460](https://github.com/El3ssar/ResDAG/pull/460) as `4735542`.
The [2.0.0 release](https://github.com/El3ssar/ResDAG/releases/tag/v2.0.0)
and its documentation deployment are verified; phase three is in progress.

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
| 2. Unified functional model workflow | Done | One PyTorch model lifecycle for presets and custom graphs. | The same documented fit/forecast/state commands work for both; functional construction and PyTorch semantics are preserved. Tuning and complete artifacts remain phases 3 and 4. |
| 3. Reliable automatic tuning | WIP | Robust tuning with little setup and complete winner provenance. | Quick/balanced/thorough studies reproduce, respect budgets and leave a fitted winner. |
| 4. Model artifacts and diagrams | Pending | Save a complete model, load it without rebuilding its DAG, and export readable graph diagrams. | Fresh-process load/forecast parity and diagram correctness for every supported graph family. |
| 5. Performance and competitive evidence | Pending | Improve measured bottlenecks; publish fair speed–accuracy–memory comparisons. | Reproducible matched-workload and equal-budget results across systems, seeds and scale regimes. |
| 6. Experiments and release readiness | Pending | Stable experiment recipes, executable documentation and migration guidance. | End-to-end experiments run from published packages with recorded environments and artifacts. |

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
| P2-01 | Done | Agree a minimal common model lifecycle and capability contract. | Native `rd.ESN` and named `rd.Model` share original-unit fit/forecast, carried state and named results across symbolic DAGs, parallel local-state graphs and coupled ensembles. Public fast/history capability contract verified. Tuning and complete artifacts remain phases 3 and 4. |
| P2-02 | Done | Preserve functional construction while evaluating existing graph tooling. | Functional construction retained; [standard-tooling review](graph-tooling.md) records FX/export/module/Graphviz boundaries. Executable FX leaf tracing preserves ownership, state, values and upstream gradients for tensor and named multi-head pipelines. No private graph API exposed to users. |
| P2-03 | Done | Named ports, branches/merges, multiple heads and explicit delayed feedback. | Named feedback, fanout, delays and separate per-head prediction leads pass independent physical-time and driver-count oracles. Includes lead-aware targets, scaling bridges and metadata. |
| P2-04 | Done | Coherent cell/state and execution semantics. | Public framewise declaration and explicit full-history execution verified. Temporal fitting preserves the complete prefix; default fast path remains. Explicit-state visibility, final state, batch switching, hooks and gradients covered across model families. |
| P2-05 | Done | Training-only preprocessing owned by the convenient workflow. | Fitted Torch statistics, inverse transforms, feedback unit conversion and off/method switches work across symbolic/parallel/coupled families. Independent training-window, gradient, native MPS and checkpoint oracles pass. Per-port reversible custom modules and identity overrides preserve module/parameter ownership through staged fit and state-only validation. Complete artifacts belong to phase 4. |
| P2-06 | Done | First-class TSDynamics trajectories through the optional extra. | Bare and named trajectory mappings preserve component names, future times and checkpoint-safe provenance. Configurable resampling, automatic shared-range alignment, physical lead-aware targets, held-out target clipping and tensor-return option pass installed-dependency and independent clock oracles. |
| P2-07 | Done | Independent named initialization seeds and public customization templates. | Public `SeedStreams` gives stable named initialization seeds; executable initializer/cell/transform/solver recipes use public APIs, including `ReadoutLayer.solve`. Seeded presets/ensembles preserve caller RNG streams; fresh-process MPS and CPU bit-pattern oracles pass. |
| P2-08 | Done | Standard PyTorch composition and migration. | Direct new `ESN` module uses the common workflow, eager/build/lazy construction and raw `.graph`; no legacy wrapper. Symbolic/parallel/coupled state, gradients/checkpoints and native MPS covered; FX leaf composition supports presets/custom models. |

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

At that checkpoint, the first increment retained the established flat forecast engine
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
packages cannot silently skip integration verification. This was an intermediate
delivery; the completion record below covers the remaining phase-two work.

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

### Completion verification

The completion increment adds parallel/coupled adapters, direct PyTorch `ESN`,
named delayed feedback and independent prediction leads, full-history temporal
execution/fitting, stable named seeds and public customization recipes. It also
repairs `Standardize` precision/device construction and fitting, scripted inverse,
accelerator RNG isolation, and MPS solve diagnostics. These public changes target
the next **major release**: the maintainer approved the new API directly, without
legacy wrappers or compatibility constraints. Reversible, signal-preserving custom
preprocessing and automatic shared-range alignment of named trajectories are
implemented and verified as well.

Verification of source `2942bb5a944248deadd977546a3dff14be41f6e2` on
22 October 2026. The runtime tree is identical to `7f294ba`; the final source
adds the stricter coverage gate and 83 dependency-free clock/CI regression cases.

| Check | Evidence |
| --- | --- |
| Complete local runtime suite with dynamics | **4,741 passed**, 231 skipped, 13 benchmark cases deselected; **92.81% coverage** on `7f294ba`, before the final 83 CI/clock cases were added. |
| Complete final local suite without dynamics | **4,713 passed**, 342 skipped, 13 benchmark cases deselected; **91.08% coverage**, with the strict floor enforced. Native MPS supplies additional local device checks. |
| Installed branch wheel | **462 passed**, three skipped, including native MPS correctness, with locked **Torch 2.10.0 / TSDynamics 6.0.3**. All **183 Python modules** match the runtime source byte-for-byte; the rebuilt final wheel is identical to this tested wheel. This is an unreleased branch build, not the published 1.2.0 package. |
| Minimum optional dependency and newer Torch | A further **462 passed**, three skipped, from the same installed branch wheel with **TSDynamics 6.0.2 / Torch 2.14.1**. |
| CPU performance gates | **Four passed**, four unavailable-device cases skipped, one deselected; thresholds unchanged. CUDA throughput was not measured. |
| Supplemental complete CPU benchmark selection | **Seven passed, two failed** locally. The same sparse/parallel timing assertions fail against unchanged `main`; these pre-existing hardware limits are recorded in phase 5 below. They are outside the current four-test CI performance job. |
| Static and package checks | CI-scoped Ruff/Black, whole-package mypy (**183 modules**), lock validation, actionlint and wheel/sdist builds pass. Both archives match runtime source and exclude local coordination notes. |
| Documentation | Strict locked/latest MkDocs builds and **15 executable guide checks** pass. Rendered desktop guide navigation, custom preprocessing, named trajectories and feedback layout inspected. |
| Full hosted suites | [Locked CI](https://github.com/El3ssar/ResDAG/actions/runs/39845805396) and [latest compatibility](https://github.com/El3ssar/ResDAG/actions/runs/39845806724) pass all eight Python 3.11–3.14 suites. Each: **4,625 passed**, 430 skipped, 13 deselected; **90.97–91.01% coverage** against the unchanged 90% floor. |
| Hosted integration and docs | All six Python 3.12–3.14 dynamics lanes pass **212 checks** each, seven skipped, using locked **6.0.3** and latest **6.1.0**. Strict docs, lint/types and CPU benchmark/report jobs pass. |

The final CI fix sets coverage report precision to two decimal places: integer
rounding had allowed a below-90% result to exit successfully. A real pytest
subprocess regression now verifies that **89.8% fails** and **90.2% passes**.
The additional clock tests exercise pure alignment/interpolation/checkpoint
logic without requiring TSDynamics; no first-party code was excluded to reach
the floor. Final prose-only roadmap updates retain this runtime/test/config
tree and run the required PR checks again.

## Phase 3: tuning and automatic solver policy

Implementation started on **22 October 2026**, from the phase-two merge.
The new `model.tune(...)` lifecycle uses the common `Model`/`ESN` workflow,
including preprocessing, named feedback and trajectory clocks. The maintainer
approved chronological 80/20 validation, refitting the winner on supplied
development data, Python recipes for structural candidates, the longest accurate
forecast as the default reward, weakest-output aggregation and joblib/loky
parallel trials. The initial implementation is verified in
[PR #461](https://github.com/El3ssar/ResDAG/pull/461). See the
[phase-three design](phase-three-design.md) for the agreed experience and
data/timing safeguards.

Independent audits reproduced a finite-input SVD overflow, acceptance of
non-finite regularization, invalid infinite-score transfer, incorrect
maximize-study summary ordering, and incomplete process/storage cleanup.
The correctness fixes, common tuner and automatic solver are implemented with
independent numerical, time-alignment, ownership and real-worker oracles.
Pruning during forecast generation and staged resource promotion remain open;
this is not a claim that the whole phase is complete.

| ID | Status | Work | Acceptance checks |
| --- | --- | --- | --- |
| P3-01 | Done | Unified `model.tune(...)` with quick/balanced/thorough presets. | Presets/custom DAGs, named targets and trajectories; original-unit forecasts, transactional winner refit, report/study access and joblib execution pass complete local/package/hosted verification below. |
| P3-02 | WIP | Several seeds/origins with transparent resource budgets. | Fixed seed banks/origins/scales, shared horizons, attempted-trial budgets and bounded timeout windows implemented. Staged screening/promotion and memory budgets remain pending; final tests stay separate. |
| P3-03 | WIP | Complete trial provenance and safe study transfer/resume. | Common studies bind prepared data, task timing, recipes/objectives, seed banks and environment; resolved fixed/sampled settings are retained and incompatible scores rejected. External finite-score transfer and maximize summaries repaired. Raw-runner transfer/resume still lacks this full protocol binding. |
| P3-04 | Pending | Pruning during rollout and device-local scoring. | Stopped trials skip forecast computation; tensor/NumPy metric parity; no hidden device transfers in the supported fast path. |
| P3-05 | Done | Stable automatic readout selection with forced overrides. | Auto primal/dual/augmented-QR/SVD with unchanged alpha, guarded precision, structured diagnostics and forced overrides; native MPS or explicit CPU bridge. Independent SVD overflow/subnormal, integration and complete CI checks pass. |
| P3-06 | WIP | Tuning process and storage lifecycle on current Python. | Common tuning uses joblib/loky, finite windows, real-worker cancellation/error tests and caller-owned studies. Parent/path-transfer RDB cleanup preserves returned memory studies and primary errors. The older raw runner's worker/monitor lifecycle still needs migration. |

### Initial implementation verification

Runtime source: `b9c3fc8`, on **23 October 2026**. Subsequent source edits in
this increment correct docstrings only; executable ASTs are unchanged.
Hosted evidence below is pinned to `3cde4ce`; the final follow-up updates this
record and includes the new trajectory-tuning tests in the optional CI lane.
Required checks run again on the resulting PR head.

| Check | Evidence |
| --- | --- |
| Independent readout checks | **318 passed**, four unavailable-device cases skipped; Decimal and augmented-system oracles, unchanged alpha, precision boundaries and actual MPS native/CPU-bridge paths. |
| Common workflow integration | **157 passed**, covering presets, custom DAGs, trajectories, refits, PyTorch gradients and report propagation. Additional configuration regressions pass in the **36-test** engine suite. These focused gates overlap. |
| Joblib and scoring | **85 passed**, including real worker processes, notebook closures, error/cancellation cleanup and actual MPS scoring. All **25 backend tests** also pass with minimum joblib **1.5.3**. |
| Independent end-to-end review | **Nine passed**, including parallel structural recipes, equivalent-model SQLite resume, changed-objective rejection, inference-mode/autograd and in-place-layer isolation. |
| Complete local suite | **5,231 passed**, 231 skipped, 13 benchmark cases deselected; **93.15% coverage**. This includes installed dynamics and native MPS correctness checks. |
| Installed branch artifacts | Wheel and sdist contain all **188 Python modules**, byte-matching runtime source, with local coordination files excluded. Installed-wheel preset/custom tuning, checkpoint parity, gradients and named trajectory clocks pass. These are local builds, not a PyPI release. |
| Documentation | **Seven tuning examples** pass with exact shipping budgets, plus **six solver examples**, migration checks and native MPS checks. Strict MkDocs and link checks pass. |
| Complete hosted suites | [Locked CI](https://github.com/El3ssar/ResDAG/actions/runs/39864096488) and [latest compatibility](https://github.com/El3ssar/ResDAG/actions/runs/39864097736) pass all eight Python 3.11–3.14 suites. Each: **5,009 passed**, 453 skipped, 13 deselected; **90.43–90.47% coverage**, with the unchanged 90% floor. |
| Hosted integration, static checks and docs | All six original dynamics lanes pass **212 checks** each, seven skipped, with locked **TSDynamics 6.0.3** and latest **6.1.0 / Torch 2.14.1**. Strict docs in both modes, Ruff/Black, mypy (**188 modules**) and CPU regression gates pass. |

The optional dynamics lane now also selects the new tuning data/workflow
modules: their **55 tests** pass locally, including trajectory tuning. The
complete local suite and installed-wheel checks above also exercise the optional
integration. See [PR #461](https://github.com/El3ssar/ResDAG/pull/461) for the
latest checks after this CI-selection update.

The hosted CPU performance gate passes without changing its thresholds. Its
supplemental report contains ResDAG measurements, but the external comparator
packages are unavailable: Julia is installed without the required reservoir
package, and the Python comparator is skipped. That report is not evidence of
a successful cross-library comparison. Comparative benchmarking remains in
phase 5; CUDA throughput was not measured.

The conservative test selector now includes common model tests for HPO changes;
the earlier isolation assertion no longer described the dependency graph after
models gained `tune()`. All **34 selector regressions** pass with that expectation
updated; test-selection safety was not weakened.

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
complete-artifact format and extension policy before introducing the new save API.

| ID | Status | Work | Acceptance checks |
| --- | --- | --- | --- |
| P4-01 | Pending | Versioned complete artifact: graph, component configuration, weights, preprocessing, metadata and optional streaming state. | Use standard PyTorch serialization and existing schema/packaging conventions where suitable; load in a fresh process without a manually built DAG. |
| P4-02 | Pending | Generic `rd.load(...)`, version validation and custom-component reconstruction. | Presets, heterogeneous custom DAGs and registered components round-trip; custom Python code requirements are explicit; unknown versions/components fail with useful remedies. |
| P4-03 | Pending | Save diagram as SVG, PNG, PDF or DOT with clear labels and shapes. | Reuse existing Graphviz/DOT rendering; show branches, merges, named ports and delayed feedback; export does not mutate model state/modes or open a viewer unexpectedly. |
| P4-04 | Pending | Unified save dispatch and user documentation. | Supported suffixes are discoverable; unsupported formats fail clearly; diagram exports and model artifacts cannot be confused. |
| P4-05 | Pending | Fresh-process and cross-device artifact verification. | Forecast and preprocessing parity after reload; CPU map-location works; unsupported artifact versions fail clearly. |

## Phase 5: speed, memory and comparisons

### Known baseline performance limits

Running **all nine** existing non-GPU benchmark cases on this Apple Silicon
Mac with Torch 2.10.0, float64 CPU tensors and four intra-op threads exposed
two timing assumptions outside the current CI benchmark directories. The
seven other cases passed. An isolated import of unchanged `main` (`a24e7a1`)
reproduced both failures with the same environment and identical test/kernel
sources; branch/main timings below are separate runs, not a controlled speedup
comparison.

| Existing assertion | Completion branch | Unchanged main | Required by current test |
| --- | --- | --- | --- |
| CSR versus dense recurrent multiplication, size 2,000 / density 5% | Dense/CSR time ratio **0.35×** | **0.44×** | Greater than **1.2×** |
| Parallel advance with 64 versus eight patches | Large/small time ratio **8.44×** | **7.57×** | Less than **5×** |

These remain open performance work in P5-03/P5-04. Sparse crossover and batching
claims must be measured for the actual hardware, dtype and batch regime;
`sparse="auto"` uses fixed size/density heuristics rather than calibration.
Users can force `sparse=False` where the dense operator wins. Keep the current
thresholds visible until the policy and measurements are reviewed; widening
them alone would not improve performance.

| ID | Status | Work | Acceptance checks |
| --- | --- | --- | --- |
| P5-01 | Pending | Correct held-out comparison windows and common prediction alignment. | Shared data/splits and one-step timing oracles across adapters; historical tables are not treated as new measurements. |
| P5-02 | Pending | Controlled numerical and idiomatic equal-budget benchmark tracks. | Several systems, trajectories, seeds and scale regimes; record uncertainty, initialization/compile cost, fit/forecast time and memory. |
| P5-03 | Pending | Actual sparse/structured operators and appropriate matrix-free readouts. | Reuse Torch/SciPy operator/sparse/solver capabilities; memory scales as advertised; no mandatory hidden dense copy; save/load and numerical contracts pass. |
| P5-04 | Pending | Profile-driven batching and compilation improvements. | Reference/fast value, state and gradient parity; measured improvement in specified workloads; setup and steady state reported separately. |
| P5-05 | Pending | Publish speed–accuracy–memory tradeoffs against current competitors. | Reproducible commands, versions, hardware and equal search budgets; no universal unsupported superiority or GPU claim. |

Python 3.14 warns that `torch.jit.script_method` is deprecated. The phase-five
execution review must address that path using supported PyTorch tooling and
verify value, state and gradient behavior before replacing it.

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
| 2026-10-22 | Phase 2 foundations and trajectory release | Done | PR #459 merged as `d63905d`; v1.2.0 publication, full post-merge tests and documentation deployment passed. |
| 2026-10-22 | Phase 2 completion | Done | Common native ESN/Model workflow, all graph families, explicit feedback, fast/history execution, reversible per-port transforms, named trajectories and PyTorch composition implemented and documented. PR #460 merged as `4735542`; v2.0.0 published. Full post-merge tests, actual published-wheel install/smoke and versioned documentation deployment passed. |
| 2026-10-23 | Phase 3 initial implementation | WIP | Approved common tuner and automatic readout policy implemented and verified in [PR #461](https://github.com/El3ssar/ResDAG/pull/461): 5,231 local tests, 93.15% coverage; all eight locked/latest full hosted suites, optional integrations, strict docs, CPU gates and installed artifacts pass. Rollout pruning, staged promotion, memory budgets and raw-runner migration remain open. |

## Standard tools to reuse

- [PyTorch serialization](https://docs.pytorch.org/docs/stable/notes/serialization.html)
  for tensor/state storage, device mapping and established checkpoint conventions.
- [Graphviz](https://graphviz.readthedocs.io/en/stable/manual.html) for DOT layouts
  and diagram formats; ResDAG already has optional Graphviz integration.
- Existing `pytorch_symbolic` functional composition while graph requirements
  are evaluated; ordinary `nn.Module` and standard Torch tooling for execution.
- Optuna for studies, samplers, pruners and storage; Torch/SciPy for numerical
  solvers; TSDynamics for dynamical-system generation and analysis.
