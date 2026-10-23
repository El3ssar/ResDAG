---
title: Phase five measurements
description: Verified initialization changes, held-out forecast skill, compact storage and bounded CPU measurements.
---

# Phase five measurements

The continuation separates three questions: whether stored models keep their
predictions, how well newly initialized models forecast, and what the execution
changes cost or save. PR #465's implementation is released in
[v5.0.0](https://github.com/El3ssar/ResDAG/releases/tag/v5.0.0).
The [roadmap](roadmap.md#phase-5-speed-memory-and-comparisons) tracks the CPU
follow-up and remaining work; each measurement identifies its own source revision.

## Current all-PyTorch closeout

The latest checks cover composition inside a trainable PyTorch network,
parallel CPU forecasting, independent-trajectory accuracy and saved-experiment
replay. Model execution stays entirely in PyTorch. CPU teacher-forced
compilation and the private SciPy recurrence backend are not added.

An encoder → fitted reservoir → prediction head passes independent tensor and
finite-difference gradient checks, repeated optimizer updates, explicit-state
chunking and standard state-dictionary reload. Vanilla, OTT and named multi-head
graphs also pass full-graph forward/backward capture with PyTorch's AOT eager
backend. CPU and native MPS forward/backward checks reject host scalar reads
and cross-device tensor operations in the tested hot path. These establish
composition correctness; AOT eager capture and MPS checks do not measure CUDA
throughput. CUDA is unavailable on this host.

The acceptance tests found integration defects. CPU autocast could request
an unsupported low-precision CSR multiplication; that operation now uses the
stored weight precision on the same device, preserving gradients through the
surrounding mixed-precision network. Full-graph capture could fail while
inspecting the stored state's `grad_fn`; the compile-aware guard now detaches
only carried state under the existing default. Current-call gradients and
explicit cross-call gradient retention remain covered.
Hosted Torch 2.10 checks on Python 3.11–3.13 also exposed evaluation of a
typing union inside the preprocessing getter. Quoting that typing-only argument
preserves its eager identity operation and lets Dynamo capture the forward
graph; all four compiled cases and the full pipeline selection pass in an
isolated Python 3.13/Torch 2.10 reproduction.

The [experiment guide](../start/forecast-experiment.md) now saves resolved
settings, versions, split identities, artifact hashes and exact replay data.
Its verification command loads the saved models and reproduces predictions
and scores without integrating a system, tuning or fitting again. Recorded
timings remain observations from the original run.

### Parallel CPU forecast

The adopted single-frame path removes redundant sequence-wrapper work during
closed-loop forecasting in the built-in parallel model. It keeps the original
projection and recurrent arithmetic, warmup and state semantics. Hooks,
overridden methods, custom activations/readouts, autocast and non-CPU devices
use the ordinary compatible path.

Two fresh-process ABBA blocks compare frozen source snapshots differing only
in this parallel implementation. The four synthetic workloads use float64,
150 units per patch, 501 training observations, 100 washout/history observations
and a 129-step forecast; patch core width is four and halo width six. Torch and
native pools are verified at one thread. Construction is outside the clock;
public fit and history-plus-forecast calls are inside it.

All **64 timed fit/forecast pairs and 32 discarded warmup pairs** retain exact
fitted tensors, fit states, complete predictions and final forecast states
across variants. Each block contains four fresh processes in ABBA order, with
one warmup and two timed repetitions per case/process. The ratios below divide
the block's pooled baseline median by its candidate median; two blocks are
descriptive process observations, not confidence intervals.

| Batch / patches | Forecast ratio, block 1 / block 2 | Fit control ratio, block 1 / block 2 |
| --- | ---: | ---: |
| 1 / 8 | 1.098 / 1.055 | 0.981 / 1.004 |
| 1 / 64 | 1.041 / 1.078 | 0.973 / 1.015 |
| 4 / 8 | 1.096 / 1.096 | 0.956 / 0.982 |
| 4 / 64 | 1.023 / 1.041 | 0.929 / 0.996 |

Forecast elapsed time falls by about **2–9%** in these workloads. Fitting
does not use the new path: its unchanged control varies from about 1.5% faster
to 7.6% slower. Read these small forecast gains with that control variation in
mind. This is bounded parallel-forecast evidence, not a universal training
improvement or a cross-library speed ranking. All 96 workflows also preserve
their exactness hashes from the earlier run before the typing-only compiler
fix. The earlier parallel patch-scaling failure was not reproduced with the
current benchmark recipe; its threshold is unchanged.

[Download every sample, exactness hash, frozen source and replay script](../assets/data/phase-five-parallel-cpu-2026-10-23.json).

### Independent trajectories

Six previously selected recipes were frozen before generating ten new
trajectories: five initial conditions for each system. Each recipe/system is
refitted on each trajectory's own training prefix with three new reservoir
seeds, then evaluated at three nonoverlapping forecast origins. There is no
new hyperparameter search or winner selection. All **90 fits and 270 forecasts**
complete with finite predictions; no failed runs are excluded.

The four primary recipes come from the corrected-initialization capability
study. At the main 0.3 normalized-error threshold, the preregistered aggregate
is the **median of the five trajectory medians**; each trajectory median contains
nine seed/origin forecasts. The independent unit is a trajectory, not each of
the 45 nested forecasts. Full individual ranges include censored lower bounds.
Each recipe retains its own selected size and settings; this is not an
architecture ablation with matched hyperparameters.

| Primary recipe | Median at 0.3, Lyapunov times | Individual range at 0.3 | Forecasts reaching the test limit at 0.3 | Median at 0.5 |
| --- | ---: | ---: | ---: | ---: |
| Lorenz, friendly vanilla | 5.995 | 1.902–≥10.016 | 6 / 45 | 6.810 |
| Lorenz, OTT | ≥10.016 | 6.575–≥10.016 | 35 / 45 | ≥10.016 |
| Rössler, friendly vanilla | 6.326 | 2.505–≥10.001 | 7 / 45 | 7.144 |
| Rössler, OTT | ≥10.001 | 4.193–≥10.001 | 25 / 45 | ≥10.001 |

The two secondary state-only vanilla recipes from the common-family comparison
reach the ten-Lyapunov-time median limit on both systems. They use different
feature/solver policies and remain separate from the primary friendly recipes.
This demonstrates sensitivity to the chosen recipe; it does not replace the
negative friendly-vanilla results. The 7–8-time typical-performance target is
still unmet for those two frozen friendly recipes. OTT meets the median target
in this study, although individual forecasts can be shorter.

These are robustness checks of frozen **recipes refitted on independent runs**,
not zero-shot transfer of one trained reservoir to a new initial condition.
They do not compare competitor accuracy on this new bank or establish that an
implementation change caused the vanilla shortfall. Previously inspected
forecast banks remain distinct from these new observations.

A paired regression for the initial implementation separately repeats the same
90 fits and 270 forecasts on clean baseline `eb70bf11`, using the sealed
observations and initial weights.
Every fitted-tensor fingerprint, complete prediction, final state and score
matches exactly. This second pass adds no tuning or test-based selection.

The preregistered runtime was frozen before the release-metadata rebase. Its
version literal is 5.0.0. Subsequent changes update that literal to 5.0.1,
tighten a parallel-only custom-readout guard unused by these six recipes, and
quote the preprocessing getter's typing-only cast for older compiler support.
The public record retains the original source and every subsequent source
bridge; it does not relabel the frozen capture as a later checkout.

On the final corrected runtime, a separate replay loads all **90 original
fitted checkpoints** and reproduces all **270 forecasts** exactly, including
complete prediction bytes, final states and both scores. Every fitted tensor
fingerprint also matches; the common graph's nonpersistent post-fit state is
restored by the original teacher-forced input prefix. This replay performs no
readout solve, fitting, integration or tuning. Two stopped restore-setup attempts
are retained separately; neither found a forecast discrepancy or contributes
to the original study's scores.

[Download all 270 scores, lossless error traces, trajectory summaries, paired regression and reproduction inputs](../assets/data/phase-five-independent-trajectories-2026-10-23.json).

### Closeout verification

The implementation is in [PR #467](https://github.com/El3ssar/ResDAG/pull/467).
The initial `3493889` checks identified the older-Python/Torch 2.10 graph-capture
failure described above; final hosted acceptance follows its quoted-cast fix.
These package checks cover the corrected candidate source; its wheel
retains the base version 5.0.1 and is not the previously published 5.0.1 artifact.
The final tracker update changes prose only.

| Check | Result |
| --- | --- |
| Complete local suite before the typing-only follow-up | **6,509 passed**, 245 skipped, 13 benchmark cases deselected; **93.86% coverage** against the unchanged 90% floor. Final corrected-source replay is being recorded. Compiler acceptance cases isolate their caches rather than changing Torch's global recompile limit. |
| Configured local CPU gates | **Four passed**, four unavailable-device skips, one deselected; thresholds unchanged. |
| Clean candidate package | **110 installed-wheel checks passed**, three unavailable-CUDA skips, plus all eight common-workflow guide blocks. All **205 runtime modules**, the typing marker, **209 RECORD entries** and actual installed import origins verify against the frozen source and sdist. |
| Minimum optional dependency | **13 experiment checks pass** with an actual isolated **TSDynamics 6.0.2** install, including fresh-process exact trajectory replay. |
| Older Python / minimum Torch | Python **3.13.16 / Torch 2.10.0** reproduces all four original graph-capture failures; the corrected source passes all **34 pipeline checks**, including those four cases, with three unavailable-CUDA skips. |
| Static and documentation | Ruff/Black and whole-package mypy (**205 modules**) pass. Strict MkDocs passes; all four changed pages were visually inspected, and all three new downloadable records match their served bytes. |

## CPU follow-up

PR #466 is merged as `3b34005` and released in
[v5.0.1](https://github.com/El3ssar/ResDAG/releases/tag/v5.0.1).
All four post-merge full suites, three TSDynamics lanes, the published-wheel
fit/forecast/save/load replay and live documentation deployment pass.

The adopted CPU changes reuse AutoReadout residual diagnostics when its admitted
Gram solution is unchanged, while recomputing them after a fallback replaces the
coefficients. Sparse cycle normalization uses an analytic spectrum, fixing
iterative eigensolver failures without dense allocation. The CSR fixes also
handle complex value views and explicitly stored zero weights. Automatic storage,
solver and compilation choices retain their established settings.

The experimental timestep-view and driver-slicing rewrites in `dc695a6` were
**rejected**: the measurements did not establish a CPU benefit. The original
sequence-loop and driver slicing are retained. Records against `dc695a6` below
remain historical candidate evidence; they do not measure the final adopted
implementation's whole-workflow speed.

### Retained-runtime accuracy regression

All **30 previously scored ResDAG forecasts** from the equal-time comparison
were replayed on the final retained runtime after restoring the original slicing.
Every complete Lorenz/Rössler forecast is byte-identical, using the same frozen
winners, five seeds and three origins. The public record identifies the source
as `final-runtime-manifest` and includes exact hashes for all **205 runtime
modules**, plus the manifest hash. This preserves those measured scores. It is
regression evidence from an already inspected bank; no new tuning, timing or
independent-trajectory claim is made.

[Download the complete replay](../assets/data/phase-five-cpu-accuracy-2026-10-23.json).

### Rejected-candidate public workflow timing

Two fresh-process ABBA blocks compare `18f78c83` with `dc695a6` on this Apple M5
with 24 GiB RAM, Torch 2.10 and float64 CPU tensors. Each revision explicitly
loads the same saved weights and synthetic observations. Torch and native
thread policies are verified at one. Seven workflow/batch cases and six
dense/CSR cases each retain seven timed repetitions per process after warmup:
**728 timed executions of each operation** (fit, history plus forecast and
reservoir forward), plus **104 discarded warmups of each operation**. Every
fitted-model, complete-prediction and final-state hash agrees
exactly between revisions within its case.

**The overall workflow speed effect is inconclusive.** During the first matrix,
latency increased sharply within a candidate process, then remained high in
the following baseline process. Dependency versions, source/import identities
and observed thread settings remained the same. The cause of this change in
execution conditions was not recorded. Every run remains included; medians
alone cannot establish a small implementation speedup or slowdown here.

Ratios below are before/after process-median times. Brackets retain the full
range of four nearest AB/BA ratios; these are descriptive observations, not
confidence intervals.

| Workflow | Fit ratio | History + forecast ratio |
| --- | ---: | ---: |
| Dense classic, 400 / batch 1 | 1.091 [0.994–10.407] | 1.001 [0.941–9.639] |
| Dense OTT, 400 / batch 1 | 1.009 [0.992–9.301] | 0.955 [0.904–8.200] |
| Sparse cycle, 400 / batch 1 | 0.998 [0.948–6.209] | 0.971 [0.800–8.451] |
| Sparse ER, 400 / batch 1 | 1.041 [0.935–8.671] | 1.086 [0.896–7.735] |
| Dense classic, 1,000 / batch 1 | 1.040 [0.758–24.056] | 1.015 [0.815–25.358] |
| Dense classic, 1,000 / batch 4 | 1.006 [0.808–1.133] | 0.982 [0.743–0.999] |
| Dense classic, 1,000 / batch 16 | 1.029 [1.014–1.594] | 1.000 [0.921–1.570] |

The storage profile uses the same ER matrices in dense and CSR form at
400/1,000/2,000 units, expected degree six and batch one. Both representations
remain exact across revisions; their cross-storage complete forecasts differ
by at most **5.14e-10** on these synthetic observations. Storage timings also
show variability, and pairs run in declared order rather than a separately
balanced crossover experiment. They do not justify a new automatic storage rule.

Two earlier attempts stopped during the first baseline worker, before a
complete comparison: one repeated normalization of placeholder weights that
would be replaced, and another compared dense negative-zero bits with omitted
CSR zeros. The harness now loads saved weights before checking their logical
identity, normalizing zero signs only for that initial recurrent comparison.
Fitted weights, predictions, states and stored artifact bytes retain exact
checks. The aborted attempts remain in the audit archive and contribute no
timings to the complete diagnostic reports.

[Download all workflow samples and hashes](../assets/data/phase-five-cpu-workflows-2026-10-23.json),
[storage samples and hashes](../assets/data/phase-five-cpu-storage-2026-10-23.json).
The [CPU protocol](https://github.com/El3ssar/ResDAG/blob/main/benchmarks/CPU_PROTOCOL.md)
documents the reusable command, all dimensions, preprocessing, untimed work and
process-level statistics.

### Within-process loop attribution

A separate diagnostic interleaves the original and candidate teacher-forced
loop functions inside each of two fresh processes using one common runtime
(`6b3b108`). It changes only `BaseReservoirLayer.forward_stateless`; it does
not compare isolated revisions or separately measure the driver-slicing rewrite.
Each case/process includes one warmup per function, then seven ABBA rounds.
All fitted-model, complete-prediction and final-state hashes match.

The following medians and full ranges use all **28 within-process paired
ratios per case** across both processes. These repeated pairs share their
process environment and are not independent process observations or confidence
intervals.

| Workflow | Fit ratio | History + forecast ratio |
| --- | ---: | ---: |
| Dense classic, 400 / batch 1 | 1.032 [0.390–1.451] | 0.893 [0.490–1.882] |
| Dense classic, 1,000 / batch 4 | 1.003 [0.445–1.730] | 0.945 [0.313–1.511] |
| Dense ER, 2,000 / batch 1 | 0.996 [0.793–1.642] | 1.014 [0.671–1.365] |

The variation remains broad and does not establish a consistent loop speedup.
Torch 2.10's `ATen/TensorIndexing.h` already returns the existing tensor for
full-axis slices during ordinary eager multidimensional indexing. The rewrite
can reduce index parsing, but its proposed removal of extra full-axis tensor
views does not describe that eager path. Both experimental slicing rewrites
were therefore rejected; their complete measurements remain available.

[Download the within-process diagnostic](../assets/data/phase-five-cpu-interleaved-2026-10-23.json).

### Adopted readout reuse and exploratory compilation

A bounded AutoReadout probe fits two well-conditioned Gaussian problems with
5,000 samples, 128/400 features, three outputs, intercept and penalty 10. Two
fresh-process ABBA blocks, seven timed fits after warmup, gave paired median
ratios **1.154 [1.025–1.514]** and **1.193 [1.128–1.389]**. All coefficients,
intercepts and complete diagnostic reports match. This measures the admitted
primal path's duplicate-diagnostic removal; it does not measure a complete
forecast or fallback QR/SVD speed. The probed AutoReadout code is byte-identical
to the retained implementation. The public workflow matrix above uses Ridge.

An exploratory 400-unit float64 classic CPU compile probe observed a **45.54 s**
first compiled request. Warm 257-step forecasts took **37.99 ms** versus
**44.79 ms** eager; 1,025-step forecasts took **106.82 ms** versus **157.40 ms**.
Both complete predictions matched exactly. The second horizon reused the same
64-step compiled chunk cache; it is not a second cold start. One cold process
and these synthetic signals do not establish a general compiler policy or
chaotic-forecast accuracy. CPU teacher-forced sequences remain eager.

These two probes have narrower provenance than the archived workflow harness.
The readout probe isolates unchanged readout dependencies. Compilation used the
then-live candidate checkout before the slicing rewrites were restored, without
an isolated overall source manifest. Its timings remain historical candidate
observations and do not establish final adopted-runtime speed. Compilation
remains opt-in. [Readout records](../assets/data/phase-five-cpu-auto-2026-10-23.json)
and [compile records](../assets/data/phase-five-cpu-compile-2026-10-23.json) retain
their raw samples, diagnostics, native settings and warnings.

## CPU recurrence investigation

After merging PR #466 as `3b34005`, we profiled the complete public fit and
history-plus-forecast workflows before choosing the next optimization. These
experiments use fresh synthetic development data and fixed saved weights;
they are neither another accuracy search nor a new comparison with competitors.

On this Apple M5, recurrence is the main cost. In a separate instrumented
1,000-unit, batch-one dense fit, recurrent multiplication accounts for 54.7%
of Torch self CPU time and tanh for 20.1%. The successful Ridge solve costs
about 26 ms, versus about 25 ms for a direct SciPy diagnostic without ResDAG's
guards. Transactional model copying accounts for roughly 1–1.6 ms in cProfile.
Removing numerical guards or failed-fit isolation would miss the main cost.
Profiler times identify work; the latency measurements below come from
uninstrumented public calls.

### Optional sparse CPU execution candidate

A private prototype substitutes SciPy CSR multiplication for Torch CSR
multiplication, leaving projection, activation, leak, scaling, training and
forecast commands unchanged. Both processes use the same 1,000-unit matrix
with 6,070 stored connections, float64, batch one, 10,001 observations,
500 washout/history steps and a 553-step forecast. The fixed readout is Ridge,
not AutoReadout. Construction is outside the clock; fit includes preprocessing,
the complete training prefix and transactional commit. Forecast includes history
replay. Lazy SciPy operator construction and every Tensor/NumPy bridge are
inside the measured calls.

Each fresh process runs four ABBA rounds after one discarded warmup per mode.
The ratios use the eight nearest AB/BA pairs within that process. Brackets
show their full observed range, not confidence intervals.

| Process | Sparse fit, Torch → SciPy | Paired fit ratio | History + forecast, Torch → SciPy | Paired forecast ratio |
| --- | ---: | ---: | ---: | ---: |
| 1 | 531.84 → 171.67 ms | 3.108 [3.061–3.125] | 60.01 → 21.23 ms | 2.822 [2.776–2.885] |
| 2 | 528.04 → 173.07 ms | 3.053 [3.005–3.081] | 59.21 → 21.38 ms | 2.764 [2.138–2.873] |

All 32 timed workflows have byte-identical fitted coefficients, complete
predictions and final states across the two modes and processes. Warmups ran
but were not fingerprinted separately. These are two processes on one host,
not independent machines or evidence about Lorenz/Rössler forecast skill.
Native thread limits are verified at one; power/thermal and process CPU/wall
receipts are retained. API thread limits do not measure operating-system
thread occupancy or CPU frequency.

**This backend was not adopted.** The maintainer chose to keep model execution
entirely in PyTorch. The private diagnostic remains useful cost attribution;
it does not establish production cache ownership, live weight/index updates,
serialization, device changes, gradients, PyTorch transforms, autocast or
compilation behavior. The current storage and execution defaults remain
unchanged.

### Dense batch candidate and rejected alternatives

At batch four, replacing one matrix multiplication with four matrix-vector
multiplications improves public fit by 1.775×/1.798× and forecast by
1.929×/1.956× in the two processes. However, the changed reduction order produces
substantially different fitted coefficients and closed-loop predictions on
this poorly conditioned synthetic fixture. This candidate is not adopted.
Those differences do not score physical-system accuracy, but they preclude
calling the change numerically identical.

At batch one, matrix-vector multiplication and transposing the dense equation
provide no measured primitive benefit. Reusing tanh output storage also gives
no meaningful gain in the bounded allocation probe. Torch CSR is slower than
dense in the initial 1,000-unit public profile; compressed storage alone is
not a speed guarantee. Sequential profile rows are not a paired storage
comparison.

[Download raw timings, hashes, source manifests and diagnostic scripts](../assets/data/phase-five-cpu-recurrence-2026-10-23.json).
The earlier cross-library CPU gap remains open until an adopted implementation
is measured on the same comparison protocol. Extending CPU teacher-forced
compilation is deferred: its startup and maintenance costs are not justified
by established workflow gains. The next work prioritizes standard PyTorch
fit/forecast performance and composition inside larger GPU models, preserving
gradients and device placement. CPU measurements do not establish GPU throughput.

## Forecast accuracy after independent initialization

Correcting the initialization streams intentionally changes new models built
with old seeds. We ran four fresh tuning studies on the corrected implementation
(`dcedbba`; runtime unchanged in `1dc8cfb`). Each attempted 24 configurations,
using three initialization seeds and three validation starts. All four winners
were frozen before evaluation on reserved seeds **401–405**, with three starts
per seed. No tuning followed those test results.

The table reports the median and full range of **15 individual forecasts**.
The primary limit is per-step normalized RMS error **0.3**, using training-only
component standard deviations; **0.5** is also reported. Horizons are in
Lyapunov times. **≥** means the forecast reached the scoring limit, so its true
valid horizon can be longer.

| System / family | Median at 0.3 | Range at 0.3 | Reached limit | Median at 0.5 |
| --- | ---: | --- | ---: | ---: |
| Lorenz / vanilla | ≥10.016 | 4.474–≥10.016 | 9/15 | ≥10.016 |
| Lorenz / OTT | ≥10.016 | ≥10.016 throughout | 15/15 | ≥10.016 |
| Rössler / vanilla | **4.203** | 0.445–6.346 | 0/15 | 4.234 |
| Rössler / OTT | ≥10.001 | 8.438–≥10.001 | 9/15 | ≥10.001 |

**Vanilla Rössler does not meet the 7–8 target in this experiment.** Its winning
configuration was the predeclared known recipe: validation gave 7.485, while
reserved testing gave 4.203. Relaxing the error limit to 0.5 does not resolve
that shortfall. The unoptimized baseline gave 0.259. More robust validation and
further model/search work remain necessary; we did not choose another winner
after seeing the test results.

The 15 forecasts share reservoir seeds and one trajectory per system. They are
not 15 independent dynamical-system runs. For example, the Lorenz vanilla
winner's five per-seed medians range from 7.535 to ≥10.016, and their median is
8.857; its pooled median above is ≥10.016. Raw trajectories and chronological
origins were historically known, while the reserved initialization bank was
fresh. These results do not establish generalization across independent
trajectories or superiority of default tuning.

| Study | Complete configurations | Failed configurations | Attempts |
| --- | ---: | ---: | ---: |
| Lorenz / vanilla | 21 | 3 | 24 |
| Lorenz / OTT | 23 | 1 | 24 |
| Rössler / vanilla | 15 | 9 | 24 |
| Rössler / OTT | 11 | 13 | 24 |

Every complete configuration received all nine full validation forecasts.
Failures remain recorded; there was no pruning or replacement budget. This
capability check uses explicit classic/OTT factories, including their direct
input features and readout intercept, with a float64 CPU SVD readout. Its
elapsed time includes concurrent correctness testing and cannot
support a speed comparison. See [the design](phase-five-design.md) for the
scoring equations and chronological windows.

[Download all capability scores, settings and provenance](../assets/data/phase-five-capability-2026-10-23.json).

### Run the capability protocol yourself

From a repository checkout, install the optional dependencies and run the
single-revision command. Generating TSDynamics observations requires Python
3.12 or newer:

```sh
python -m pip install -e ".[hpo,dynamics]"
python -m pip install -r benchmarks/requirements.txt
python benchmarks/tuned_accuracy.py --mode capability --after HEAD \
  --test-seeds 401 402 403 404 405 --n-trials 24 \
  --artifacts-dir /tmp/resdag-capability --output capability.json
```

It archives the **committed** source, generates the observations, records their
versions and hashes, and freezes all four winners before evaluation. Use fresh
output paths. To reuse saved observations exactly, add
`--data-dir /path/to/saved-data`; that directory contains `data.json`,
`lorenz.npy` and `rossler.npy`. For a bounded plumbing check, use
`--quick --n-trials 1` and omit `--test-seeds`; quick mode uses the separate
smoke seed 101.

This command repeats the protocol. Reusing seeds 401–405 reproduces an existing
bank; it does not create a fresh test. A new scientific experiment needs a
previously unused bank chosen before tuning. The measurements above used an
archived external wrapper and its frozen workers; this later supported command
was not their original driver. Backend and dependency differences can change
numerical results.

## Stored models and regression anchors

Four trained 4.1.0 checkpoints were loaded in both the published implementation
and the corrected runtime. All **12 history-conditioned forecasts** and their
fitted tensor fingerprints were byte-identical. This checks materialized model
behavior; it does not imply that newly constructed models keep the old draws.
Current-memory and RNG round trips are also covered by the artifact test suite.

Two existing CI quality tests had implicitly identified their reservoir weights
only by a seed. Their strict bounds failed after those seeds generated different
weights. The tests now use explicit public initializer callbacks to reproduce
their historical weights. All **18 reservoir tensors** match the pinned old
implementation byte for byte. Seeds, target data and accuracy limits are
unchanged; the FORCE suite and parallel forecast floor pass. Fresh-default skill
is measured separately above.

## Compact storage and paired CPU execution

For a fixed 400-unit model, compressed-only storage reduced the sparse cycle's
retained tensor storage from **1,321,042 to 37,842 bytes**, and the sparse random
example from **1,351,978 to 58,466 bytes**. These counts deduplicate tensor storage;
they are not peak process-memory measurements. Read
[memory and sparse models](../workflows/sparse-reservoirs.md) for temporary
initialization, feature-history, solver and device boundaries.

Two independent fresh-process ABBA runs compared merged `041db61` with
`1dc8cfb`, explicitly loading the **same baseline weights** into both versions.
Each used 1,501 synthetic training samples, washout 100 and 256 forecast steps.
Each setting retains four processes and 28 timed repetitions per version after
warmup. Every prediction digest matched across all processes. The second run
was prompted by one apparent 10% slowdown; both runs remain in the pooled result.

Ratios below are baseline time divided by new time. Above 1 is faster; below 1
is slower. These are workload-specific medians, not a universal speed guarantee.

| Model | Torch threads | Fit ratio | Forecast ratio |
| --- | ---: | ---: | ---: |
| Dense vanilla | 1 | 0.987× | 0.969× |
| Dense OTT | 1 | 0.989× | 0.950× |
| Sparse cycle | 1 | 1.112× | 1.139× |
| Sparse random | 1 | 0.978× | 0.997× |
| Dense vanilla | 4 | 1.017× | 1.007× |
| Dense OTT | 4 | 1.025× | 1.011× |
| Sparse cycle | 4 | 1.133× | 1.142× |
| Sparse random | 4 | 1.028× | 1.018× |

The one-thread dense forecast medians are roughly 3–5% slower in this sample.
The configured CPU gates pass with unchanged thresholds, but that does not erase
these measured costs. Broader batching and compilation profiling remain open.
The paired experiment controls Torch threads; native Accelerate uses its default
policy in both versions. It is separate from the verified single-thread
equal-budget protocol below. Initialization is excluded from fit/forecast timing.

## Equal tuning time across libraries

The first full comparison completed on **23 October 2026**, using source
`1dc8cfb`. ResDAG, ReservoirPy and ReservoirComputing.jl each received **600
tuning seconds per system**, followed by the same reserved seeds **301–305**
and three starts. All six winners froze before final evaluation. Independent
audits checked every forecast, scale, cutoff and winning trial. All **90 final
forecasts** completed without API errors or nonfinite predictions.

Each row summarizes 15 forecasts at the primary 0.3 limit. **≥** marks the
scoring ceiling; it is not an exact prediction limit.

| System / library | Median LT | Full range LT | Reached limit |
| --- | ---: | --- | ---: |
| Lorenz / ResDAG | ≥10.016 | 8.893–≥10.016 | 14/15 |
| Lorenz / ReservoirPy | ≥10.016 | 8.911–≥10.016 | 14/15 |
| Lorenz / ReservoirComputing.jl | ≥10.016 | 8.422–≥10.016 | 10/15 |
| Rössler / ResDAG | **9.628** | 7.609–≥10.001 | 6/15 |
| Rössler / ReservoirPy | 9.639 | 7.599–≥10.001 | 7/15 |
| Rössler / ReservoirComputing.jl | 9.701 | 7.609–≥10.001 | 7/15 |

All six medians reach the scoring ceiling at the secondary **0.5** limit.
ResDAG meets the 7–8 target in this common vanilla experiment, with accuracy
close to both other libraries. The classic input-skip/SVD experiment above
still has its separate Rössler shortfall. Different families, initialization
distributions, settings and history windows prevent attributing the difference
to one feature.

All selected reservoirs have **1,000 units**, with independently tuned settings.
The following medians describe those selected models on this **Apple M5,
24 GiB, macOS 27.0.1** CPU. They are not matched-weight repeated timing trials.

| System / library | Build + fit seconds | Synchronize + forecast + score seconds |
| --- | ---: | ---: |
| Lorenz / ResDAG | 1.7965 | 0.1280 |
| Lorenz / ReservoirPy | 0.3839 | 0.0390 |
| Lorenz / ReservoirComputing.jl | 2.0427 | 0.0392 |
| Rössler / ResDAG | 1.1705 | 0.1321 |
| Rössler / ReservoirPy | 0.2690 | 0.0412 |
| Rössler / ReservoirComputing.jl | 0.9860 | 0.0262 |

**ResDAG is slower in these selected CPU workflows.** Its comparable accuracy
does not establish a speed advantage. Batching, compiled execution and the
single-sequence hot path remain profiling priorities.

The native thread APIs reported one for Torch, BLAS and Julia, with fresh CLI
environment controls. This verifies the requested API settings, not operating
system thread occupancy. The fixed synthetic Julia storage pilot selected CSC
before accuracy tuning. Python model-memory fields count tensors/arrays;
Julia's parameter-tree counts have a different scope. Process peak RSS includes
startup and earlier work, so these fields do not support a common memory ranking.

Each track's final attempted configuration crossed the cutoff and was excluded;
eligible validation work completed within 600 seconds. Actual track durations
were **600.024–601.188 seconds**, and the partially completed fits/forecasts
remain recorded. There were no other trial failures or automatic retries.
Eligible complete counts were **300 / 801 / 207** on Lorenz and
**310 / 881 / 351** on Rössler, in ResDAG / ReservoirPy / Julia order.
Different sampled configurations make these counts unsuitable as standalone
throughput ratios.

Native solver safeguards also matter: ResDAG recovery can project the Gram
matrix or increase effective regularization. Logged final evaluation includes
one recovery that explicitly kept its requested penalty, plus condition
diagnostics; ReservoirPy also reports ill-conditioning warnings. Warnings are
deduplicated and do not provide complete per-fit effective-penalty accounting.
The same requested search range does not imply identical numerical objectives.

[Download all 90 scores, selected settings, timings and provenance](../assets/data/phase-five-equal-time-2026-10-23.json).

The [protocol](../workflows/benchmarks.md#equal-tuning-time) describes native
mask/RNG/solver differences, one-worker float64 execution, actual native thread
controls, deadlines, failures, startup and memory accounting. This common
state-only vanilla family is distinct from the classic/OTT capability check
above. An equal-time study compares selected models; matched-weight execution
checks measure implementation costs separately. This is one timed study on one
historically known trajectory per system, not evidence of universal superiority
or generalization to independent trajectories.

## Verification

| Check on `1dc8cfb` | Result |
| --- | --- |
| Complete local suite | **6,267 passed**, 242 skipped, 13 deselected; **93.83% coverage**. |
| Installed wheel | **334 passed**, two unavailable-CUDA skips; all **205 Python modules plus the typing marker** match source, wheel, sdist and installed package. |
| Winning artifacts | All four seed-401 checkpoints load and forecast in the fresh installed environment without reconstructing their DAGs. |
| Full hosted suites | All eight locked/latest Python 3.11–3.14 lanes: **6,018 passed**, 491 skipped, 13 deselected; **91.59–91.66% coverage**. |
| Installed TSDynamics | All six locked/latest Python 3.12–3.14 lanes: **315 passed**, seven skipped. |
| Benchmark contracts | **76 Python checks** and **18 real Julia checks** pass. |
| CPU timing gates | **Four passed**, four unavailable-device skips, one deselected; thresholds unchanged. |
| Static and docs | Ruff/Black, mypy on **205 modules**, lock validation, actionlint and both strict documentation builds pass. |

Hosted runs: [locked](https://github.com/El3ssar/ResDAG/actions/runs/39971476477),
[latest compatibility](https://github.com/El3ssar/ResDAG/actions/runs/39971485792).
Native MPS correctness is covered locally. No CUDA throughput measurement is
claimed.

The reporting follow-up adds the public evidence records, their scientific
summary guard and the supported capability command. Runtime changes after
`1dc8cfb` are limited to eleven factory docstrings; stripping docstrings gives
identical runtime syntax trees. The complete follow-up suite passes **6,328
tests** at **93.83% coverage**, with 242 skips and 13 deselections. All **145 focused command,
scientific-record and selector checks** pass. A real quick command run completes
four studies and ten forecasts, verifying data/source/artifact hashes, frozen
winners, dependencies and native thread settings. A newly built wheel again
passes **334 installed checks**, and all runtime bytes match source, sdist and
installed package. Strict docs and static checks pass. The final hosted checks
are recorded on [PR #465](https://github.com/El3ssar/ResDAG/pull/465).
