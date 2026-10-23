---
title: Phase five measurements
description: Verified initialization changes, held-out forecast skill, compact storage and bounded CPU measurements.
---

# Phase five measurements

The continuation separates three questions: whether stored models keep their
predictions, how well newly initialized models forecast, and what the execution
changes cost or save. These measurements cover an unreleased implementation;
the [roadmap](roadmap.md#phase-5-speed-memory-and-comparisons) tracks remaining work.

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
