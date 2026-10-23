# Equal tuning time for vanilla reservoir forecasts

This is a separate track from `rc_bench.run`'s matched-configuration timings.
It asks which accuracy each library reaches with the same **600-second CPU
tuning budget**, not which library happened to receive the luckiest settings.
There are six independent studies: three libraries on each of Lorenz and
Rössler. Do not present a small smoke run as a performance comparison.
The verified-thread protocol is `equal-wall-time-v2`. Earlier `v1` runs without
verified backend limits are not valid references for an equal-thread headline.

## Common model and implementation choices

All candidates use Float64, one CPU worker/thread, a leaky `tanh` reservoir,
zero initial memory, and a linear ridge readout of the reservoir states.
There are no direct input features or readout intercepts.

| Control | Shared search |
|---|---|
| Reservoir size | 300, 400, 600, 1000 |
| Spectral radius | uniform 0.05–1.4 |
| Leak rate | uniform 0.1–1.0 |
| Input scale | log-uniform 0.01–1.0 |
| Reservoir bias scale | log-uniform 0.01–0.5 |
| Mean degree | 3, 6, 12 |
| Ridge penalty | log-uniform 1e-12–1e-3 |

Recurrent nonzero weights are standard Gaussian before spectral rescaling.
The matrix maps source column `j` to destination row `i`; edges are directed
and diagonal/self-connections are permitted. Nominal density is
`mean_degree / reservoir_size`. Input weights are dense independent uniform
draws in `[-input_scale, input_scale]`. Reservoir bias is a **random vector**
of uniform draws in `[-bias_scale, bias_scale]`, not a constant offset.
The ResDAG recipe explicitly selects its uniform bias initializer so the
existing salted bias seed is separate from the input-weight seed. This avoids
reusing the beginning of the input generator's stream; public model defaults
are unchanged. ReservoirPy spawns separate input/recurrent/bias streams; Julia
advances its native generator between the component draws.

Native random streams and sparse-mask construction differ. ResDAG's explicit
callback uses independent Bernoulli edges, reservoirpy uses its native SciPy
sparse initializer, and pinned Julia's initializer keeps a fixed nonzero count
per column before shuffling. This compares the same nominal model/distributions,
not identical realized weights. All sampling, normalization to spectral radius
and any native initializer work are charged to candidate construction/fit.
There are no external candidate retries or extra initialization budgets.

Storage is an implementation choice. ResDAG uses dense Torch execution for
this CPU recipe, reservoirpy native SciPy CSR. Julia's default `auto` policy
selects dense Matrix or sparse CSC with a **fixed synthetic, runtime-only**
pilot before tuning. Both layouts are warmed, then tested in ABBA order at
widths 300 and 600, using 401 sine samples, washout 20 and three 50-step
forecasts per fit. No system trajectory or accuracy score enters this pilot.
Its measurements, startup and selected layout remain in the result JSON.
`--julia-storage dense|sparse` forces a documented layout for reproduction.

Readout implementations also differ: ResDAG's `RidgeReadoutLayer`,
reservoirpy's `Ridge`, and Julia's `StandardRidge`. The requested ridge penalty
is shared, but native numerical policies differ. ResDAG's Cholesky recovery
may increase effective regularization or project a numerically indefinite
Gram matrix onto the positive-semidefinite cone. Warnings disclose recovery;
Python may deduplicate repeated warnings, and the current readout has no
structured per-fit effective-alpha report. These results therefore do not
promise an identical realized regularized objective. The harness itself does
not retry failed candidates, alter their requested penalty or choose another
solver. ReservoirPy uses a symmetric Gram solve; pinned Julia uses augmented QR.

## Data and held-out boundaries

The runner reuses `tuned_accuracy.py`'s raw TSDynamics datasets when supplied,
or generates them with that existing helper. `data.json` records system
parameters, integration settings, sampling interval and raw-data fingerprints.

| System | Observations admitted to fitting | Washout | dt | Lyapunov exponent |
|---|---:|---:|---:|---:|
| Lorenz | 10,001 | 500 | 0.02 | 0.9056 |
| Rössler | 8,501 | 500 | 0.145 | 0.0714 |

For `train_stop` observations, fitting uses inputs `[0:train_stop-1]` and
next-value targets `[1:train_stop]`, discarding the first `washout` pairs.
Scaling is learned **only** from admitted labels
`raw[washout+1:train_stop]`, using population standard deviation (`ddof=0`);
a constant signal's scale is one. That fixed transformation is shared by
all libraries and by training, validation and final tests.

Each forecast spans at least ten Lyapunov times. Three consecutive validation
windows follow training, and three disjoint test windows follow validation.
Before each origin, fresh recurrent memory is synchronized on the preceding
`washout` observed samples. The first output predicts the sample at the origin;
the model never receives that origin's truth or future samples during rollout.
Each seed is fitted **once**, then reused for its three origins.

Validation seeds are **7, 42, 12345**. The reserved final initialization seeds
are **301, 302, 303, 304, 305**, fixed before this experiment's final evaluation.
All six winners are written to `winners.json` before reading any final-test
slice. The SHA256 of that frozen file is checked throughout testing. The
reused raw trajectories and origins are held out from this study, but are
not historically unseen. There is one trajectory per system; this protocol
does not establish generalization across independent trajectories.

## Budget and failures

The CLI starts a fresh Python process with `VECLIB_MAXIMUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `OMP_NUM_THREADS`,
`BLIS_NUM_THREADS` and `JULIA_NUM_THREADS` set to one, overriding inherited
values before numerical imports. It holds the standard `threadpoolctl` limits
and verifies detected BLAS/OpenMP counters. Torch intra-op and inter-op limits
are both one. On Apple Accelerate, environment variables alone are insufficient:
the benchmark calls the supported macOS 15+ `BLASSetThreading` API and requires
`BLASGetThreading == BLAS_THREADING_SINGLE_THREADED` on its synchronous calling
thread. Unsupported or unverifiable backends stop the experiment before a
scientific headline is accepted. Settings are checked again after fits and
forecasts, including failed operations.

Julia separately sets and verifies its BLAS limit and Julia thread count at
worker startup. Reports record requested variables, observed native counters,
actual build/backend configuration and module origins. These verify API
configuration, not operating-system thread occupancy or every kernel's internal
scheduling. The synthetic storage pilot uses the same controls.

If an earlier run lacks these controls, preserve it as aborted or unverified;
start a fresh CLI process and new output directory. Do not resume its study or
reuse its scores as an equal-thread result. Only the newly capped study's own
validation scores may choose its winners.

The tuner is standard Optuna `study.optimize`, seeded TPE, `NopPruner`,
`n_jobs=1`, with persistent SQLite studies. It starts after imports and one
fixed synthetic warmup of each library. Julia stays in one standard subprocess
during each study, avoiding repeated interpreter startup. Startup/JIT warmup
is reported separately and uses no trajectory observations.
The per-track startup field measures adapter creation and the small synthetic
warmup. Global Python imports and thread-policy initialization are also outside
the search budget; that field is not a measurement of total process startup.
Any later size- or algorithm-specific compilation remains inside candidate work.

The budget includes candidate sampling, construction, three fits, nine
history/rollout evaluations, scoring, Python↔Julia file/pipe IPC and study
bookkeeping. A trial must finish all checks before the deadline to be eligible.
No shorter screening, failed-seed omission or early forecast pruning is used.
The objective is the median valid prediction time across the nine validation
forecasts at normalized RMS threshold **0.3**; the report also records **0.5**.
Equality with a threshold ends the uninterrupted valid prefix.

This is an eligibility deadline, **not an operating-system hard cap**. An
individual in-flight fit or forecast may finish after the deadline. The runner
checks between operations, excludes that candidate, stops further search work,
and records actual elapsed time, overrun and completed/partial work. Failed
attempts stay in the Optuna history with their errors and fit/forecast counts.
There are no hidden fallback candidates. If none finish in budget, there is
no winner.

Final evaluation attempts every reserved seed once. A returned forecast that
diverges still has a measurable valid prefix, with a divergence flag. A
build/fit/API failure is **not** assigned an invented forecast horizon: the
failed seed/origin stays an error, other attempts continue, the track has no
scientific aggregate, and the CLI exits nonzero. No final-test retuning or
automatic retry occurs.

Model storage/parameter-tree bytes, estimated retained feature-matrix bytes
and process peak RSS are recorded with each fit. RSS includes interpreter,
startup and earlier allocations; Python's process is shared across tracks,
whereas Julia has its own process. These are explicit memory observations,
not a clean per-trial peak-memory comparison or an operating-system cap.

## Reproduce

From the repository root, install the existing optional dependencies and the
pinned Julia project:

```sh
python -m pip install -e '.[hpo,dynamics]' -r benchmarks/requirements.txt
JULIA_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0 julia --project=benchmarks/julia \
  -e 'using Pkg; Pkg.instantiate(; allow_autoprecomp=false)'
```

The committed manifest was verified with **Julia 1.12.5** and
**ReservoirComputing.jl 0.12.23**. An isolated project/depot is supported:
copy `Project.toml` and `Manifest.toml`, set `RESDAG_JULIA_PROJECT` and
`JULIA_DEPOT_PATH`, instantiate that copy and verify both files' hashes remain
unchanged. The worker source continues to come from this repository.

Run the full approved matrix in a quiet, serialized CPU window:

```sh
python -m benchmarks.rc_bench.budget \
  --data-dir /tmp/resdag-comparison-data \
  --output-dir /tmp/resdag-comparison-results
```

This requests six ten-minute searches, followed by independent final tests;
startup, storage pilot and any recorded deadline overruns add wall time.
The runner saves dependency versions and hashes of the actual Python/Julia
source files, and rejects results if source changes during the experiment.

For a **plumbing smoke only**, use tiny widths, shorter windows and one trial:

```sh
python -m benchmarks.rc_bench.budget --smoke --budget-seconds 10 --max-trials 1 \
  --data-dir /tmp/resdag-comparison-data \
  --output-dir /tmp/resdag-comparison-smoke
```

`--libraries` or `--systems` selects a subset. Use a new output directory for
each run: the runner refuses to replace frozen winners. A trial-count cap is
an explicit smoke/debug override, not an equal-ten-minute headline result.

`results.json` retains every attempt, per-forecast threshold score, censoring
and failure status. `report.md` shows final medians and full ranges;
`*-test-predictions.npz` retains each returned final forecast. `*.sqlite3`
preserves the ordinary Optuna studies. The frozen `winners.json` contains no
test scores.

Fast clock/leakage tests run without Julia. Real Julia protocol checks are
opt-in after environment setup:

```sh
RESDAG_TEST_JULIA=1 JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest tests/test_benchmark_julia_protocol.py --no-cov -q
```

Those tests independently check NumPy augmented ridge, Gaussian/uniform bias
and input semantics, dense/CSC recurrence, fit-once resets, later origins,
the first unseen output, future-truth isolation and requested repetitions.
