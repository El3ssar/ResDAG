# Reservoir-computing library benchmark

The older cross-library tables in this directory are historical. New runs use the explicit
`heldout-v1` protocol below; old JSON files retain their historical semantics.
Correcting the harness does not validate old accuracy rankings or speed ratios.
The [current equal-time results](../docs/project/phase-five-results.md#equal-tuning-time-across-libraries)
report the completed six-track study. See [phase 5 of the roadmap](../docs/project/roadmap.md)
for remaining work.

For the new **equal tuning time** track, see [the complete protocol](BUDGET_PROTOCOL.md).
It compares vanilla ESNs in all three libraries on Lorenz and Rössler with
600 seconds of CPU tuning per library/system, then five reserved seeds and
three forecast starts. It records late and failed attempts and freezes all
winners before test access. This is separate from the matched-setting timing
sweeps below. Run it from the repository root:

```sh
python -m benchmarks.rc_bench.budget \
  --data-dir /tmp/resdag-comparison-data \
  --output-dir /tmp/resdag-comparison-results
```

For CPU implementation changes, use the separate
[fixed-weight workflow protocol](CPU_PROTOCOL.md). It compares isolated source
revisions with the same saved weights and synthetic data, times public
`Model.fit` and `Model.forecast`, and checks full prediction/state hashes.
Its optional storage profile compares dense and CSR representations of the
same matrices. These measurements do not run a new accuracy search.

An extensible, re-runnable harness that compares resdag against other reservoir
-computing libraries on **matched nominal ESN settings and identical data**, then
renders speedup tables (relative to reservoirpy).

Currently compared:

| Library | Backend | Device(s) |
|---|---|---|
| **resdag** | PyTorch | CPU (float64) + GPU (float32) |
| [reservoirpy](https://github.com/reservoirpy/reservoirpy) | NumPy / SciPy | CPU |
| [ReservoirComputing.jl](https://github.com/SciML/ReservoirComputing.jl) | Julia | CPU |

## What it measures

Four contexts, all on a shared Lorenz-63 trajectory (one-step-ahead training,
then closed-loop forecasting):

1. **Training vs reservoir size** — fit the ridge readout while growing the reservoir.
2. **Training on lots of data** — fix the reservoir, grow the training sequence.
3. **Long-horizon autoregressive forecasting** — generate ever-longer closed-loop forecasts.
4. **Forecast precision** — contiguous valid prediction time, reported in samples and Lyapunov times.

## Held-out protocol v1

For training length `T`, washout/synchronization length `W` and horizon `H`,
all bounds below are zero-based and half-open:

- Training inputs are `raw[0:T]`; next-sample targets are `raw[1:T+1]`.
  The readout ignores the first `W` input/target pairs.
- Mean and population standard deviation come only from supervised training
  targets, `raw[W+1:T+1]`, matching the common tuning workflow. The transform
  is applied once per point and every adapter receives an independent copy of
  the same transformed bytes. Constant training channels use scale one.
- Recurrent state is reset before each forecast. Synchronization observes
  `raw[T-W+1:T+1]`, ending at the last admitted training target.
- The first emitted prediction is for index `T+1`; truth is
  `raw[T+1:T+1+H]`. No test sample is supplied as feedback.
- Valid prediction time ends at the first normalized RMS error **at or above
  0.3**, using the fixed training scales. Threshold **0.5** is reported as a
  secondary diagnostic in JSON. A non-finite prediction ends the valid prefix
  and sets `diverged`; an accurate complete horizon is marked censored (a lower
  bound, rather than a measured failure time).

Every new result records `meta.protocol="heldout-v1"`, exact point bounds,
normalization statistics, dependency availability and both thresholds. These
single-seed nominal-setting sweeps are protocol smoke checks, separate from the completed
multi-seed/origin, equal-search-budget comparison or proof of a 7–8 Lyapunov-time
median. Adapters use dense input connections, zero reservoir bias and no readout
intercept; matrices
are independently sampled and solver numerics differ across libraries even
when nominal hyperparameters match.

Each cell is the median wall-clock time over several repeats (a warmup iteration
is discarded; GPU work is `cuda.synchronize`'d before the clock stops). A
short-horizon forecast RMSE is recorded as a cross-library sanity check that the
models are doing comparable work.

## Setup

```bash
# from the repo root, in the project venv
uv pip install -e ".[dev]"                  # resdag + torch
uv pip install -r benchmarks/requirements.txt   # reservoirpy
# GPU: install a CUDA torch build for the resdag-gpu rows

# Julia comparison (optional — skipped automatically if Julia is absent)
julia --project=benchmarks/julia -e 'using Pkg; Pkg.instantiate()'
```

## Run

```bash
cd benchmarks

# full matrix -> results/latest.json
python -m rc_bench.run

# quick smoke test
python -m rc_bench.run --quick

# a subset
python -m rc_bench.run --only resdag-cpu resdag-gpu reservoirpy --contexts forecast

# render Markdown speedup tables
python -m rc_bench.report --in results/latest.json --out results/REPORT.md

# run AND render in one deterministic step (refreshes results/REPORT.md)
python -m rc_bench.run --report
```

CPU predictions are seed reproducible within a fixed software/hardware
environment; measured timings vary between runs. CUDA predictions may also
vary with nondeterministic kernels. Use a new `--out` path for corrected runs
so historical snapshots remain intact. Available-adapter execution failures and
malformed results make the command exit nonzero while preserving error details
in JSON. Missing optional libraries remain explicit skips.

## Performance-regression gate (committed targets)

The cross-library `REPORT.md` numbers depend on external libraries and specific
hardware, so they can't gate CI. What *can* gate CI are resdag's **internal**
speedups — each fast path benchmarked against resdag's own naive path on the
same machine — checked against committed floors. Two files implement this:

| File | Role |
|---|---|
| [`targets.json`](targets.json) | Committed floors. `internal_ratios` are the **hard gate** (hardware-portable CPU A/B ratios, no external libs). `external_floors` are **informational only** — cross-library numbers transcribed from `REPORT.md`, never asserted. |
| [`../tests/test_benchmarks/test_perf_regression.py`](../tests/test_benchmarks/test_perf_regression.py) | The `benchmark`-marked test that reads `targets.json` and asserts each internal ratio (and each fast path's numerical correctness against its naive twin). |

Three internal ratios are gated, each tied to the ticket that introduced the
fast path:

| Ratio (`targets.json` key) | What it compares | Floor | Ticket |
|---|---|---|---|
| `flat_forecast_vs_graph_reexec` | flat single-step forecast engine vs per-step pytorch_symbolic graph walk | ≥ 1.8× | #254 |
| `vectorized_ngrc_vs_stepwise` | vectorized `NGCell.forward_sequence` vs a Python loop over `forward` | ≥ 5× | #255 |
| `fast_spectral_radius_vs_dense_eigvals` | sparse SciPy Arnoldi estimate vs dense `eigvals` | ≥ 2× | #185 |

The floors are deliberately conservative against a loaded CI runner — they catch
a fast path silently reverting to the naive path (an *algorithmic* regression),
not small timing drift. Each cell also records an `aspirational_ratio`: the
headline speedup from the audit on favourable hardware (GPU and/or large *N*),
documented but **not** enforced because it isn't hardware-portable. Timing is an
interleaved best-of-N A/B (both paths timed back-to-back per rep, best ratio
kept) so the gate is robust to transient load.

Run the gate locally — it is deselected from the normal lane via the `benchmark`
marker:

```bash
# from the repo root
pytest -m benchmark --no-cov -q tests/test_benchmarks tests/test_performance
```

In CI it runs in the `benchmark` lane of [`ci.yml`](../.github/workflows/ci.yml)
— nightly, on manual dispatch, and on any PR carrying the **`benchmark`** label —
which also re-renders and uploads `REPORT.md` as a build artifact. To retune a
floor, edit the number in `targets.json`; the test reads it, so the gate and the
documented target never drift apart.

## Adding a library

Drop a new adapter in `rc_bench/adapters/` implementing
`Adapter.is_available / time_train / time_forecast` (see `base.py`), register it
in `rc_bench/adapters/__init__.py`, and add its column to `rc_bench/report.py`.
In-process libraries are timed by the shared harness; out-of-process ones (like
the Julia worker in `julia/rc_bench.jl`) time themselves and print
`TIMES`/`RMSE` lines. That is the whole contract — e.g. a
ReservoirComputing.jl-style `reservoir-computing` (PyPI) or `pytorch-esn`
adapter is a single new file.

## Fairness notes

The *nominal* architecture (size, spectral radius, leak rate, connectivity,
ridge) is identical everywhere, but each library uses its own idiomatic
implementation: resdag runs a **dense** reservoir and a direct ridge solver;
reservoirpy and ReservoirComputing.jl use their own reservoir operators and
direct readout solves. CPU rows run float64; the resdag GPU row runs
float32 (the realistic GPU dtype). These are real, documented differences — the
tables report each library as it is actually used. All adapters honor the
requested timing repeat/warmup counts; Julia startup is outside its internal
clock, and an explicit warmup is needed to exclude compilation. The Julia
worker still requires an installed, working compatible Julia project; protocol
unit checks alone do not establish that its optional runtime is available.
