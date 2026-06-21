# Reservoir-computing library benchmark

An extensible, re-runnable harness that compares resdag against other reservoir
-computing libraries on **identical ESN architectures and identical data**, then
renders speedup tables (relative to reservoirpy).

Currently compared:

| Library | Backend | Device(s) |
|---|---|---|
| **resdag** | PyTorch | CPU (float64) + GPU (float32) |
| [reservoirpy](https://github.com/reservoirpy/reservoirpy) | NumPy / SciPy | CPU |
| [ReservoirComputing.jl](https://github.com/SciML/ReservoirComputing.jl) | Julia | CPU |

## What it measures

Three contexts, all on a shared Lorenz-63 trajectory (one-step-ahead training,
then closed-loop forecasting):

1. **Training vs reservoir size** — fit the ridge readout while growing the reservoir.
2. **Training on lots of data** — fix the reservoir, grow the training sequence.
3. **Long-horizon autoregressive forecasting** — generate ever-longer closed-loop forecasts.

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
```

## Compile / scan microbenchmark

`compile_scan_reservoir.py` is a standalone, single-library microbenchmark for
the reservoir *recurrence* itself — it compares four ways to run the per-step
time loop and is the empirical backing for `compile_reservoirs()` and
`ESNLayer(compile_mode="scan")` (issue #257):

```bash
# defaults: N=200, T=500, CPU
python benchmarks/compile_scan_reservoir.py

# CUDA, persist results
python benchmarks/compile_scan_reservoir.py --size 200 --steps 500 --device cuda \
    --json results/compile_scan.json
```

It reports, for each variant, median wall time, per-step latency, speedup vs the
eager loop, and a `max_err` correctness check (every path must match the eager
loop within `1e-4`):

- **eager_loop** — the default Python time loop (reference).
- **compiled_unroll** — `torch.compile(reservoir)`; also prints its
  *first-compile* wall time to expose the unroll cost.
- **compiled_step** — `cell.step` wrapped in `torch.compile` (the
  `compile_reservoirs()` strategy).
- **scan** — the `compile_mode="scan"` lowering.

The per-step-latency and first-compile-time *acceptance* numbers in #257 are
CUDA figures (the per-step update is launch-bound, so the compiled-step / scan
wins are a GPU phenomenon); on CPU the script still proves correctness and
relative cost but the absolute latencies are a proxy.

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
implementation: resdag runs a **dense** reservoir + a conjugate-gradient ridge
solve (GPU-friendly), while reservoirpy and ReservoirComputing.jl use **sparse**
reservoirs + direct solves. CPU rows run float64; the resdag GPU row runs
float32 (the realistic GPU dtype). These are real, documented differences — the
tables report each library as it is actually used.
