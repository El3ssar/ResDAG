# Reproducing the benchmark artifacts

Every new comparison run records the `heldout-v1` protocol from
[`README.md`](README.md#held-out-protocol-v1): chronological held-out truth,
training-only preprocessing/scoring scales, headline threshold 0.3 and secondary
threshold 0.5. Historical JSON files do not acquire this protocol retroactively.
CPU predictions are seed reproducible within a fixed environment, but wall-clock
timings vary; CUDA kernels may introduce prediction variability too.
Each results JSON embeds an `env` block
(library versions, CUDA + GPU, CPU model, git SHA, shell-sourced timestamp — see
[`rc_bench/env.py`](rc_bench/env.py)) so a file always records the machine and
commit it came from.

All commands are run **from the `benchmarks/` directory** with `uv run` so they
use the project venv:

```bash
cd benchmarks
```

## 0. One-time setup

```bash
# resdag + torch (from repo root, in the project venv)
uv pip install -e ".[dev]"

# reservoirpy competitor
uv pip install -r benchmarks/requirements.txt

# GPU rows: install a CUDA torch build (the resdag-gpu / internal-cells cuda rows
# are skipped automatically when torch.cuda.is_available() is False)

# Julia competitor (ReservoirComputing.jl) — optional; skipped if Julia is absent
julia --project=benchmarks/julia -e 'using Pkg; Pkg.instantiate()'
```

Confirm the competitors are visible to the harness:

```bash
uv run python -c "import reservoirpy; print('reservoirpy', reservoirpy.__version__)"
julia --version   # ReservoirComputing.jl adapter needs Julia on PATH
```

## 1. Cross-library table -> `results/latest.json` (+ `results/REPORT.md`)

The full cross-library matrix (resdag-cpu, resdag-gpu, reservoirpy, RC.jl) over
the four contexts (train-vs-size, train-vs-data, forecast, precision):

```bash
# write corrected results separately, preserving historical snapshots
uv run python -m rc_bench.run --out results/heldout-v1.json --report
```

This writes `results/heldout-v1.json` (including exact per-point split and scale
metadata) and refreshes `results/REPORT.md`. Pass `--report results/HELDOUT.md`
to retain the historical Markdown report as well.

Useful subsets:

```bash
# quick smoke (tiny matrix)
uv run python -m rc_bench.run --quick --out results/heldout-smoke.json

# a single library / a single context
uv run python -m rc_bench.run --only resdag-cpu reservoirpy --contexts forecast
uv run python -m rc_bench.run --only rcjl --contexts forecast   # Julia adapter key is "rcjl"

# render a report from an existing JSON
uv run python -m rc_bench.report --in results/latest.json --out results/REPORT.md
```

> Adapter keys: `resdag-cpu`, `resdag-gpu`, `reservoirpy`, `rcjl` (the Julia
> ReservoirComputing.jl worker). Note `--only rcjl`, **not** `--only julia`.

## 2. resdag-internal cell profile -> `results/internal_cells.json`

A resdag-only cost profile timing **init / train / forecast** wall-clock (median
of repeats, one warmup discarded, `cuda.synchronize` before each clock stop) for
every premade family — `classic_esn`, `ott_esn`, `es2n`, `resesn`, `gated_esn`,
`eusn`, `lsm`, `deep_esn` — across reservoir sizes on CPU and (if present) GPU,
plus per-cell peak GPU memory (`torch.cuda.max_memory_allocated`):

```bash
# full profile: sizes 500/1000/2000, CPU + CUDA -> results/internal_cells.json
uv run python -m rc_bench.internal_cells
```

Useful subsets:

```bash
# quick smoke (tiny sizes, short train/horizon)
uv run python -m rc_bench.internal_cells --quick

# restrict sizes / devices / families
uv run python -m rc_bench.internal_cells --sizes 500 1000 --devices cpu
uv run python -m rc_bench.internal_cells --only classic_esn eusn deep_esn
```

Competitor libraries have none of these cells, so this is a resdag-internal cost
profile, **not** a cross-library table. **LSM caveat:** an LSM is an
input-driven regressor, not a closed-loop chaos forecaster — its `forecast`
timing is a mechanical rollout *cost* only, not predictive skill (the cell
carries a `notes` field saying so, and its RMSE is omitted).

## 3. Internal performance-regression gate (CI floors)

Independent of the two artifacts above; asserts resdag's internal fast-path
ratios against committed floors in [`targets.json`](targets.json). Run from the
**repo root**:

```bash
pytest -m benchmark --no-cov -q tests/test_benchmarks tests/test_performance
```

## Determinism notes

- CPU rows and precision (VPT) metrics are seed-deterministic — re-running `-m
  rc_bench.run` reproduces predictions within a fixed environment. Timing tables
  are measurements and are not byte deterministic.
- The `resdag-gpu` row carries run-to-run floating-point jitter (inherent CUDA
  non-determinism); treat the committed `results/REPORT.md` as the canonical
  cross-library snapshot and regenerate it deliberately on the benchmark machine.
- The `env` block's timestamp is read from the shell (`date`), never from the
  Python process clock, so it reflects the machine that produced the artifact.
- Reservoir sizes are capped to fit a 6 GB GPU (the dense `N×N` reservoir at
  `N=2000` float32 is ~16 MB, so headroom is ample).

Run protocol regression checks from the repository root:

```bash
uv run pytest tests/test_benchmark_protocol.py --no-cov
```

An installed ReservoirPy enables the real one-step alignment check; otherwise
that test is an explicit skip. Available adapters returning malformed forecasts,
wrong timing counts or runtime errors make the comparison command fail. A model
that numerically diverges receives its valid prefix and an explicit divergence
status instead of becoming an adapter execution error. Full-horizon accurate
forecasts are marked censored; their reported horizon is a lower bound.
