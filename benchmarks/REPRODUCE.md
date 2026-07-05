# Reproducing the benchmark artifacts

Every artifact this harness writes is deterministic on a fixed machine (CPU rows
and precision metrics are seed-deterministic; only the `resdag-gpu` row carries
inherent CUDA floating-point jitter). Each results JSON embeds an `env` block
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
# run the full matrix and render the Markdown report in one deterministic step
uv run python -m rc_bench.run --report
```

This writes [`results/latest.json`](results/latest.json) (with the `env` block)
and refreshes [`results/REPORT.md`](results/REPORT.md).

Useful subsets:

```bash
# quick smoke (tiny matrix)
uv run python -m rc_bench.run --quick

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
  rc_bench.run --report` regenerates the same tables.
- The `resdag-gpu` row carries run-to-run floating-point jitter (inherent CUDA
  non-determinism); treat the committed `results/REPORT.md` as the canonical
  cross-library snapshot and regenerate it deliberately on the benchmark machine.
- The `env` block's timestamp is read from the shell (`date`), never from the
  Python process clock, so it reflects the machine that produced the artifact.
- Reservoir sizes are capped to fit a 6 GB GPU (the dense `N×N` reservoir at
  `N=2000` float32 is ~16 MB, so headroom is ample).
```
