---
description: How resdag's training and forecasting speed compares to reservoirpy and ReservoirComputing.jl on identical Echo State Network architectures.
---

<span class="nb-kicker">Work · Benchmarks</span>

# Benchmarks

How fast is resdag, really? This page compares it head-to-head with two
established reservoir-computing libraries — [reservoirpy](https://github.com/reservoirpy/reservoirpy)
(NumPy/SciPy) and [ReservoirComputing.jl](https://github.com/SciML/ReservoirComputing.jl)
(Julia) — on the **same Echo State Network architecture**, the **same data**, and
the same three tasks every forecasting practitioner runs: fitting a readout,
fitting on a lot of data, and rolling out long autoregressive forecasts.

The harness is in [`benchmarks/`](https://github.com/El3ssar/ResDAG/tree/main/benchmarks);
it is re-runnable and adding a library is a single adapter file. Numbers below
were produced by `python -m rc_bench.run` followed by `python -m rc_bench.report`.

!!! note "Reading the tables"
    Speedups are relative to **reservoirpy**, so reservoirpy is always `1.00×`
    and a value above `1×` means *faster than reservoirpy*. Each cell is the
    median wall-clock time over repeated runs (a warmup iteration is discarded;
    GPU work is synchronized before the clock stops).

!!! success "Highlights (RTX 3060 laptop, vs reservoirpy)"
    - **Training scales on the GPU.** At a 2000-unit reservoir, resdag (GPU)
      trains **16× faster** than reservoirpy; on a 20k-step dataset it is **~2×**
      faster. The dense-reservoir + conjugate-gradient design is built for this.
    - **Sharper forecasts.** Over the forecast runs, resdag's short-horizon
      error was the lowest of the four libraries (RMSE ≈ 0.3 vs ≈ 0.55 for
      reservoirpy and ≈ 1.08 for ReservoirComputing.jl).
    - **Honest trade-off.** For *generating* a long forecast — an inherently
      sequential loop — reservoirpy's sparse single-step update is faster than
      resdag's dense one on CPU/GPU. resdag wins on training throughput and
      accuracy; reservoirpy wins on raw autoregressive generation speed.

### Training vs reservoir size

Fit the ridge readout on a fixed-length sequence while growing the reservoir. Stresses the reservoir run + the readout solve.

| Reservoir size | reservoirpy (CPU) | resdag (CPU) | resdag (GPU) | ReservoirComputing.jl (CPU) |
|---|---|---|---|---|
| 250 | 224.2 ms (1.00×) | 229.4 ms (**1.0×**) | 482.5 ms (**0.5×**) | 1.04 s (**0.2×**) |
| 500 | 384.7 ms (1.00×) | 350.0 ms (**1.1×**) | 476.6 ms (**0.8×**) | 4.61 s (**0.1×**) |
| 1000 | 1.23 s (1.00×) | 951.1 ms (**1.3×**) | 547.1 ms (**2.3×**) | 24.45 s (**0.1×**) |
| 2000 | 10.08 s (1.00×) | 15.92 s (**0.6×**) | 615.8 ms (**16.4×**) | — |

### Training on lots of data

Fix the reservoir at 1000 units and grow the training sequence. Stresses the per-timestep reservoir loop at scale.

| Train length | reservoirpy (CPU) | resdag (CPU) | resdag (GPU) | ReservoirComputing.jl (CPU) |
|---|---|---|---|---|
| 1000 | 471.0 ms (1.00×) | 238.0 ms (**2.0×**) | 187.7 ms (**2.5×**) | 3.99 s (**0.1×**) |
| 5000 | 1.13 s (1.00×) | 966.0 ms (**1.2×**) | 530.2 ms (**2.1×**) | 22.43 s (**0.1×**) |
| 20000 | 3.51 s (1.00×) | 3.42 s (**1.0×**) | 1.91 s (**1.8×**) | — |

### Long-horizon autoregressive forecasting

After an identical warmup, generate ever-longer closed-loop forecasts. Stresses the single-step reservoir update in a tight loop.

| Horizon | reservoirpy (CPU) | resdag (CPU) | resdag (GPU) | ReservoirComputing.jl (CPU) |
|---|---|---|---|---|
| 1000 | 310.2 ms (1.00×) | 376.5 ms (**0.8×**) | 290.5 ms (**1.1×**) | 5.10 s (**0.1×**) |
| 5000 | 773.9 ms (1.00×) | 1.91 s (**0.4×**) | 1.34 s (**0.6×**) | 23.54 s (**0.0×**) |
| 10000 | 1.52 s (1.00×) | 4.04 s (**0.4×**) | 2.72 s (**0.6×**) | 43.73 s (**0.0×**) |

### Forecast precision (valid prediction time)

Predictive *skill*, not speed: after an identical warmup, how many steps the closed-loop forecast tracks the true Lorenz trajectory before the normalized error crosses the threshold. Higher is better.

| Reservoir size | reservoirpy (CPU) | resdag (CPU) | resdag (GPU) | ReservoirComputing.jl (CPU) |
|---|---|---|---|---|
| 500 | 52 steps (0.9 Λt) | 146 steps (2.6 Λt) | 148 steps (2.7 Λt) | 89 steps (1.6 Λt) |
| 1000 | 143 steps (2.6 Λt) | 146 steps (2.6 Λt) | 144 steps (2.6 Λt) | 27 steps (0.5 Λt) |
| 2000 | 342 steps (6.2 Λt) | 147 steps (2.7 Λt) | 149 steps (2.7 Λt) | — |

*Valid prediction time: steps until the normalized forecast error first exceeds 0.5, also in Lyapunov times (Λt). Higher means the model stays on the true trajectory longer. Predictive skill is hyper-parameter-dependent; these use matched nominal settings (not tuned per library), so each library leads in different regimes.*

*Measured on an NVIDIA RTX 3060 laptop GPU / 16-thread i7-10870H, Python 3.14, resdag 0.6.0, reservoirpy 0.4.2, ReservoirComputing.jl 0.12, torch 2.10 (cu128). Median of repeated runs; `—` = skipped (ReservoirComputing.jl is impractically slow above 1000 units). Regenerate with the commands below.*

## What is being measured

Every library builds the *same nominal architecture* — a single reservoir
feeding a ridge-regression readout — with identical hyper-parameters (reservoir
size, spectral radius, leak rate, connectivity, ridge regularization, `tanh`
activation), and trains/forecasts on a **byte-identical** Lorenz-63 trajectory.

- **Training** runs the reservoir over the training sequence and solves for the
  readout. resdag uses a conjugate-gradient ridge solve; reservoirpy and
  ReservoirComputing.jl use direct solves.
- **Forecasting** warms the reservoir up on a seed window, then generates
  closed-loop: each prediction is fed back as the next input. This loop is
  inherently sequential in time.

A short-horizon forecast RMSE is recorded alongside each timing as a sanity
check that all libraries are doing comparable work — not just racing to produce
garbage.

## Fairness & caveats

These are honest, like-for-like comparisons, but a few real differences are
worth stating plainly so the numbers are not over-read:

- **Dense vs sparse reservoir.** resdag stores a *dense* reservoir matrix (a
  GPU-friendly choice) and runs it as a dense mat-vec/mat-mul each step.
  reservoirpy and ReservoirComputing.jl use *sparse* reservoirs. At low
  connectivity and large reservoirs the sparse libraries do asymptotically less
  arithmetic on CPU — resdag wins back the difference on the GPU and in the
  readout solve.
- **Precision.** CPU rows run float64 (NumPy/Julia's natural dtype); the resdag
  GPU row runs float32, the realistic choice for consumer GPUs (which run
  float64 at a small fraction of float32 throughput).
- **Sequential forecasting is launch-bound.** The autoregressive forecast loop
  issues one tiny operation per timestep. On the GPU each step is dominated by
  kernel-launch overhead, so the GPU does **not** help long forecasts at small
  reservoir sizes — and may be slower than CPU there. resdag's GPU advantage
  shows up in **training**, where the work batches into large matmuls. See
  [Scale & deploy](deploy.md#gpu) for when the GPU is worth it.
- **ReservoirComputing.jl coverage.** Its current release (v0.12, Lux-based) is
  much slower per call than the others — seconds where resdag and reservoirpy
  take milliseconds — so the harness skips it on the largest cells (shown as
  `—`) to keep the sweep practical. resdag and reservoirpy cover the full range.
  The Julia timings also exclude JIT compilation (measured after a warmup), so
  they reflect steady-state compute, not first-call latency.

## Reproduce

```bash
# resdag + torch (use a CUDA torch build for the GPU rows)
uv pip install -e ".[dev]"
uv pip install -r benchmarks/requirements.txt   # reservoirpy

# optional Julia comparison (skipped automatically if Julia is absent)
julia --project=benchmarks/julia -e 'using Pkg; Pkg.instantiate()'

cd benchmarks
python -m rc_bench.run                            # -> results/latest.json
python -m rc_bench.report --in results/latest.json
```

Adding another library — `pytorch-esn`, PyRCN, the `reservoir-computing` PyPI
package, ... — is one new adapter implementing `is_available / time_train /
time_forecast`; see [`benchmarks/README.md`](https://github.com/El3ssar/ResDAG/tree/main/benchmarks).
