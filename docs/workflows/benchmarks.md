---
description: Reproduce held-out reservoir forecasts, measure accuracy and runtime, and distinguish current protocols from historical comparison tables.
---

<span class="nb-kicker">Work · Benchmarks</span>

# Benchmarks

## Current measurement protocol

The comparison harness checks alignment, numerical precision and timing before
publishing new rankings.
The `heldout-v1` protocol uses raw shared data and fits normalization on the
supervised training targets after washout. Training inputs end at index `T-1`,
their next-sample targets end at `T`, and forecasts begin at the first unseen
sample, `T+1`. Every adapter receives the same transformed data.

Accuracy is the uninterrupted prefix with per-step RMS error below **0.3**,
using fixed training component scales. The report also includes **0.5**.
Compute each seed/origin's horizon separately, then report its median and
spread. A forecast that stays valid throughout the evaluated window gives a
lower bound. Report actual `dt`, system parameters and the Lyapunov exponent
when converting steps to Lyapunov times.

The [forecast experiment](../start/forecast-experiment.md) shows how to build,
draw, train, save and score vanilla, OTT and custom DAG models. Its untuned
examples are separate from a controlled comparison. The
[phase-five design](../project/phase-five-design.md) records the agreed
7–8 Lyapunov-time **median target** and the remaining experiment work.

Run a bounded Python-adapter smoke check from the repository:

```sh
python -m pip install -r benchmarks/requirements.txt
cd benchmarks
python -m rc_bench.run --quick --only resdag-cpu reservoirpy --report --out results/local-smoke.json
```

This checks both real Python adapters, chronological preparation and reporting.
An execution failure from an available adapter makes the command fail;
unavailable optional adapters are explicit skips. The settings and random
weights are realized by each library independently. Equal nominal settings
alone do not establish identical numerical models or an equal tuning budget.

### Equal tuning time

The new headline protocol gives **each library and each system ten CPU minutes**
to tune a vanilla ESN: ResDAG, reservoirpy and ReservoirComputing.jl on Lorenz
and Rössler. It uses Optuna's ordinary `study.optimize`, one worker and one CPU
thread, with the same seven controls: reservoir size, spectral radius, leak,
input scale, random bias scale, mean degree and ridge penalty.

Every complete candidate fits once for each of three initialization seeds,
then forecasts from three validation starts using that fitted model. Startup
and one fixed synthetic warmup are reported separately. Construction, fitting,
history synchronization, rollout, scoring and Julia IPC count toward the
budget. Failed, unfinished or deadline-overrun candidates cannot win; their
work and actual overrun remain in the report. There are no automatic retries.

All six winners are saved before reading final test values. Final evaluation
uses five unseen seeds and three later starts, with the same training-only
normalization, **0.3** headline and **0.5** secondary thresholds. The JSON keeps
every seed/start, failure, divergence flag and censored horizon; the Markdown
report shows the median and full range. This compares achievable accuracy
under equal tuning time. Separate matched-configuration measurements compare
throughput without confusing different winning architectures.

The common model explicitly uses Gaussian recurrent weights, directed
connectivity at `mean_degree / reservoir_size` including possible self-loops,
dense uniform input weights and a **random uniform reservoir bias**. Initial
memory is zero. A linear readout sees only reservoir states, with no direct
input features or readout intercept. Native random streams, sparse-mask
construction and direct readout solvers still differ. The precise protocol,
commands and accounting are in
[`benchmarks/BUDGET_PROTOCOL.md`](https://github.com/El3ssar/ResDAG/blob/main/benchmarks/BUDGET_PROTOCOL.md).

The pinned Julia environment has been executed with **Julia 1.12.5 and
ReservoirComputing.jl 0.12.23**. Independent NumPy ridge and recurrent-update
oracles check Float64 fitting, the first unseen forecast, history resets,
future-data isolation and requested timing repetitions. This verifies the
adapter; it does not establish a cross-library speed ranking.

## Archived comparisons

!!! warning "Historical results; current comparisons are pending"
    These tables are archived measurements, not evidence for the current
    release. The phase-one audit found mismatched forecast indexing and
    unsupported optional adapter paths in the comparison harness. Treat its
    accuracy rankings and cross-library speed ratios as unverified until the
    aligned workloads are rerun. The [roadmap](../project/roadmap.md#phase-5-speed-memory-and-comparisons)
    tracks the corrected speed–accuracy study. Internal CPU regression gates
    compare ResDAG execution paths and do not establish superiority to another library.

The historical measurements below compare ResDAG 0.6.0 head-to-head with two
established reservoir-computing libraries — [reservoirpy](https://github.com/reservoirpy/reservoirpy)
(NumPy/SciPy) and [ReservoirComputing.jl](https://github.com/SciML/ReservoirComputing.jl)
(Julia) — on the **same Echo State Network architecture**, the **same data**, and
the same three tasks every forecasting practitioner runs: fitting a readout,
fitting on a lot of data, and rolling out long autoregressive forecasts.

The harness is in [`benchmarks/`](https://github.com/El3ssar/ResDAG/tree/main/benchmarks);
it is re-runnable and adding a library is a single adapter file. Numbers below
were produced by `python -m rc_bench.run` followed by `python -m rc_bench.report`.
They describe that version and workload; rerun the harness to measure the current
release on your own hardware. The repository's
[benchmark report](https://github.com/El3ssar/ResDAG/blob/main/benchmarks/results/REPORT.md)
records later sweeps with their own environment and settings.

!!! note "Reading the tables"
    Speedups are relative to **reservoirpy**, so reservoirpy is always `1.00×`
    and a value above `1×` means *faster than reservoirpy*. Each cell is the
    median wall-clock time over repeated runs (a warmup iteration is discarded;
    GPU work is synchronized before the clock stops).

!!! note "Archived observations (RTX 3060 laptop, vs reservoirpy)"
    The following tables preserve the recorded timings and scores for that
    historical setup. Their indexing, initialization and solver caveats prevent
    using them to claim that a current library is faster or more accurate.
    GPU suitability depends on the actual model, precision and workload;
    no new CUDA throughput claim has been verified in this phase.

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

*Archived setup: NVIDIA RTX 3060 laptop GPU / 16-thread i7-10870H, Python 3.14, resdag 0.6.0, reservoirpy 0.4.2, ReservoirComputing.jl 0.12, torch 2.10 (cu128). Median of recorded repetitions; `—` means the old sweep did not measure that cell. Current commands below run the corrected protocol and will not reproduce these historical values.*

## What is being measured

The matched-workload harness builds a single reservoir feeding a ridge readout
with the same nominal settings and **byte-identical** transformed data.
Its native initializer distributions are recorded separately from the
explicit common distributions in the equal-budget track.

- **Training** runs the reservoir over the training sequence and solves for the
  readout. The current ResDAG adapter uses **RidgeReadoutLayer**, a direct
  solver; reservoirpy uses `Ridge`, and Julia uses `StandardRidge`.
  An earlier adapter used CG; those historical results do not describe the
  current direct-solver path.
- **Forecasting** warms the reservoir up on a seed window, then generates
  closed-loop: each prediction is fed back as the next input. This loop is
  inherently sequential in time.

A short-horizon forecast RMSE is recorded alongside each timing as a sanity
check that all libraries are doing comparable work — not just racing to produce
garbage.

## Fairness & caveats

Read the protocol and implementation choices alongside every result:

- **Connectivity and storage are separate.** ResDAG supports native sparse
  recurrent storage through `sparse=True`. The historical/default timing
  adapter builds a dense tensor, reservoirpy uses SciPy sparse storage, and the
  Julia timing worker currently requests a dense matrix from its sparse
  initializer. A sparsely connected mathematical reservoir does not imply a
  sparse storage backend. The new equal-budget recipe records its explicit
  storage policy separately.
- **Precision.** CPU rows run float64 (NumPy/Julia's natural dtype); the resdag
  GPU row runs float32, the realistic choice for consumer GPUs (which run
  float64 at a small fraction of float32 throughput).
- **Sequential forecasting is launch-bound.** The autoregressive forecast loop
  issues one tiny operation per timestep. On the GPU each step is dominated by
  kernel-launch overhead, so the eager GPU loop does **not** help long forecasts
  at small reservoir sizes — and may be slower than CPU there. `forecast(compile=True)`
  fuses the per-step engine with `torch.compile` to claw back launch overhead on
  long GPU rollouts; the tables above are historical eager measurements. Measure
  both fitting and forecasting on the target device before choosing it.
  See [Scale & deploy](deploy.md#gpu) for when the GPU is worth it.
- **ReservoirComputing.jl coverage.** The current adapter has no implicit
  reservoir-size cap and honors the requested repeats and warmups. Julia's
  standalone matched-workload timings are measured inside the worker after
  warmup, excluding startup/JIT. The budget track keeps one Julia process alive
  and reports its startup/synthetic warmup separately. Missing Julia or an
  uninstantiated optional environment must be reported, without treating it as
  a measured performance result.

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
