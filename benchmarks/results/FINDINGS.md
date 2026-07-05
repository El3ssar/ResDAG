# Benchmark findings — resdag 0.7.0

**Date:** 2026-07-05 · **Machine:** RTX 3060 Laptop GPU (6 GB), CUDA · CPU float64, GPU float32 ·
**Libraries:** resdag 0.7.0 (PyTorch), reservoirpy 0.4.2 (NumPy/SciPy), ReservoirComputing.jl 0.12.23 (Julia).

Full speedup tables: [`REPORT.md`](REPORT.md). Raw data: [`latest.json`](latest.json) (cross-library),
[`internal_cells.json`](internal_cells.json) (resdag-only cell profile). Reproduce: [`../REPRODUCE.md`](../REPRODUCE.md).

All libraries run the same nominal ESN on a byte-identical Lorenz-63 series; each cell is the median
wall-clock over 3 repeats (1 warmup discarded; GPU `cuda.synchronize`'d before the clock stops).

---

## Headline

**resdag forecasts dynamical systems markedly longer and faster than either competitor, and the gap
widens with scale.**

- **Valid prediction time (accuracy that matters):** at N=1000, resdag holds the attractor for
  **551 steps ≈ 10.0 Lyapunov times**, vs reservoirpy **143 (2.6 Λt)** and RC.jl **27 (0.5 Λt)** —
  **≈ 3.9× reservoirpy, ≈ 20× RC.jl.** The lead holds across N=500/1000/2000.
- **Long-horizon forecast (H=10000):** resdag-GPU **297 ms @ rmse 0.146** vs reservoirpy
  **1439 ms @ 0.555** vs RC.jl **40065 ms @ 1.079** — ~5× faster than reservoirpy and ~135× faster
  than RC.jl, at ~3.8× lower error.
- **Training at scale (N=2000):** resdag-GPU **556 ms** vs reservoirpy **13.9 s** vs RC.jl **failed
  (NaN)** — **~25× reservoirpy**. resdag-GPU is the only library that stays sub-second at N=2000.

## Cross-library summary (speedup vs reservoirpy = 1.00×)

| Context | resdag-CPU | resdag-GPU | RC.jl | Notes |
|---|---|---|---|---|
| Train, N=250 | 1.4× | 0.6× | 0.3× | small N: GPU launch overhead loses to CPU |
| Train, N=1000 | 1.7× | 2.6× | 0.1× | |
| Train, N=2000 | 1.0× | **25.0×** | — (fail) | CPU dense path saturates; GPU dominates |
| Train data, T=1000 | 4.4× | 6.0× | 0.2× | |
| Forecast, H=10000 | 1.0× | ~4.8× | 0.04× | resdag rmse 0.146 vs rpy 0.555 vs jl 1.079 |
| Precision (VPT), N=1000 | — | — | — | resdag 10.0 Λt · rpy 2.6 Λt · jl 0.5 Λt |

Where resdag does **not** lead: at small reservoirs (N≤250) the GPU loses to its own CPU path and to
reservoirpy — the CPU→GPU crossover is ~N=500. At N=2000 the resdag **CPU** path (dense) ties
reservoirpy's sparse CPU path (14.5 s vs 13.9 s); this is the one place a competitor is level, and it
is exactly what sparse-CSR `weight_hh` would fix.

## Internal new-cell cost profile (resdag-only; no competitor equivalent)

GPU, forecast rmse where meaningful (Lorenz closed-loop):

| Cell | N=2000 GPU train | fcast | peak GPU mem | closed-loop rmse | verdict |
|---|---|---|---|---|---|
| `classic_esn` | 342 ms | 115 ms | 480 MB | 0.144 | strong forecaster |
| `ott_esn` | 332 ms | 126 ms | 352 MB | 0.145 | strong forecaster |
| `es2n` | 509 ms | 144 ms | 329 MB | 0.142 | strong forecaster |
| `deep_esn` | 564 ms | 186 ms | 466 MB | 0.146 | strong forecaster |
| `resesn` | 509 ms | 155 ms | 362 MB | 0.117–1.68 | tuning-sensitive |
| `gated_esn` | 635 ms | 181 ms | 396 MB | diverges (1e0–1e5) | **input-driven — not closed-loop** |
| `eusn` | 534 ms | 155 ms | 413 MB | diverges (3.8–460) | **input-driven — not closed-loop** |
| `lsm` | 903 ms | 253 ms | 430 MB | n/a | **input-driven — not closed-loop** |

The three input-driven reservoirs (`gated_esn`, `eusn`, `lsm`) diverge in an autoregressive Lorenz
rollout — **exactly as their docstrings warn**; the benchmark ran them in a forecast context only to
profile cost, not to score accuracy. `classic/ott/es2n/deep` are the closed-loop forecasters and all
sit at rmse ≈ 0.14. Spiking (`lsm`) is the most expensive per step (per-neuron surrogate spike); the
edge-of-stability and gated cells add ~1.5× over `classic_esn` for their extra ops.

## Perf opportunities (prioritized; each seeds a later phase)

1. **Sparse CSR `weight_hh`** *(impact: high · effort: M)* — the only context where resdag-CPU merely
   ties a competitor is N=2000 dense (14.5 s). CSR spmv at RC-typical densities restores the lead and
   lifts the dense memory ceiling. *Validated by the N=2000 CPU row.* → Phase 14.
2. **GPU compile/cudagraph warmup** *(impact: med · effort: M)* — the H=1000 GPU forecast paid a
   one-time **21.9 s** `torch.compile` inductor warmup (step itself 54.8 ms). `compile=True` only pays
   off once the horizon amortizes it; a manual CUDA-graph capture (`compile='cudagraph'`) would cut the
   warmup to ~ms for interactive/HPO use. Default `compile=False` remains correct. → Phase 14/perf.
3. **Enable TF32** *(impact: low-med · effort: S)* — inductor warns TensorFloat32 tensor cores are
   available but off; an opt-in `torch.set_float32_matmul_precision('high')` would speed GPU float32
   matmuls at a small precision cost. Consider a documented facade/env knob.
4. **Device auto-selection by size** *(impact: low · effort: S)* — CPU beats GPU below ~N=500; the
   facade could pick the device by reservoir size to avoid the small-N GPU tax.
5. **Divergence guard for input-driven cells** *(impact: low · effort: S)* — `gated_esn`/`eusn`
   closed-loop rmse reached 1e5/460; a NaN/overflow guard (or a louder runtime warning when someone
   autoregressively rolls an input-driven reservoir) would fail safe. Docs already warn; code could too.

## Reproducibility

- Every results JSON embeds an `env` block (library versions, CUDA/GPU, CPU, git SHA, timestamp) via
  `rc_bench/env.py`. Exact commands are in [`../REPRODUCE.md`](../REPRODUCE.md).
- CPU and precision (VPT) rows are seed-deterministic → re-running `--report` regenerates the same
  tables. Only the `resdag-gpu` timing rows carry inherent CUDA float32 jitter; `REPORT.md` is the
  canonical committed snapshot, regenerated deliberately on this machine.
- The internal perf gate (`pytest -m benchmark`) passed 8/8 — the committed internal-speedup floors in
  `targets.json` still hold on 0.7.0.
