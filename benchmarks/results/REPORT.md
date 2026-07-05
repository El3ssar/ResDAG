## Reservoir-computing library benchmark

Speedups are relative to **reservoirpy** (so reservoirpy = 1.00×; a value above 1× is *faster than reservoirpy*). Every library runs the same nominal ESN architecture on a byte-identical Lorenz-63 series; each cell is the median wall-clock time over 2 repeats.


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
| 500 | 52 steps (0.9 Λt) | 548 steps (9.9 Λt) | re-run¹ | 89 steps (1.6 Λt) |
| 1000 | 143 steps (2.6 Λt) | 559 steps (10.1 Λt) | re-run¹ | 27 steps (0.5 Λt) |
| 2000 | 342 steps (6.2 Λt) | 554 steps (10.0 Λt) | re-run¹ | — |

*Valid prediction time: steps until the normalized forecast error first exceeds 0.5, also in Lyapunov times (Λt). Higher = the model stays on the true trajectory longer.*

> **resdag readout correction (2026-07-04).** The resdag (CPU) VPT column was refreshed after the benchmark adapter switched its readout from `CGReadoutLayer` to `RidgeReadoutLayer` (direct Cholesky solve — the class actually comparable to reservoirpy's `Ridge`/RC.jl's direct solve, and resdag's own premade default). The previous 146–147-step figures were a solver artifact: the iterative CG readout silently under-converged at its default `max_iter=100` on the near-unregularized (`alpha=1e-6`) Gram, flatlining precision at ~146 steps regardless of reservoir size. With the direct solve, resdag's VPT tracks ~550 steps (~10 Λt) across N=500–2000. These resdag (CPU) numbers are **measured** on the machine noted below under "resdag CPU re-run".
>
> ¹ **resdag (GPU) VPT needs a re-run on the CUDA benchmark machine** — the prior GPU figures (148/144/149) were also CG-solver artifacts, but GPU (float32) cannot be re-measured on the CPU-only machine used for this correction. Re-run `python -m rc_bench.run --only resdag-gpu --contexts precision --report` on the benchmark box to refresh this column.


### Setup

- **Architecture (identical across libraries):** 3-D input, spectral radius 0.9, leak rate 0.3, connectivity 0.1, ridge 1e-06, tanh activation.
- **Hardware:** Linux-7.0.12-zen1-1-zen-x86_64-with-glibc2.43, NVIDIA GeForce RTX 3060 Laptop GPU, 16 CPU threads.
- **Versions:** python 3.14.2, resdag 0.6.0, reservoirpy 0.4.2, torch 2.10.0+cu128.
- **Method:** median of 2 repeats (1 warmup discarded), generated 2026-06-14T17:38:11.734662+00:00.

### Which rows are current after the readout correction (2026-07-04)

- **Forecast precision — resdag (CPU):** ✅ measured with the `RidgeReadoutLayer`
  adapter. resdag CPU re-run box: Linux-7.0.14-zen1-1-zen, 16 CPU threads,
  resdag 0.4.0 (audit branch), torch 2.10.0, float64, seed 1234 — same
  Lorenz-63 series and VPT metric as the harness (verified by running
  `python -m rc_bench.run --only resdag-cpu --contexts precision`).
- **Forecast precision — resdag (GPU):** ⚠️ marked "re-run¹" — needs the CUDA
  benchmark box (see footnote).
- **Timing tables (train × size, train × data, forecast) — all libraries:**
  ⚠️ **stale after the readout switch and must be re-run in full** on the
  original benchmark machine with `python -m rc_bench.run --report`. The
  cross-library speedup ratios there are hardware- and library-specific and
  were computed against reservoirpy/RC.jl on the RTX 3060 box; they cannot be
  faithfully recomputed on the CPU-only correction box, and the direct-solve
  readout also changes resdag's own train timing (a direct Cholesky solve on
  the `(F, F)` Gram vs iterative CG). The reservoirpy/RC.jl columns are
  unchanged by this correction and carry over as recorded.
