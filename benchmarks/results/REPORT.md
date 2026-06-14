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
| 500 | 52 steps (0.9 Λt) | 146 steps (2.6 Λt) | 148 steps (2.7 Λt) | 89 steps (1.6 Λt) |
| 1000 | 143 steps (2.6 Λt) | 146 steps (2.6 Λt) | 144 steps (2.6 Λt) | 27 steps (0.5 Λt) |
| 2000 | 342 steps (6.2 Λt) | 147 steps (2.7 Λt) | 149 steps (2.7 Λt) | — |

*Valid prediction time: steps until the normalized forecast error first exceeds 0.5, also in Lyapunov times (Λt). Higher = the model stays on the true trajectory longer.*


### Setup

- **Architecture (identical across libraries):** 3-D input, spectral radius 0.9, leak rate 0.3, connectivity 0.1, ridge 1e-06, tanh activation.
- **Hardware:** Linux-7.0.12-zen1-1-zen-x86_64-with-glibc2.43, NVIDIA GeForce RTX 3060 Laptop GPU, 16 CPU threads.
- **Versions:** python 3.14.2, resdag 0.6.0, reservoirpy 0.4.2, torch 2.10.0+cu128.
- **Method:** median of 2 repeats (1 warmup discarded), generated 2026-06-14T17:38:11.734662+00:00.
