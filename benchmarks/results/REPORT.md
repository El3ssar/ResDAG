## Reservoir-computing library benchmark

> **Historical, unverified for current releases.** The phase-one audit found
> forecast-index alignment differences and unsupported optional adapter paths
> in the comparison harness. These archived numbers must not be used to claim
> current speed or accuracy superiority. Correcting the harness and rerunning
> matched workloads is tracked in `docs/project/roadmap.md`, phase 5.

Speedups are relative to **reservoirpy** (so reservoirpy = 1.00×; a value above 1× is *faster than reservoirpy*). Every library runs the same nominal ESN architecture on a byte-identical Lorenz-63 series; each cell is the median wall-clock time over 3 repeats.


### Training vs reservoir size

Fit the ridge readout on a fixed-length sequence while growing the reservoir. Stresses the reservoir run + the readout solve.

| Reservoir size | reservoirpy (CPU) | resdag (CPU) | resdag (GPU) | ReservoirComputing.jl (CPU) |
|---|---|---|---|---|
| 250 | 219.6 ms (1.00×) | 161.7 ms (**1.4×**) | 366.2 ms (**0.6×**) | 813.6 ms (**0.3×**) |
| 500 | 426.9 ms (1.00×) | 264.5 ms (**1.6×**) | 358.6 ms (**1.2×**) | 3.23 s (**0.1×**) |
| 1000 | 1.13 s (1.00×) | 649.6 ms (**1.7×**) | 437.2 ms (**2.6×**) | 15.85 s (**0.1×**) |
| 2000 | 13.92 s (1.00×) | 14.46 s (**1.0×**) | 555.8 ms (**25.0×**) | — |


### Training on lots of data

Fix the reservoir at 1000 units and grow the training sequence. Stresses the per-timestep reservoir loop at scale.

| Train length | reservoirpy (CPU) | resdag (CPU) | resdag (GPU) | ReservoirComputing.jl (CPU) |
|---|---|---|---|---|
| 1000 | 541.4 ms (1.00×) | 122.0 ms (**4.4×**) | 90.9 ms (**6.0×**) | 2.99 s (**0.2×**) |
| 5000 | 1.10 s (1.00×) | 662.9 ms (**1.7×**) | 437.6 ms (**2.5×**) | 25.09 s (**0.0×**) |
| 20000 | 3.34 s (1.00×) | 2.31 s (**1.4×**) | 1.52 s (**2.2×**) | — |


### Long-horizon autoregressive forecasting

After an identical warmup, generate ever-longer closed-loop forecasts. Stresses the single-step reservoir update in a tight loop.

| Horizon | reservoirpy (CPU) | resdag (CPU) | resdag (GPU) | ReservoirComputing.jl (CPU) |
|---|---|---|---|---|
| 1000 | 283.1 ms (1.00×) | 175.1 ms (**1.6×**) | 54.8 ms (**5.2×**) | 3.66 s (**0.1×**) |
| 5000 | 694.7 ms (1.00×) | 665.5 ms (**1.0×**) | 158.0 ms (**4.4×**) | 15.83 s (**0.0×**) |
| 10000 | 1.44 s (1.00×) | 1.39 s (**1.0×**) | 297.4 ms (**4.8×**) | 40.07 s (**0.0×**) |


### Forecast precision (valid prediction time)

Predictive *skill*, not speed: after an identical warmup, how many steps the closed-loop forecast tracks the true Lorenz trajectory before the normalized error crosses the threshold. Higher is better.

| Reservoir size | reservoirpy (CPU) | resdag (CPU) | resdag (GPU) | ReservoirComputing.jl (CPU) |
|---|---|---|---|---|
| 500 | 52 steps (0.9 Λt) | 153 steps (2.8 Λt) | 153 steps (2.8 Λt) | 89 steps (1.6 Λt) |
| 1000 | 143 steps (2.6 Λt) | 551 steps (10.0 Λt) | 551 steps (10.0 Λt) | 27 steps (0.5 Λt) |
| 2000 | 342 steps (6.2 Λt) | 555 steps (10.1 Λt) | 555 steps (10.1 Λt) | — |

*Valid prediction time: steps until the normalized forecast error first exceeds 0.5, also in Lyapunov times (Λt). Higher = the model stays on the true trajectory longer.*


### Setup

- **Architecture (identical across libraries):** 3-D input, spectral radius 0.9, leak rate 0.3, connectivity 0.1, ridge 1e-06, tanh activation.
- **Hardware:** Linux-7.0.14-zen1-1-zen-x86_64-with-glibc2.43, NVIDIA GeForce RTX 3060 Laptop GPU, 16 CPU threads.
- **Versions:** python 3.14.2, resdag 0.7.0, reservoirpy 0.4.2, torch 2.10.0+cu128.
- **Method:** median of 3 repeats (1 warmup discarded), generated 2026-07-05T13:15:53Z.
