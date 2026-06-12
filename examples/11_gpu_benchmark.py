"""11 — GPU benchmark: when does CUDA actually pay off for ESNs?

Times the three core operations — forward pass, readout fitting, and
autoregressive forecasting — on CPU vs GPU for three (batch, reservoir)
configurations:

    (1, 1000)    small single-trajectory workload
    (16, 1000)   batched workload (16 trajectories in parallel)
    (1, 3000)    large reservoir

Each op is run once to warm up (CUDA kernel/cache initialization), then
timed. Skips cleanly when CUDA is unavailable.

Expected runtime: ~30 s to ~2 min depending on the GPU/CPU pair;
instant skip on CPU-only machines.
"""

import time

import torch

import resdag as rd
from resdag.training import ESNTrainer

CONFIGS = [(1, 1000), (16, 1000), (1, 3000)]  # (batch, reservoir_size)
SEQ_LEN = 500  # timesteps for the forward benchmark
WARMUP_STEPS = 100
TRAIN_STEPS = 400
HORIZON = 200
FEATURES = 3


def timed(fn, device: torch.device) -> float:
    """Wall time of fn() with proper CUDA synchronization."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def bench_config(batch: int, size: int) -> list[tuple[str, float, float]]:
    """Return [(op, cpu_seconds, gpu_seconds), ...] for one configuration."""
    results: dict[str, dict[str, float]] = {"forward": {}, "fit": {}, "forecast": {}}

    for dev_name in ("cpu", "cuda"):
        device = torch.device(dev_name)
        # Same seed -> identical weights on both devices
        torch.manual_seed(42)
        model = rd.classic_esn(size, feedback_size=FEATURES, output_size=FEATURES).to(device)

        torch.manual_seed(0)
        x = torch.randn(batch, SEQ_LEN, FEATURES, device=device)  # (batch, time, features)
        warmup = torch.randn(batch, WARMUP_STEPS, FEATURES, device=device)
        train = torch.randn(batch, TRAIN_STEPS, FEATURES, device=device)
        target = torch.randn(batch, TRAIN_STEPS, FEATURES, device=device)
        f_warmup = torch.randn(batch, WARMUP_STEPS, FEATURES, device=device)

        def run_forward() -> None:
            model.reset_reservoirs()
            with torch.no_grad():
                model(x)

        def run_fit() -> None:
            ESNTrainer(model).fit((warmup,), (train,), {"output": target})

        def run_forecast() -> None:
            model.forecast(f_warmup, horizon=HORIZON)

        for op, fn in (("forward", run_forward), ("fit", run_fit), ("forecast", run_forecast)):
            fn()  # warm-up run (CUDA kernels, allocator, caches)
            results[op][dev_name] = timed(fn, device)

    return [(op, results[op]["cpu"], results[op]["cuda"]) for op in results]


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA is not available on this machine — benchmark skipped.")
        print("The qualitative picture from machines with a GPU:")
        print("  - GPU wins at scale (big reservoirs, big batches, heavy fits)")
        print("  - tiny configs are launch-bound and often FASTER on CPU")
        return

    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}")
    print(f"CPU threads: {torch.get_num_threads()}")
    print(f"forward: {SEQ_LEN} steps | fit: {TRAIN_STEPS} steps | forecast: {HORIZON} steps\n")

    header = f"{'(batch, size)':<16} {'op':<10} {'CPU [s]':>9} {'GPU [s]':>9} {'GPU speedup':>12}"
    print(header)
    print("-" * len(header))

    for batch, size in CONFIGS:
        for op, cpu_s, gpu_s in bench_config(batch, size):
            speedup = cpu_s / gpu_s
            print(
                f"{f'({batch}, {size})':<16} {op:<10} {cpu_s:>9.3f} {gpu_s:>9.3f} "
                f"{speedup:>11.2f}x"
            )
        print("-" * len(header))

    print(
        """
How to read this
----------------
- GPU wins at scale: larger batches and larger reservoirs turn the
  per-step matmuls and the readout's Gram-matrix formation into real GPU
  work (speedups grow with batch and reservoir size).
- Tiny configs are launch-bound: the reservoir loop is sequential in
  time, so each timestep is one small kernel launch. At (1, 1000) the GPU
  mostly waits and the CPU wins every op.
- fit() on GPU keeps the heavy Gram matmuls in float32 by default
  (CGReadoutLayer's gram_dtype auto rule) — float64 Gram formation on
  consumer GPUs runs at 1/32-1/64 throughput and would erase the win.
- Rule of thumb: big batches, big reservoirs, or many models -> GPU;
  a single small model -> CPU.
"""
    )


if __name__ == "__main__":
    main()
