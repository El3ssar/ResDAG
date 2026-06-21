#!/usr/bin/env python3
"""Benchmark: unroll vs compiled-step vs scan reservoir recurrence (issue #257).

Compares four ways to run the per-timestep reservoir loop, to make concrete the
claim behind :meth:`resdag.core.ESNModel.compile_reservoirs` and
``ESNLayer(compile_mode="scan")``:

1. **eager loop** — the default Python time loop over ``cell.step``.
2. **compiled whole-forward** — ``torch.compile(reservoir)``, which *unrolls*
   the Python loop into one graph node per timestep.  This is the anti-pattern:
   its first-compile cost grows with sequence length and the compiled call is
   often slower than eager.  We report its first-compile wall time to expose the
   unroll cost (this is the headline acceptance number — meaningful on CUDA;
   on CPU it is only a proxy).
3. **compiled step** — ``cell.step`` wrapped in ``torch.compile`` (what
   ``compile_reservoirs`` does): the graph is a single timestep, captured once
   and replayed every step, so launch overhead amortises without unrolling.
4. **scan** — ``compile_mode="scan"`` lowers the loop to
   :func:`torch._higher_order_ops.scan`, a single ``combine_fn`` over the whole
   sequence (one graph region rather than ``T`` nodes).

Correctness is checked first: every path must match the eager loop within
``1e-4`` before any timing is trusted.

Usage
-----
    python benchmarks/compile_scan_reservoir.py
    python benchmarks/compile_scan_reservoir.py --size 200 --steps 500 --device cuda
    python benchmarks/compile_scan_reservoir.py --json results/compile_scan.json

Notes
-----
The per-step-latency and first-compile-time acceptance criteria in #257 are
specified on CUDA (the per-step update is launch-bound, not FLOP-bound, so the
compiled-step / scan wins are a GPU phenomenon).  On CPU this script still runs
end to end and proves correctness + relative cost, but treat the absolute
latency numbers as a proxy, not the CUDA acceptance figures.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Callable

import torch

from resdag.layers import ESNLayer
from resdag.layers.reservoirs.base_reservoir import _scan_available


def _sync(device: torch.device) -> None:
    """Block until queued device work finishes (no-op on CPU)."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time(
    fn: Callable[[], torch.Tensor],
    device: torch.device,
    repeats: int,
    warmups: int,
) -> tuple[float, torch.Tensor]:
    """Return (median seconds per call, last output) after discarding warmups."""
    last: torch.Tensor | None = None
    times: list[float] = []
    for i in range(warmups + repeats):
        _sync(device)
        t0 = time.perf_counter()
        last = fn()
        _sync(device)
        dt = time.perf_counter() - t0
        if i >= warmups:
            times.append(dt)
    assert last is not None
    return statistics.median(times), last


def _build(
    size: int,
    feedback: int,
    device: torch.device,
    *,
    compile_mode: str = "loop",
    seed: int = 7,
) -> ESNLayer:
    """A frozen, eval-mode ESN layer on ``device`` with a fixed seed."""
    layer = ESNLayer(
        size,
        feedback_size=feedback,
        spectral_radius=0.9,
        seed=seed,
        compile_mode=compile_mode,
    )
    return layer.to(device).eval()


def run(
    size: int,
    steps: int,
    feedback: int,
    batch: int,
    device: torch.device,
    repeats: int,
    warmups: int,
) -> dict:
    """Run all four variants once and collect timings + correctness."""
    torch.manual_seed(0)
    fb = torch.randn(batch, steps, feedback, device=device)

    results: dict[str, dict] = {}

    # --- 1. eager Python loop (the reference) -----------------------------
    eager = _build(size, feedback, device)
    with torch.no_grad():
        ref = eager(fb)
    eager.reset_state()
    med, _ = _time(lambda: _run_fresh(eager, fb), device, repeats, warmups)
    results["eager_loop"] = {"median_s": med, "max_err": 0.0}

    # --- 2. compiled whole-forward (unroll) — first-compile cost ----------
    unroll = _build(size, feedback, device)
    compiled_forward = torch.compile(unroll)
    _sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = compiled_forward(fb)  # first call traces + unrolls the loop
    _sync(device)
    first_compile_s = time.perf_counter() - t0
    err = (out - ref).abs().max().item()
    unroll.reset_state()
    med, _ = _time(
        lambda: _run_fresh_compiled(unroll, compiled_forward, fb),
        device,
        repeats,
        warmups,
    )
    results["compiled_unroll"] = {
        "median_s": med,
        "first_compile_s": first_compile_s,
        "max_err": err,
    }

    # --- 3. compiled step (compile_reservoirs strategy) -------------------
    step_layer = _build(size, feedback, device)
    step_layer.cell.step = torch.compile(step_layer.cell.step)  # type: ignore[method-assign]
    with torch.no_grad():
        out = step_layer(fb)  # first call compiles the single-step kernel
    err = (out - ref).abs().max().item()
    step_layer.reset_state()
    med, _ = _time(lambda: _run_fresh(step_layer, fb), device, repeats, warmups)
    results["compiled_step"] = {"median_s": med, "max_err": err}

    # --- 4. scan lowering -------------------------------------------------
    if _scan_available():
        scan_layer = _build(size, feedback, device, compile_mode="scan")
        with torch.no_grad():
            out = scan_layer(fb)
        err = (out - ref).abs().max().item()
        scan_layer.reset_state()
        med, _ = _time(lambda: _run_fresh(scan_layer, fb), device, repeats, warmups)
        results["scan"] = {"median_s": med, "max_err": err}
    else:
        results["scan"] = {"median_s": float("nan"), "max_err": float("nan"), "skipped": True}

    return {
        "config": {
            "size": size,
            "steps": steps,
            "feedback": feedback,
            "batch": batch,
            "device": device.type,
            "torch": torch.__version__,
            "scan_available": _scan_available(),
        },
        "variants": results,
    }


def _run_fresh(layer: ESNLayer, fb: torch.Tensor) -> torch.Tensor:
    """Reset state then run one forward (so repeats are independent)."""
    layer.reset_state()
    with torch.no_grad():
        return layer(fb)


def _run_fresh_compiled(
    layer: ESNLayer, compiled: Callable[[torch.Tensor], torch.Tensor], fb: torch.Tensor
) -> torch.Tensor:
    """Reset the underlying layer state then call its compiled wrapper."""
    layer.reset_state()
    with torch.no_grad():
        return compiled(fb)


def _report(payload: dict) -> None:
    """Pretty-print the timing table to stdout."""
    cfg = payload["config"]
    print(
        f"\nReservoir recurrence benchmark — N={cfg['size']} T={cfg['steps']} "
        f"batch={cfg['batch']} on {cfg['device']} (torch {cfg['torch']})\n"
    )
    eager = payload["variants"]["eager_loop"]["median_s"]
    header = (
        f"{'variant':<20}{'median (ms)':>14}{'per-step (us)':>16}{'speedup':>10}{'max_err':>12}"
    )
    print(header)
    print("-" * len(header))
    for name, r in payload["variants"].items():
        med = r["median_s"]
        if r.get("skipped"):
            print(f"{name:<20}{'skipped (no scan HOP)':>52}")
            continue
        per_step_us = med / cfg["steps"] * 1e6
        speedup = eager / med if med > 0 else float("nan")
        extra = ""
        if "first_compile_s" in r:
            extra = f"  [first-compile {r['first_compile_s']:.2f}s]"
        print(
            f"{name:<20}{med * 1e3:>14.3f}{per_step_us:>16.2f}"
            f"{speedup:>10.2f}{r['max_err']:>12.2e}{extra}"
        )
    print(
        "\nNote: per-step latency and first-compile-time targets in #257 are "
        "CUDA acceptance figures;\nCPU numbers here are a correctness + relative-cost proxy.\n"
    )


def main() -> None:
    """Parse args, run the benchmark, print and optionally persist results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=200, help="Reservoir size N")
    parser.add_argument("--steps", type=int, default=500, help="Sequence length T")
    parser.add_argument("--feedback", type=int, default=3, help="Feedback dimension")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--repeats", type=int, default=10, help="Timed repeats")
    parser.add_argument("--warmups", type=int, default=3, help="Discarded warmup runs")
    parser.add_argument("--json", type=str, default=None, help="Write results to this JSON path")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")

    payload = run(
        size=args.size,
        steps=args.steps,
        feedback=args.feedback,
        batch=args.batch,
        device=device,
        repeats=args.repeats,
        warmups=args.warmups,
    )
    _report(payload)

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
