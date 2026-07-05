"""resdag-internal reservoir-cell cost profile (NOT a cross-library table).

The cross-library harness (:mod:`rc_bench.run`) compares one *standard* ESN
against reservoirpy and ReservoirComputing.jl. This module is the complementary
**resdag-only** benchmark: it times the ``init`` / ``train`` / ``forecast``
wall-clock of every premade reservoir *family* resdag ships — ``classic_esn``,
``ott_esn``, ``es2n``, ``resesn``, ``gated_esn``, ``eusn``, ``lsm``,
``deep_esn`` — on one shared Lorenz-63 trajectory, across a few reservoir sizes,
on CPU and (if present) GPU. Competitor libraries have none of these cells, so
this is a cost profile of resdag's own zoo, not a speedup table.

Methodology (mirrors the cross-library harness)
------------------------------------------------
* One shared, normalized Lorenz-63 trajectory (``rc_bench.data.lorenz``), same
  seed as the cross-library run.
* Each phase is timed separately: **init** = build the model + move to device;
  **train** = warmup + algebraic ridge fit (:class:`~resdag.training.ESNTrainer`);
  **forecast** = ``horizon``-step closed-loop rollout.
* ``torch.cuda.synchronize()`` before every clock stop on GPU; one warmup repeat
  discarded; the **median** of the remaining repeats is reported.
* On GPU, per-cell **peak allocated memory** (``torch.cuda.max_memory_allocated``)
  is captured around the whole build->train->forecast cycle.

.. important::
   An **LSM** is an input-driven regressor, not a closed-loop chaos forecaster
   (its readout reads a spike trace, not the feedback variable, and the
   unbounded readout can diverge when fed back). Its ``forecast`` timing here is
   a mechanical rollout **cost** measurement only — do not read its forecast
   numbers as predictive skill. It is included so the cost profile covers every
   family; the caveat is recorded per-cell in the JSON (``notes``).

Output
------
``results/internal_cells.json`` with the same ``env`` reproducibility block as
``results/latest.json`` (see :mod:`rc_bench.env`).

Examples
--------
    python -m rc_bench.internal_cells                 # full profile
    python -m rc_bench.internal_cells --quick         # tiny smoke
    python -m rc_bench.internal_cells --sizes 500 1000
    python -m rc_bench.internal_cells --devices cpu
    python -m rc_bench.internal_cells --only classic_esn eusn
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import HParams
from .data import lorenz
from .env import collect_env

_DEFAULT_OUT = Path(__file__).resolve().parents[1] / "results" / "internal_cells.json"

# The premade families to profile. LSM is a driven regressor (see module note);
# it is profiled for *cost*, flagged with a caveat in the JSON.
FAMILIES: tuple[str, ...] = (
    "classic_esn",
    "ott_esn",
    "es2n",
    "resesn",
    "gated_esn",
    "eusn",
    "lsm",
    "deep_esn",
)

_LSM_NOTE = (
    "LSM is an input-driven regressor, not a closed-loop chaos forecaster; "
    "forecast timing is a mechanical rollout cost, not predictive skill."
)


def _p(*a: Any, **k: Any) -> None:
    k.setdefault("flush", True)
    print(*a, **k)


@dataclass(frozen=True)
class CellPoint:
    """One profiled configuration: a family at a reservoir size on a device."""

    reservoir_size: int = 1000
    train_len: int = 5000
    warmup: int = 200
    horizon: int = 1000


def _build_model(family: str, reservoir_size: int, hp: HParams, dtype: Any) -> Any:
    """Construct a premade model for ``family`` and move it to ``dtype``.

    All families share the ``(reservoir_size, feedback_size, output_size, ...)``
    signature except ``deep_esn`` (takes a list of layer sizes). The reservoir
    hyper-parameters (spectral radius, connectivity, leak) are held identical to
    the cross-library harness so the numbers are comparable to ``latest.json``.
    """
    from resdag.models import (
        classic_esn,
        deep_esn,
        es2n,
        eusn,
        gated_esn,
        lsm,
        ott_esn,
        resesn,
    )

    common = dict(
        feedback_size=hp.dim,
        output_size=hp.dim,
        spectral_radius=hp.spectral_radius,
        topology=("erdos_renyi", {"p": hp.connectivity}),
        readout_alpha=hp.ridge,
    )

    if family == "classic_esn":
        model = classic_esn(reservoir_size=reservoir_size, leak_rate=hp.leak_rate, **common)
    elif family == "ott_esn":
        model = ott_esn(reservoir_size=reservoir_size, **common)
    elif family == "es2n":
        model = es2n(reservoir_size=reservoir_size, **common)
    elif family == "resesn":
        model = resesn(reservoir_size=reservoir_size, **common)
    elif family == "gated_esn":
        model = gated_esn(reservoir_size=reservoir_size, **common)
    elif family == "eusn":
        model = eusn(reservoir_size=reservoir_size, **common)
    elif family == "lsm":
        model = lsm(reservoir_size=reservoir_size, **common)
    elif family == "deep_esn":
        # A 2-layer deep ESN whose total unit count matches the single-reservoir
        # families (two layers of reservoir_size//2 each).
        half = max(1, reservoir_size // 2)
        model = deep_esn(
            [half, half],
            feedback_size=hp.dim,
            output_size=hp.dim,
            spectral_radius=hp.spectral_radius,
            leak_rate=hp.leak_rate,
            topology=("erdos_renyi", {"p": hp.connectivity}),
            readout_alpha=hp.ridge,
        )
    else:  # pragma: no cover - guarded by FAMILIES
        raise ValueError(f"unknown family {family!r}")

    return model.to(dtype=dtype)


def _tensor(arr: np.ndarray, dtype: Any, device: str) -> Any:
    import torch

    return torch.as_tensor(arr, dtype=dtype, device=device).unsqueeze(0)


def _median_timed(
    op: Callable[[], Any],
    repeats: int,
    warmups: int,
    sync: Callable[[], None],
) -> tuple[float, list[float], Any]:
    """Run ``op`` ``warmups + repeats`` times; return (median, timed_secs, last).

    ``sync`` is called before stopping each clock (``cuda.synchronize`` on GPU).
    """
    times: list[float] = []
    last: Any = None
    for i in range(warmups + repeats):
        sync()
        t0 = time.perf_counter()
        last = op()
        sync()
        elapsed = time.perf_counter() - t0
        if i >= warmups:
            times.append(elapsed)
    median = statistics.median(times) if times else float("nan")
    return median, times, last


def _profile_cell(
    family: str,
    reservoir_size: int,
    device: str,
    hp: HParams,
    point: CellPoint,
    series: np.ndarray,
    repeats: int,
    warmups: int,
) -> dict:
    """Time init / train / forecast for one family at one size on one device."""
    import torch
    from resdag.training import ESNTrainer

    dtype = torch.float64 if device == "cpu" else torch.float32

    def sync() -> None:
        if device == "cuda":
            torch.cuda.synchronize()

    torch.manual_seed(hp.seed)

    warm = _tensor(series[: point.warmup], dtype, device)
    x = _tensor(series[point.warmup : point.train_len], dtype, device)
    y = _tensor(series[point.warmup + 1 : point.train_len + 1], dtype, device)

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # --- init: build the model + move to device -------------------------------
    def init_op() -> Any:
        torch.manual_seed(hp.seed)
        model = _build_model(family, reservoir_size, hp, dtype)
        if device == "cuda":
            model = model.to(device=device)
        return model

    init_median, init_times, model = _median_timed(init_op, repeats, warmups, sync)

    # --- train: warmup + algebraic ridge fit ----------------------------------
    def train_op() -> Any:
        model.reset_reservoirs()
        ESNTrainer(model).fit(
            warmup_inputs=(warm,), train_inputs=(x,), targets={"output": y}
        )
        return None

    train_median, train_times, _ = _median_timed(train_op, repeats, warmups, sync)

    # --- forecast: closed-loop rollout ----------------------------------------
    def forecast_op() -> Any:
        model.reset_reservoirs()
        with torch.no_grad():
            return model.forecast(warm, horizon=point.horizon)

    forecast_median, forecast_times, last = _median_timed(
        forecast_op, repeats, warmups, sync
    )

    peak_mem_mb: float | None = None
    if device == "cuda":
        peak_mem_mb = round(torch.cuda.max_memory_allocated() / 1e6, 2)

    # Short-horizon forecast RMSE sanity metric (skip for LSM — not a forecaster).
    rmse: float | None = None
    if last is not None and family != "lsm":
        pred = last.squeeze(0).detach().cpu().numpy()
        truth = series[point.warmup : point.warmup + len(pred)]
        n = min(len(truth), len(pred), 200)
        if n > 0:
            rmse = float(np.sqrt(np.mean((pred[:n] - truth[:n]) ** 2)))

    cell: dict = {
        "family": family,
        "reservoir_size": reservoir_size,
        "device": device,
        "dtype": "float64" if device == "cpu" else "float32",
        "init_s": init_median,
        "train_s": train_median,
        "forecast_s": forecast_median,
        "init_times": init_times,
        "train_times": train_times,
        "forecast_times": forecast_times,
        "peak_mem_mb": peak_mem_mb,
        "rmse": rmse,
    }
    if family == "lsm":
        cell["notes"] = _LSM_NOTE
    return cell


def main(argv: list[str] | None = None) -> int:
    import torch

    p = argparse.ArgumentParser(
        description="resdag-internal reservoir-cell cost profile (init/train/forecast)"
    )
    p.add_argument("--quick", action="store_true", help="tiny sizes for a smoke test")
    p.add_argument(
        "--sizes",
        nargs="*",
        type=int,
        default=None,
        help="reservoir sizes to sweep (default: 500 1000 2000; quick: 100 200)",
    )
    p.add_argument(
        "--devices",
        nargs="*",
        default=None,
        help="devices to run (default: cpu + cuda if available)",
    )
    p.add_argument(
        "--only", nargs="*", default=None, help="family keys to include (default: all)"
    )
    p.add_argument("--repeats", type=int, default=2, help="timed repeats per phase")
    p.add_argument("--warmups", type=int, default=1, help="discarded warmup iterations")
    p.add_argument("--train-len", type=int, default=None, help="training sequence length")
    p.add_argument("--horizon", type=int, default=None, help="forecast horizon")
    p.add_argument("--out", type=Path, default=_DEFAULT_OUT, help="results JSON path")
    args = p.parse_args(argv)

    quick = args.quick
    sizes = args.sizes or ([100, 200] if quick else [500, 1000, 2000])
    # Defaults are tuned so the full CPU+GPU sweep (8 families x 3 sizes x 2
    # devices) finishes in a few minutes; grow train_len / horizon / repeats via
    # flags for a heavier profile.
    train_len = args.train_len if args.train_len is not None else (600 if quick else 3000)
    horizon = args.horizon if args.horizon is not None else (200 if quick else 500)
    warmup = 200 if not quick else 100

    families = tuple(f for f in FAMILIES if not args.only or f in args.only)
    if not families:
        _p("No families selected.", file=sys.stderr)
        return 1

    if args.devices:
        devices = list(args.devices)
    else:
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    hp = HParams()
    need = max(train_len + 1, warmup + horizon + 1) + 10
    t0 = time.perf_counter()
    _p(f"Generating shared Lorenz series of {need} steps...", end=" ")
    series = lorenz(need, seed=hp.seed)
    _p(f"done ({time.perf_counter() - t0:.1f}s)")

    point = CellPoint(train_len=train_len, warmup=warmup, horizon=horizon)

    _p(f"Families:  {', '.join(families)}")
    _p(f"Sizes:     {sizes}")
    _p(f"Devices:   {devices}")
    _p(f"train_len={train_len}  warmup={warmup}  horizon={horizon}  "
       f"repeats={args.repeats}  warmups={args.warmups}\n")

    results: dict = {
        "meta": {
            "kind": "internal_cells",
            "description": (
                "resdag-internal per-family init/train/forecast cost profile "
                "(not a cross-library comparison)"
            ),
            "families": list(families),
            "sizes": list(sizes),
            "devices": list(devices),
            "repeats": args.repeats,
            "warmups": args.warmups,
            "hparams": hp.__dict__,
            "point": point.__dict__,
            "env": collect_env(),
        },
        "cells": [],
    }

    for device in devices:
        _p(f"=== device: {device} ===")
        for size in sizes:
            _p(f"  N={size}")
            for family in families:
                _p(f"    {family:12s} ", end="")
                cstart = time.perf_counter()
                try:
                    cell = _profile_cell(
                        family, size, device, hp, point, series, args.repeats, args.warmups
                    )
                    results["cells"].append(cell)
                    mem = (
                        f"  peak {cell['peak_mem_mb']:.0f} MB"
                        if cell["peak_mem_mb"] is not None
                        else ""
                    )
                    rmse = f"  rmse {cell['rmse']:.3f}" if cell["rmse"] is not None else ""
                    wall = time.perf_counter() - cstart
                    _p(
                        f"init {cell['init_s'] * 1e3:7.1f}  "
                        f"train {cell['train_s'] * 1e3:7.1f}  "
                        f"fcast {cell['forecast_s'] * 1e3:7.1f} ms"
                        f"{mem}{rmse}   (wall {wall:5.1f}s)"
                    )
                except Exception as e:  # keep going; record the failure
                    results["cells"].append(
                        {
                            "family": family,
                            "reservoir_size": size,
                            "device": device,
                            "error": str(e),
                        }
                    )
                    _p(f"ERROR: {str(e)[:120]}")
            # checkpoint after each size so a Ctrl-C keeps finished work
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(results, indent=2))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    _p(f"\nWrote {args.out}  (total {time.perf_counter() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
