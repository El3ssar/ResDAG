"""Benchmark runner.

Sweeps the configuration matrix over every available adapter and writes a JSON
results file. Progress is printed **live, per library**, smallest configs first,
with the wall-clock time each library takes — so a slow library is obvious as it
happens (and you can Ctrl-C without losing the libraries already recorded; pass
``--out`` and re-run a subset to fill in the rest).

Examples
--------
    python -m rc_bench.run                  # full matrix -> results/latest.json
    python -m rc_bench.run --quick          # tiny smoke test
    python -m rc_bench.run --only resdag-cpu reservoirpy --contexts forecast precision
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .adapters import all_adapters
from .config import CONTEXTS, QUICK_CONTEXTS, HParams
from .data import lorenz
from .metrics import LORENZ_DT, LORENZ_LAMBDA_MAX, lyapunov_times, valid_prediction_time

_DEFAULT_OUT = Path(__file__).resolve().parents[1] / "results" / "latest.json"


def _p(*a, **k):
    k.setdefault("flush", True)
    print(*a, **k)


def _collect_versions() -> dict:
    versions: dict[str, str] = {"python": platform.python_version()}
    for mod, key in (("resdag", "resdag"), ("reservoirpy", "reservoirpy")):
        try:
            versions[key] = __import__(mod).__version__
        except Exception:
            pass
    try:
        import torch

        versions["torch"] = torch.__version__
        versions["cuda"] = torch.version.cuda or "n/a"
        if torch.cuda.is_available():
            versions["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return versions


def _hardware() -> dict:
    import os

    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": str(os.cpu_count()),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Reservoir-computing library benchmark")
    p.add_argument("--quick", action="store_true", help="trimmed matrix for smoke tests")
    p.add_argument("--only", nargs="*", default=None, help="adapter keys to include")
    p.add_argument("--contexts", nargs="*", default=None, help="context keys to include")
    p.add_argument("--repeats", type=int, default=3, help="timed repeats per cell")
    p.add_argument("--warmups", type=int, default=1, help="discarded warmup iterations")
    p.add_argument("--out", type=Path, default=_DEFAULT_OUT, help="results JSON path")
    args = p.parse_args(argv)

    contexts = QUICK_CONTEXTS if args.quick else CONTEXTS
    if args.contexts:
        contexts = tuple(c for c in contexts if c.key in args.contexts)

    hp = HParams()
    adapters = all_adapters(only=args.only)
    available = []
    _p("Adapters:")
    for a in adapters:
        ok, reason = a.is_available()
        _p(f"  {a.key:20s} {'ok' if ok else f'SKIP ({reason})'}")
        if ok:
            available.append(a)
    if not available:
        _p("No adapters available.", file=sys.stderr)
        return 1

    need = 0
    for c in contexts:
        for pt in c.points:
            need = max(need, pt.train_len + 1, pt.warmup + pt.horizon + 1)
    t0 = time.perf_counter()
    _p(f"\nGenerating shared Lorenz series of {need} steps...", end=" ")
    series = lorenz(need, seed=hp.seed)
    _p(f"done ({time.perf_counter() - t0:.1f}s)")

    results: dict = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hardware": _hardware(),
            "versions": _collect_versions(),
            "hparams": hp.__dict__,
            "repeats": args.repeats,
            "warmups": args.warmups,
            "series_len": len(series),
            "lyapunov": {"lambda_max": LORENZ_LAMBDA_MAX, "dt": LORENZ_DT},
        },
        "contexts": {},
    }

    for ci, c in enumerate(contexts, 1):
        _p(f"\n=== [{ci}/{len(contexts)}] {c.key}: {c.title}  (metric: {c.metric}) ===")
        ctx_out = {
            "title": c.title,
            "description": c.description,
            "metric": c.metric,
            "axis": c.axis,
            "points": [],
        }
        for pt in c.points:
            cell = {"point": pt.__dict__, "results": {}}
            axis_val = getattr(pt, c.axis)
            _p(f"  {c.axis}={axis_val}  ({pt.label})")
            for a in available:
                _p(f"    {a.label:26s} ", end="")
                cstart = time.perf_counter()
                try:
                    if c.metric == "precision":
                        pred = a.predict_trajectory(series, hp, pt)
                        truth = series[pt.warmup : pt.warmup + len(pred)]
                        vpt = valid_prediction_time(pred, truth, hp.vpt_threshold)
                        cell["results"][a.key] = {
                            "vpt": vpt,
                            "vpt_lyap": lyapunov_times(vpt),
                            "horizon": pt.horizon,
                        }
                        wall = time.perf_counter() - cstart
                        _p(
                            f"VPT {vpt:5d} steps ({lyapunov_times(vpt):4.1f} Λt)   (wall {wall:5.1f}s)"
                        )
                    else:
                        fn = a.time_train if c.metric == "train" else a.time_forecast
                        rr = fn(series, hp, pt, args.repeats, args.warmups)
                        cell["results"][a.key] = rr.as_dict()
                        wall = time.perf_counter() - cstart
                        extra = f"  rmse {rr.rmse:.3f}" if rr.rmse is not None else ""
                        _p(f"{rr.median * 1e3:9.1f} ms{extra}   (wall {wall:5.1f}s)")
                except Exception as e:
                    cell["results"][a.key] = {"error": str(e)}
                    _p(f"ERROR: {str(e)[:120]}")
            ctx_out["points"].append(cell)
        results["contexts"][c.key] = ctx_out
        # checkpoint after each context so a Ctrl-C keeps finished work
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))

    _p(f"\nWrote {args.out}  (total {time.perf_counter() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
