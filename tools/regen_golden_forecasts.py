#!/usr/bin/env python
"""Regenerate (or verify) the golden-forecast regression fixtures (issue #401).

The fixtures under ``tests/fixtures/golden_forecast/`` pin the behavioural
contract of :meth:`resdag.core.ESNModel.forecast` for every premade model: a
fully-seeded warmup → forecast on a canonical Lorenz-63 series, in CPU
``float64``, checked in as the expected trajectory plus a Valid-Prediction-Time
floor.  The model specs, data splits, metric, and tolerances live in
:mod:`tests.test_models.golden_forecast` so this script and the regression test
can never disagree.

Usage
-----
Regenerate every fixture (run after an *intended* forecast-path change)::

    python tools/regen_golden_forecasts.py --all

Regenerate a single model::

    python tools/regen_golden_forecasts.py --model ott_esn

Verify the committed fixtures still reproduce, without writing anything (this is
the same thing the test asserts; handy as a quick local/CI drift check)::

    python tools/regen_golden_forecasts.py --check

``--check`` exits non-zero if any model drifts beyond the committed tolerances,
so it can gate a pre-commit hook or CI step.

This script is intentionally a thin, deterministic CLI over the shared helpers —
running it (especially ``--check``) is the runnable, self-checking example for
this test-infrastructure change.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Make the in-repo ``tests`` package importable when run as a plain script.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.test_models.golden_forecast import (  # noqa: E402
    SPECS,
    GoldenSpec,
    build_payload,
    fixture_path,
    load_fixture,
    make_lorenz_splits,
    save_fixture,
    valid_prediction_time,
)


def _specs_for(names: list[str] | None) -> list[GoldenSpec]:
    """Resolve ``--model`` names to specs, erroring on unknown names."""
    if not names:
        return list(SPECS)
    by_name = {spec.name: spec for spec in SPECS}
    unknown = [n for n in names if n not in by_name]
    if unknown:
        available = ", ".join(s.name for s in SPECS)
        raise SystemExit(f"unknown model(s): {', '.join(unknown)}. Available: {available}")
    return [by_name[n] for n in names]


def _regenerate(specs: list[GoldenSpec]) -> None:
    """Build/train/forecast each spec and overwrite its committed fixture."""
    splits = make_lorenz_splits()
    print(f"{'model':16s} {'vpt':>5s} {'floor':>6s}  fixture")
    for spec in specs:
        payload = build_payload(spec, splits)
        path = fixture_path(spec.name)
        save_fixture(path, payload)
        meta = payload["meta"]
        rel = path.relative_to(_ROOT)
        print(f"{spec.name:16s} {meta['vpt']:5d} {meta['vpt_floor']:6d}  wrote {rel}")
    print(f"\nRegenerated {len(specs)} fixture(s). Review the diff before committing.")


def _check(specs: list[GoldenSpec]) -> int:
    """Verify each committed fixture reproduces; return a process exit code."""
    splits = make_lorenz_splits()
    *_, val = splits
    ok = True
    print(f"{'model':16s} {'prefixΔ':>10s} {'vpt':>5s} {'floor':>6s}  status")
    for spec in specs:
        try:
            golden, meta = load_fixture(spec.name)
        except FileNotFoundError:
            print(f"{spec.name:16s} {'-':>10s} {'-':>5s} {'-':>6s}  MISSING fixture")
            ok = False
            continue

        payload = build_payload(spec, splits)
        live = payload["golden_forecast"]
        k = int(meta["prefix_steps"])
        prefix_delta = float(np.abs(live[:, :k] - golden[:, :k]).max())
        prefix_ok = np.allclose(
            live[:, :k], golden[:, :k], atol=meta["prefix_atol"], rtol=meta["prefix_rtol"]
        )
        live_vpt = valid_prediction_time(
            torch.as_tensor(live),
            val,
            threshold=meta["vpt_threshold"],
        )
        vpt_ok = live_vpt >= meta["vpt_floor"]
        status = "ok" if (prefix_ok and vpt_ok) else "DRIFT"
        if not (prefix_ok and vpt_ok):
            ok = False
        print(
            f"{spec.name:16s} {prefix_delta:10.2e} {live_vpt:5d} {meta['vpt_floor']:6d}  {status}"
        )

    print("\nAll fixtures reproduce." if ok else "\nDRIFT detected — see rows above.")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Returns a process exit code."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="regenerate every fixture")
    group.add_argument(
        "--check",
        action="store_true",
        help="verify committed fixtures reproduce; write nothing; non-zero exit on drift",
    )
    parser.add_argument(
        "--model",
        action="append",
        metavar="NAME",
        help="restrict to a model (repeatable); default is all models",
    )
    args = parser.parse_args(argv)

    specs = _specs_for(args.model)

    if args.check:
        return _check(specs)

    if not (args.all or args.model):
        parser.error("nothing to do: pass --all, --model NAME, or --check")
    _regenerate(specs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
