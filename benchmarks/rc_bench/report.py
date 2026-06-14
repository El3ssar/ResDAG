"""Turn a results JSON into Markdown speedup tables.

Speedups are relative to reservoirpy (the de-facto reference library), so
reservoirpy is always 1.00x and a value >1 means *faster than reservoirpy*.

    python -m rc_bench.report                       # -> stdout
    python -m rc_bench.report --out results/REPORT.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_DEFAULT_IN = Path(__file__).resolve().parents[1] / "results" / "latest.json"

# Display order and labels; baseline must come first.
_BASELINE = "reservoirpy"
_COLUMNS = [
    ("reservoirpy", "reservoirpy (CPU)"),
    ("resdag-cpu", "resdag (CPU)"),
    ("resdag-gpu", "resdag (GPU)"),
    ("rcjl", "ReservoirComputing.jl (CPU)"),
]


def _median(cell_results: dict, key: str) -> float | None:
    r = cell_results.get(key)
    if not r or "error" in r or not r.get("times"):
        return None
    return r.get("median")


def _fmt_time(seconds: float) -> str:
    ms = seconds * 1e3
    if ms >= 1000:
        return f"{ms / 1000:.2f} s"
    return f"{ms:.1f} ms"


def _cell(median: float | None, baseline: float | None, is_baseline: bool) -> str:
    if median is None:
        return "—"
    if baseline is None or median <= 0:
        return _fmt_time(median)
    speedup = baseline / median
    if is_baseline:
        return f"{_fmt_time(median)} (1.00×)"
    return f"{_fmt_time(median)} (**{speedup:.1f}×**)"


def _vpt_cell(cell_results: dict, key: str) -> str:
    r = cell_results.get(key)
    if not r or "error" in r or r.get("vpt") is None:
        return "—"
    return f"{r['vpt']} steps ({r['vpt_lyap']:.1f} Λt)"


def render_tables(results: dict, present_keys: set[str] | None = None) -> str:
    """Per-context tables: timing contexts show speedups vs reservoirpy;
    the precision context shows valid prediction time (higher is better)."""
    cols = [(k, lbl) for k, lbl in _COLUMNS if present_keys is None or k in present_keys]
    axis_label = {
        "reservoir_size": "Reservoir size",
        "train_len": "Train length",
        "horizon": "Horizon",
    }
    lines: list[str] = []
    for ckey, ctx in results.get("contexts", {}).items():
        lines.append(f"\n### {ctx['title']}\n")
        if ctx.get("description"):
            lines.append(f"{ctx['description']}\n")
        axis = ctx["axis"]
        label = axis_label.get(axis, axis)
        precision = ctx.get("metric") == "precision"

        lines.append(f"| {label} | " + " | ".join(lbl for _, lbl in cols) + " |")
        lines.append("|" + "---|" * (len(cols) + 1))
        for cell in ctx["points"]:
            pt = cell["point"]
            res = cell["results"]
            row = [str(pt[axis])]
            if precision:
                row += [_vpt_cell(res, k) for k, _ in cols]
            else:
                base = _median(res, _BASELINE)
                row += [_cell(_median(res, k), base, k == _BASELINE) for k, _ in cols]
            lines.append("| " + " | ".join(row) + " |")
        if precision:
            lines.append(
                "\n*Valid prediction time: steps until the normalized forecast "
                "error first exceeds 0.5, also in Lyapunov times (Λt). Higher = "
                "the model stays on the true trajectory longer.*"
            )
        lines.append("")
    return "\n".join(lines)


def render_provenance(results: dict) -> str:
    """Methodology / hardware / versions footer as a bullet list."""
    meta = results.get("meta", {})
    versions = meta.get("versions", {})
    hw = meta.get("hardware", {})
    hp = meta.get("hparams", {})
    lines: list[str] = []
    lines.append(
        f"- **Architecture (identical across libraries):** {hp.get('dim', '?')}-D input, "
        f"spectral radius {hp.get('spectral_radius')}, leak rate {hp.get('leak_rate')}, "
        f"connectivity {hp.get('connectivity')}, ridge {hp.get('ridge')}, tanh activation."
    )
    gpu = versions.get("gpu")
    lines.append(
        f"- **Hardware:** {hw.get('processor') or hw.get('platform', '?')}"
        + (f", {gpu}" if gpu else "")
        + f", {hw.get('cpu_count', '?')} CPU threads."
    )
    vparts = [f"{k} {v}" for k, v in versions.items() if k not in ("gpu", "cuda")]
    lines.append("- **Versions:** " + ", ".join(vparts) + ".")
    lines.append(
        f"- **Method:** median of {meta.get('repeats', '?')} repeats "
        f"(1 warmup discarded), generated {meta.get('timestamp', '?')}."
    )
    return "\n".join(lines)


def render(results: dict, present_keys: set[str] | None = None) -> str:
    """Full standalone report: heading + tables + provenance."""
    meta = results.get("meta", {})
    out = ["## Reservoir-computing library benchmark\n"]
    out.append(
        "Speedups are relative to **reservoirpy** (so reservoirpy = 1.00×; "
        "a value above 1× is *faster than reservoirpy*). Every library runs the "
        "same nominal ESN architecture on a byte-identical Lorenz-63 series; each "
        "cell is the median wall-clock time over "
        f"{meta.get('repeats', '?')} repeats.\n"
    )
    out.append(render_tables(results, present_keys))
    out.append("\n### Setup\n")
    out.append(render_provenance(results))
    out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Render benchmark results as Markdown")
    p.add_argument("--in", dest="inp", type=Path, default=_DEFAULT_IN)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--tables-only", action="store_true", help="emit only the tables")
    args = p.parse_args(argv)

    results = json.loads(args.inp.read_text())
    md = render_tables(results) if args.tables_only else render(results)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md)
        print(f"Wrote {args.out}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
