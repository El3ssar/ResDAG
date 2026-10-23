"""Compare single-pass fitting with an independently imported historical trainer.

Run from the repository with its development dependencies installed::

    uv run python benchmarks/single_pass_training.py --rss --output results.json

Only the baseline trainer comes from ``--baseline-ref``. Both trainers use the
current reservoir/readout implementations, isolating fitting traversal costs.
This is a measurement tool; it introduces no performance thresholds.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

import resdag as rd

ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = ROOT / "src/resdag/training/trainer.py"
METRICS = ("total_ms", "reservoir_ms", "readout_fit_ms", "other_ms")


def _git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _baseline_trainer(reference: str) -> tuple[type, dict[str, str]]:
    resolved = _git("rev-parse", "--verify", f"{reference}^{{commit}}")
    source = subprocess.check_output(
        ["git", "show", f"{resolved}:src/resdag/training/trainer.py"], cwd=ROOT
    )
    with tempfile.TemporaryDirectory(prefix="resdag-baseline-trainer-") as directory:
        path = Path(directory) / "trainer.py"
        path.write_bytes(source)
        name = f"resdag_baseline_trainer_{resolved}"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not import the historical trainer")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module.ESNTrainer, {
        "requested_ref": reference,
        "resolved_commit": resolved,
        "trainer_source_sha256": _sha256(source),
    }


def _workload(depth: int, args: argparse.Namespace) -> tuple[Any, ...]:
    torch.manual_seed(20261023 + depth)
    source = rd.reservoir_input(args.features, dtype=torch.float64)
    node = source
    for index in range(depth):
        memory = rd.ESNLayer(
            args.reservoir_size,
            feedback_size=args.features,
            spectral_radius=0.7,
            leak_rate=0.8,
            noise=0.0,
            seed=500 + index,
            dtype=torch.float64,
        )
        head = rd.RidgeReadoutLayer(
            args.reservoir_size, args.features, alpha=1e-3, name=f"head_{index}"
        ).double()
        node = head(memory(node))
    model = rd.ESNModel(source, node)
    model.reset_reservoirs()
    values = torch.randn(
        args.batch, args.warmup_steps + args.train_steps, args.features, dtype=torch.float64
    )
    warmup, train = values[:, : args.warmup_steps], values[:, args.warmup_steps :]
    targets = {f"head_{index}": train * (0.7 + 0.1 * index) for index in range(depth)}
    return model, warmup, train, targets


def _timed_fit(
    trainer: type, model: Any, warmup: torch.Tensor, train: torch.Tensor, targets: dict
) -> tuple[dict, dict]:
    times = {metric: 0.0 for metric in METRICS}
    counts: dict[str, Any] = {
        "model_calls": [],
        "reservoir_calls": {},
        "reservoir_timesteps": {},
        "reservoir_batch_samples": {},
        "readout_fit_calls": {},
    }
    handles, original_fits = [], []
    starts = {}
    for name, layer in model.named_modules():
        if isinstance(layer, rd.BaseReservoirLayer):
            for key in ("reservoir_calls", "reservoir_timesteps", "reservoir_batch_samples"):
                counts[key][name] = 0

            def begin(module: Any, inputs: tuple, name: str = name) -> None:
                counts["reservoir_calls"][name] += 1
                counts["reservoir_timesteps"][name] += inputs[0].shape[1]
                counts["reservoir_batch_samples"][name] += inputs[0].shape[0] * inputs[0].shape[1]
                starts[name] = time.perf_counter()

            def end(module: Any, inputs: tuple, output: Any, name: str = name) -> None:
                times["reservoir_ms"] += (time.perf_counter() - starts[name]) * 1000

            handles += [layer.register_forward_pre_hook(begin), layer.register_forward_hook(end)]
        if isinstance(layer, rd.ReadoutLayer):
            original = layer.fit
            original_fits.append((layer, original))
            counts["readout_fit_calls"][name] = 0

            def fit(
                states: torch.Tensor,
                labels: torch.Tensor,
                original: Any = original,
                name: str = name,
            ) -> Any:
                counts["readout_fit_calls"][name] += 1
                start = time.perf_counter()
                try:
                    return original(states, labels)
                finally:
                    times["readout_fit_ms"] += (time.perf_counter() - start) * 1000

            layer.fit = fit
    handles.append(
        model.register_forward_pre_hook(
            lambda module, inputs: counts["model_calls"].append(list(inputs[0].shape))
        )
    )
    torch.manual_seed(1000)
    start = time.perf_counter()
    try:
        trainer(model).fit(warmup, train, targets)
    finally:
        times["total_ms"] = (time.perf_counter() - start) * 1000
        for handle in handles:
            handle.remove()
        for layer, original in original_fits:
            layer.fit = original
    times["other_ms"] = times["total_ms"] - times["reservoir_ms"] - times["readout_fit_ms"]
    for field in ("reservoir_calls", "reservoir_timesteps", "reservoir_batch_samples"):
        counts[f"{field}_total"] = sum(counts[field].values())
    counts["readout_fits_total"] = sum(counts["readout_fit_calls"].values())
    return times, counts


def _compare(left: Any, right: Any) -> dict[str, float]:
    coefficient_names = {
        f"{path}.{name}"
        for path, layer in left.named_modules()
        if isinstance(layer, rd.ReadoutLayer)
        for name, _ in layer.named_parameters(recurse=False)
    }
    coefficient_abs = coefficient_relative = model_abs = state_abs = 0.0
    for name, value in left.state_dict().items():
        other = right.state_dict()[name]
        if value.is_floating_point():
            difference = (value - other).abs()
            model_abs = max(model_abs, float(difference.max()))
            if name in coefficient_names:
                coefficient_abs = max(coefficient_abs, float(difference.max()))
                coefficient_relative = max(
                    coefficient_relative, float((difference / value.abs().clamp_min(1e-12)).max())
                )
            torch.testing.assert_close(value, other, rtol=1e-8, atol=1e-10)
        else:
            assert torch.equal(value, other), name
    for name, value in left.get_state().items():
        other = right.get_state()[name]
        state_abs = max(state_abs, float((value - other).abs().max()))
        torch.testing.assert_close(value, other, rtol=1e-8, atol=1e-10)
    return {
        "max_coefficient_abs_difference": coefficient_abs,
        "max_coefficient_relative_difference": coefficient_relative,
        "max_model_tensor_abs_difference": model_abs,
        "max_final_state_abs_difference": state_abs,
    }


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _peak_rss() -> dict[str, float | str]:
    import resource

    native = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "native_ru_maxrss": native,
        "native_unit": "bytes" if sys.platform == "darwin" else "KiB",
        "peak_rss_mib": native / (1024 * 1024 if sys.platform == "darwin" else 1024),
    }


def _rss_cell(mode: str, depth: int, trainers: dict[str, type], args: argparse.Namespace) -> dict:
    model, warmup, train, targets = _workload(depth, args)
    before = _peak_rss()
    times, _ = _timed_fit(trainers[mode], model, warmup, train, targets)
    return {
        **_peak_rss(),
        "peak_rss_mib_before_fit": before["peak_rss_mib"],
        "fit_total_ms": times["total_ms"],
    }


def _fresh_rss(mode: str, depth: int, baseline: str, args: argparse.Namespace) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--baseline-ref",
        baseline,
        "--rss-cell",
        mode,
        "--depths",
        str(depth),
    ]
    for option in ("batch", "reservoir_size", "features", "warmup_steps", "train_steps", "threads"):
        command += ["--" + option.replace("_", "-"), str(getattr(args, option))]
    completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="c0c61d0")
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--pairs", type=int, default=7)
    parser.add_argument("--warmup-pairs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--reservoir-size", type=int, default=64)
    parser.add_argument("--features", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--rss", action="store_true", help="Measure fresh-process peak RSS cells")
    parser.add_argument("--output", type=Path, help="Save JSON here; otherwise print JSON")
    parser.add_argument("--rss-cell", choices=("baseline", "current"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if any(depth < 1 for depth in args.depths):
        parser.error("depths must be positive")
    for name in ("pairs", "batch", "reservoir_size", "features", "train_steps", "threads"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup_pairs < 0 or args.warmup_steps < 0:
        parser.error("warmup counts must be nonnegative")
    if args.rss_cell and len(args.depths) != 1:
        parser.error("an RSS cell needs exactly one depth")
    return args


def main() -> None:
    args = _arguments()
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = str(args.threads)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    historical, baseline = _baseline_trainer(args.baseline_ref)
    trainers = {"baseline": historical, "current": rd.ESNTrainer}
    if args.rss_cell:
        print(json.dumps(_rss_cell(args.rss_cell, args.depths[0], trainers, args)))
        return
    report: dict[str, Any] = {
        "schema_version": 1,
        "baseline": baseline,
        "current": {
            "head_commit": _git("rev-parse", "HEAD"),
            "trainer_source_sha256_before": _sha256(TRAINER_PATH.read_bytes()),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "dtype": "float64",
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        },
        "workload": {
            key: getattr(args, key)
            for key in ("batch", "reservoir_size", "features", "warmup_steps", "train_steps")
        },
        "measurement": {
            "discarded_pairs": args.warmup_pairs,
            "timed_pairs": args.pairs,
            "scheduling": "Interleaved pairs; baseline first on even pairs, current first on odd pairs",
            "initialization": "Fresh identical model deepcopies and identical data/targets; preparation excluded",
            "baseline_scope": "Only historical trainer source differs; both use current reservoir/readout implementations",
            "readout_fit_ms": "Entire readout.fit call, including validation, solve and parameter copies; not isolated solver time",
            "reservoir_ms": "Sum of ordinary reservoir forward durations, measured with hooks",
            "other_ms": "Per-run residual: total minus reservoir minus readout fitting",
            "rss": "When requested, absolute process peak RSS from one fit in a fresh process per mode/depth; includes imports and allocator state",
            "noise": 0.0,
        },
        "depths": {},
    }
    for depth in args.depths:
        base, warmup, train, targets = _workload(depth, args)
        results: dict[str, Any] = {mode: {"runs_ms": []} for mode in trainers}
        parity = {}
        for pair in range(args.warmup_pairs + args.pairs):
            order = ("baseline", "current") if pair % 2 == 0 else ("current", "baseline")
            models = {mode: copy.deepcopy(base) for mode in order}
            for mode in order:
                times, counts = _timed_fit(trainers[mode], models[mode], warmup, train, targets)
                if pair >= args.warmup_pairs:
                    results[mode]["runs_ms"].append(times)
                    if "counts" in results[mode]:
                        assert (
                            counts == results[mode]["counts"]
                        ), "Structural counts changed between fits"
                    results[mode]["counts"] = counts
            for metric, difference in _compare(models["baseline"], models["current"]).items():
                parity[metric] = max(parity.get(metric, 0.0), difference)
        for result in results.values():
            result["summary_ms"] = {
                metric: _summary([run[metric] for run in result["runs_ms"]]) for metric in METRICS
            }
        speedups = [
            before["total_ms"] / after["total_ms"]
            for before, after in zip(results["baseline"]["runs_ms"], results["current"]["runs_ms"])
        ]
        entry = {
            **results,
            "numerical_parity": parity,
            "paired_speedup": {"runs": speedups, **_summary(speedups)},
        }
        if args.rss:
            entry["rss"] = {
                mode: _fresh_rss(mode, depth, baseline["resolved_commit"], args)
                for mode in trainers
            }
        report["depths"][str(depth)] = entry
        print(
            f"Depth {depth}: median paired speedup {statistics.median(speedups):.3f}x",
            file=sys.stderr,
        )
    report["current"]["trainer_source_sha256_after"] = _sha256(TRAINER_PATH.read_bytes())
    report["current"]["working_source_unchanged_during_run"] = (
        report["current"]["trainer_source_sha256_before"]
        == report["current"]["trainer_source_sha256_after"]
    )
    if not report["current"]["working_source_unchanged_during_run"]:
        raise RuntimeError("Current trainer source changed during the measurement")
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
        print(f"Wrote {args.output}")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
