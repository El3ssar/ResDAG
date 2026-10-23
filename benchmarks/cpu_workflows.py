"""Paired, fixed-weight CPU workflows in isolated source revisions.

Run from the repository, with its benchmark dependencies installed::

    uv run python benchmarks/cpu_workflows.py --before 041db61 --after HEAD \
        --artifacts-dir /tmp/resdag-cpu-workflows

``--after WORKTREE`` snapshots current source, including uncommitted changes.
The default seven-row matrix uses synthetic data only. Four 400-unit workflows
use 1,501 input steps, 100 washout steps and a 257-step forecast. Three 1,000-unit
classic workflows use batches 1/4/16 and shorter 501/129-step workloads. Each
fresh process runs every selected row; two ABBA blocks give four processes per
revision, each with seven timed repetitions after one discarded warmup.

No numerical packages are imported by the supervisor or argument parser. This
tool measures eager ``rd.Model.fit`` and original-unit history-plus-forecast,
including public workflow staging/preprocessing, without profiling hooks. It is
neither an accuracy search nor a cold-compilation benchmark. Existing benchmark
thresholds are not used or changed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import io
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CASE_NAMES = (
    "classic-400-b1",
    "ott-400-b1",
    "cycle-400-b1",
    "er-400-b1",
    "classic-1000-b1",
    "classic-1000-b4",
    "classic-1000-b16",
)
STORAGE_CASE_NAMES = tuple(
    f"er-{width}-b1-{storage}" for width in (400, 1000, 2000) for storage in ("dense", "csr")
)


class BaselineWeightMismatchError(RuntimeError):
    """Keep complete expected/actual fingerprints in a failed worker's receipt."""

    def __init__(self, case: dict, expected: dict, actual: dict, artifact: dict):
        super().__init__(f"Baseline weights were not preserved: {case['id']}")
        self.receipt = {
            "case": case,
            "expected_initial_model": expected,
            "actual_initial_model": actual,
            "weight_artifact": artifact,
        }


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _manifest(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): _digest(path.read_bytes())
        for path in sorted(directory.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }


def _snapshot(reference: str, destination: Path) -> dict:
    commit = subprocess.check_output(
        [
            "git",
            "rev-parse",
            "--verify",
            f"{reference if reference != 'WORKTREE' else 'HEAD'}^{{commit}}",
        ],
        cwd=ROOT,
        text=True,
    ).strip()
    with tempfile.TemporaryDirectory(prefix="resdag-cpu-source-") as temporary:
        staging = Path(temporary)
        if reference == "WORKTREE":
            shutil.copytree(
                ROOT / "src/resdag",
                staging / "src/resdag",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        else:
            archive = subprocess.check_output(["git", "archive", commit, "src/resdag"], cwd=ROOT)
            with tarfile.open(fileobj=io.BytesIO(archive)) as source:
                if hasattr(tarfile, "data_filter"):
                    source.extractall(staging, filter="data")
                else:  # Older supported Python 3.11 patch releases.
                    for member in source.getmembers():
                        path = Path(member.name)
                        if (
                            path.is_absolute()
                            or ".." in path.parts
                            or not (member.isfile() or member.isdir())
                        ):
                            raise ValueError("Git source archive contains unsafe members")
                    source.extractall(staging)
        shutil.copytree(staging, destination)
    files = _manifest(destination)
    return {
        "requested_ref": reference,
        "head_commit": commit,
        "snapshot_kind": "working source" if reference == "WORKTREE" else "git archive",
        "files": files,
        "manifest_sha256": _digest(json.dumps(files, sort_keys=True).encode()),
    }


def _summary(values: list[float]) -> dict[str, float]:
    if not values or any(not (0 < value < float("inf")) for value in values):
        raise ValueError("Timings and ratios must be finite and positive")
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _cases(args: argparse.Namespace) -> list[dict]:
    smoke = args.profile == "smoke"
    storage = args.profile == "storage"
    rows = []
    for name in args.only or (STORAGE_CASE_NAMES if storage else CASE_NAMES):
        parts = name.split("-")
        family, width, batch = parts[:3]
        scaling = storage or width == "1000"
        rows.append(
            {
                "id": name,
                "family": family,
                "reservoir_size": (64 if scaling else 32) if smoke else int(width),
                "batch": int(batch[1:]),
                "features": 3,
                "seed": 20261023,
                "sparse": parts[3] == "csr" if storage else family in {"cycle", "er"},
                "input_steps": args.scaling_train_steps if scaling else args.train_steps,
                "washout": args.washout,
                "history_steps": args.history_steps,
                "horizon": args.scaling_horizon if scaling else args.horizon,
                "artifact_key": (
                    f"{family}-{64 if scaling else 32}" if smoke else f"{family}-{width}"
                ),
            }
        )
        if storage:
            rows[-1].update(storage=parts[3], storage_pair=f"er-{width}-b1")
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--before")
    parser.add_argument(
        "--after", default="HEAD", help="git ref, or WORKTREE for current saved source"
    )
    parser.add_argument("--artifacts-dir", type=Path)
    parser.add_argument(
        "--output", type=Path, help="exclusive JSON output; default ARTIFACTS/results.json"
    )
    parser.add_argument("--profile", choices=("full", "smoke", "storage"), default="full")
    parser.add_argument("--only", nargs="+", choices=CASE_NAMES + STORAGE_CASE_NAMES)
    for name in (
        "blocks",
        "repeats",
        "warmups",
        "train-steps",
        "scaling-train-steps",
        "washout",
        "history-steps",
        "horizon",
        "scaling-horizon",
    ):
        parser.add_argument(f"--{name}", type=int)
    parser.add_argument(
        "--reservoir-forward",
        action="store_true",
        help="also time reset + public reservoir forward",
    )
    parser.add_argument(
        "--timeout", type=float, default=1800, help="maximum seconds for each serialized subprocess"
    )
    parser.add_argument("--worker", choices=("prepare", "measure"), help=argparse.SUPPRESS)
    parser.add_argument("--label", choices=("before", "after"), help=argparse.SUPPRESS)
    parser.add_argument("--run-id", help=argparse.SUPPRESS)
    return parser


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.worker:
        if args.artifacts_dir is None or args.output is None or args.label is None:
            parser.error("worker needs artifacts directory, output and label")
        return args
    if args.before is None or args.artifacts_dir is None:
        parser.error("--before and --artifacts-dir are required")
    allowed = STORAGE_CASE_NAMES if args.profile == "storage" else CASE_NAMES
    if args.only and not set(args.only).issubset(allowed):
        parser.error("--only rows must belong to the selected profile")
    defaults = dict(
        blocks=2,
        repeats=7,
        warmups=1,
        train_steps=1501,
        scaling_train_steps=501,
        washout=100,
        history_steps=100,
        horizon=257,
        scaling_horizon=129,
    )
    if args.profile == "smoke":
        defaults.update(
            blocks=1,
            repeats=2,
            train_steps=81,
            scaling_train_steps=41,
            washout=10,
            history_steps=10,
            horizon=17,
            scaling_horizon=17,
        )
    for key, default in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, default)
        minimum = 0 if key == "warmups" else 1
        if getattr(args, key) < minimum:
            parser.error(f"--{key.replace('_', '-')} must be >= {minimum}")
    if args.washout >= min(args.train_steps, args.scaling_train_steps):
        parser.error("washout must leave supervised training inputs in both profiles")
    if args.history_steps > min(args.train_steps, args.scaling_train_steps) + 1:
        parser.error("history must fit in the synthetic observations")
    if not (0 < args.timeout < float("inf")):
        parser.error("timeout must be finite and positive")
    if args.only and len(args.only) != len(set(args.only)):
        parser.error("--only must not contain duplicates")
    return args


def _thread_module(directory: Path):
    spec = importlib.util.spec_from_file_location(
        "cpu_workflow_threads", directory / "thread_policy.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load frozen benchmark thread policy")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tensor_fingerprint(value) -> dict:
    value = value.detach().cpu()
    if value.layout != __import__("torch").strided:
        value = value.to_dense()
    value = value.contiguous()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": _digest(value.numpy().tobytes()),
    }


def _fingerprints(values: dict) -> dict:
    tensors = {name: _tensor_fingerprint(value) for name, value in sorted(values.items())}
    return {"tensors": tensors, "sha256": _digest(json.dumps(tensors, sort_keys=True).encode())}


def _model_fingerprints(model, rd, *, logical_recurrence: bool = False) -> dict:
    # Compare logical weights across historical dense-plus-CSR and compact-only
    # representations. All other persistent tensors, including fitted flags and
    # readout coefficients, retain their ordinary state-dict names.
    state = model.state_dict()
    values = {name: value for name, value in state.items() if hasattr(value, "detach")}
    for name, layer in model.named_modules():
        if isinstance(layer, rd.BaseReservoirLayer):
            prefix = f"{name}.cell." if name else "cell."
            for suffix in (
                "weight_hh",
                "_weight_hh_csr",
                "_hh_crow_indices",
                "_hh_col_indices",
                "_hh_values",
                "_hh_single_columns",
            ):
                values.pop(prefix + suffix, None)
            recurrent = layer.cell.weight_hh
            if logical_recurrence:
                # CSR omits structural zeros, losing the sign of dense -0.0.
                # Normalize only this initial logical-operator comparison.
                # Fitted tensors, predictions, states and saved artifact bytes
                # retain their ordinary exact fingerprints.
                if str(recurrent.layout) != "torch.strided":
                    recurrent = recurrent.to_dense()
                recurrent = recurrent.detach().masked_fill(recurrent == 0, 0.0)
            values[prefix + "weight_hh"] = recurrent
    result = _fingerprints(values)
    result["extra_state"] = {
        name: _canonical_metadata(value)
        for name, value in sorted(state.items())
        if not hasattr(value, "detach")
    }
    result["metadata_schema_version"] = 1
    result["initial_recurrent_zero_sign_canonicalization"] = logical_recurrence
    result["sha256"] = _digest(
        json.dumps(
            {
                "tensors": result["tensors"],
                "extra_state": result["extra_state"],
                "metadata_schema_version": 1,
                "initial_recurrent_zero_sign_canonicalization": logical_recurrence,
            },
            sort_keys=True,
            allow_nan=False,
        ).encode()
    )
    return result


def _canonical_metadata(value):
    """Retain checkpoint configuration and nested tensors without repr/address hashes."""
    if hasattr(value, "detach"):
        return {"tensor_fingerprint": _tensor_fingerprint(value)}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [_canonical_metadata(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _canonical_metadata(item) for key, item in sorted(value.items())}
    raise TypeError(f"Unsupported checkpoint metadata: {type(value).__name__}")


def _build(case: dict, rd, torch, *, reference: bool = False):
    topology = None
    if case["family"] == "cycle":
        topology = ("simple_cycle", {"weight": 0.7})
    elif case["family"] == "er":
        topology = ("erdos_renyi", {"p": 6 / case["reservoir_size"]})
    torch.manual_seed(case["seed"])
    builder = rd.ott_esn if case["family"] == "ott" else rd.classic_esn
    return builder(
        reservoir_size=case["reservoir_size"],
        feedback_size=case["features"],
        output_size=case["features"],
        topology=topology,
        # Only reference preparation normalizes the actual retained operator.
        # Measurement workers immediately load that checkpoint, so normalizing
        # disposable placeholder weights would add irrelevant eigensolver work
        # and can fail (notably ARPACK on a large simple cycle) before loading.
        spectral_radius=0.7 if reference else None,
        leak_rate=0.8,
        noise=0.0,
        feedback_initializer=("uniform", {"input_scaling": 0.2}),
        readout="ridge",
        readout_alpha=1e-3,
        readout_bias=True,
        sparse=False if reference else case["sparse"],
        seed=case["seed"],
        dtype=torch.float64,
    ).eval()


def _prepare(directory: Path, protocol: dict, rd, torch) -> dict:
    artifacts = {"weights": {}, "data": {}}
    for case in protocol["cases"]:
        key = case["artifact_key"]
        if key not in artifacts["weights"]:
            model = _build(case, rd, torch, reference=True)
            path = directory / "artifacts" / f"weights-{key}.pt"
            torch.save(model.state_dict(), path)
            artifacts["weights"][key] = {
                "path": str(path.relative_to(directory)),
                "file_sha256": _digest(path.read_bytes()),
                "logical_model": _model_fingerprints(model, rd, logical_recurrence=True),
            }
        # Generate once with the baseline environment, then load exactly these
        # values in both revisions. No trajectories or final accuracy banks.
        t = torch.arange(case["input_steps"] + 1, dtype=torch.float64)[None, :, None]
        channel = torch.arange(1, case["features"] + 1, dtype=torch.float64)[None, None, :]
        batch = torch.arange(case["batch"], dtype=torch.float64)[:, None, None]
        observations = torch.sin(t * (0.013 * channel) + batch * 0.17)
        observations = observations + 0.1 * torch.cos(t * 0.037 + channel + batch * 0.11)
        path = directory / "artifacts" / f"data-{case['id']}.pt"
        torch.save(observations, path)
        artifacts["data"][case["id"]] = {
            "path": str(path.relative_to(directory)),
            "file_sha256": _digest(path.read_bytes()),
            "observations": _tensor_fingerprint(observations),
        }
    _write(directory / "artifacts.json", artifacts)
    return artifacts


def _checked_load(directory: Path, entry: dict, torch):
    path = directory / entry["path"]
    if _digest(path.read_bytes()) != entry["file_sha256"]:
        raise RuntimeError(f"Artifact changed: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def _measure(directory: Path, protocol: dict, rd, torch, threads) -> list[dict]:
    artifacts = json.loads((directory / "artifacts.json").read_text())
    rows = []
    storage_predictions = {}
    for case in protocol["cases"]:
        weight = artifacts["weights"][case["artifact_key"]]
        observations = _checked_load(directory, artifacts["data"][case["id"]], torch)
        graph = _build(case, rd, torch)
        graph.load_state_dict(_checked_load(directory, weight, torch), strict=True)
        initial = _model_fingerprints(graph, rd, logical_recurrence=True)
        if initial != weight["logical_model"]:
            raise BaselineWeightMismatchError(case, weight["logical_model"], initial, weight)
        model = rd.Model(graph=graph, washout=case["washout"]).eval()
        history = observations[:, -case["history_steps"] :]
        repeats, reference = [], None
        for iteration in range(protocol["warmups"] + protocol["repeats"]):
            fitted = copy.deepcopy(model)
            threads.verify_thread_policy()
            started = time.perf_counter()
            fitted.fit(observations)
            fit_seconds = time.perf_counter() - started
            threads.verify_thread_policy()
            fitted_hash = _model_fingerprints(fitted, rd)
            fit_state = _fingerprints(fitted.get_state())
            threads.verify_thread_policy()
            started = time.perf_counter()
            prediction = fitted.forecast(history=history, horizon=case["horizon"], compile=False)
            forecast_seconds = time.perf_counter() - started
            threads.verify_thread_policy()
            if prediction.shape != (case["batch"], case["horizon"], case["features"]):
                raise RuntimeError(f"Wrong complete forecast shape: {case['id']}")
            if not bool(torch.isfinite(prediction).all()):
                raise RuntimeError(f"Nonfinite synthetic forecast: {case['id']}")
            check = {
                "fitted_model": fitted_hash["sha256"],
                "fit_state": fit_state["sha256"],
                "prediction": _tensor_fingerprint(prediction)["sha256"],
                "forecast_state": _fingerprints(fitted.get_state())["sha256"],
            }
            timings = {"fit_seconds": fit_seconds, "history_forecast_seconds": forecast_seconds}
            if protocol["reservoir_forward"]:
                reservoir = next(
                    layer
                    for layer in copy.deepcopy(graph).modules()
                    if isinstance(layer, rd.BaseReservoirLayer)
                )
                threads.verify_thread_policy()
                started = time.perf_counter()
                reservoir.reset_state()
                forward = reservoir(observations[:, :-1])
                timings["reservoir_forward_seconds"] = time.perf_counter() - started
                threads.verify_thread_policy()
                check["reservoir_forward"] = _tensor_fingerprint(forward)["sha256"]
                check["reservoir_forward_state"] = _tensor_fingerprint(reservoir.state)["sha256"]
            if reference is not None and reference != check:
                raise RuntimeError(f"Repeated execution changed numerical results: {case['id']}")
            reference = check
            repeats.append(
                {
                    "iteration": iteration,
                    "warmup": iteration < protocol["warmups"],
                    "timings": timings,
                    "hashes": check,
                }
            )
        timed = [row for row in repeats if not row["warmup"]]
        rows.append(
            {
                "case": case,
                "initial_model": initial,
                "fitted_model": fitted_hash,
                "runs": repeats,
                "summary": {
                    metric: _summary([row["timings"][metric] for row in timed])
                    for metric in timed[0]["timings"]
                },
            }
        )
        if "storage_pair" in case:
            pair = case["storage_pair"]
            previous = storage_predictions.get(pair)
            if previous is not None:
                maximum = float((prediction - previous["prediction"]).abs().max())
                rows[-1]["storage_pair_diagnostic"] = {
                    "pair": pair,
                    "case_ids": [previous["case_id"], case["id"]],
                    "initial_logical_weights_equal": initial == previous["initial_model"],
                    "cross_storage_hash_agreement": {
                        key: reference[key] == previous["hashes"][key] for key in reference
                    },
                    "complete_forecast_max_abs_difference": (
                        maximum if maximum < float("inf") else None
                    ),
                    "exact_cross_storage_agreement_required": False,
                    "scope": "Last complete prediction per case, compared outside timing; arithmetic/storage diagnostic only",
                }
            else:
                storage_predictions[pair] = {
                    "case_id": case["id"],
                    "initial_model": initial,
                    "prediction": prediction.detach().clone(),
                    "hashes": reference,
                }
    return rows


def _worker(args: argparse.Namespace) -> None:
    directory = args.artifacts_dir.resolve()
    threads = _thread_module(directory)
    threads.bootstrap_cli()  # stdlib-only restart, before any numerical import
    protocol = json.loads((directory / "protocol.json").read_text())
    source = directory / "sources" / args.label
    if _manifest(source) != protocol["sources"][args.label]["files"]:
        raise RuntimeError("Source snapshot changed before worker start")
    sys.path.insert(0, str(source / "src"))
    result = {"label": args.label, "run_id": args.run_id, "mode": args.worker}
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        try:
            with threads.single_thread_policy() as policy:
                import numpy as np
                import scipy
                import torch

                import resdag as rd

                with torch.no_grad():
                    result["rows"] = (
                        _prepare(directory, protocol, rd, torch)
                        if args.worker == "prepare"
                        else _measure(directory, protocol, rd, torch, threads)
                    )
                threads.verify_thread_policy()
                result["native_thread_policy"] = policy
                result["environment"] = {
                    "python": sys.version,
                    "executable": sys.executable,
                    "platform": platform.platform(),
                    "machine": platform.machine(),
                    "cpu_count": os.cpu_count(),
                    "versions": {
                        "torch": torch.__version__,
                        "numpy": np.__version__,
                        "scipy": scipy.__version__,
                        "resdag": rd.__version__,
                    },
                }
                origins = {
                    name: str(Path(module.__file__).resolve())
                    for name, module in sys.modules.items()
                    if (name == "resdag" or name.startswith("resdag."))
                    and getattr(module, "__file__", None)
                }
                if not origins or not all(
                    Path(path).is_relative_to(source / "src") for path in origins.values()
                ):
                    raise RuntimeError("Mixed source revisions imported")
                result["loaded_resdag_modules"] = origins
            if _manifest(source) != protocol["sources"][args.label]["files"]:
                raise RuntimeError("Source snapshot changed during measurement")
        except BaseException as error:
            result["error"] = {"type": type(error).__name__, "message": str(error)}
            if isinstance(error, BaselineWeightMismatchError):
                result["failure_receipt"] = error.receipt
            raise
        finally:
            result["warnings"] = [
                {
                    "category": warning.category.__name__,
                    "message": str(warning.message),
                    "filename": warning.filename,
                    "lineno": warning.lineno,
                }
                for warning in recorded
            ]
            _write(args.output, result)


def _compare(processes: list[dict], cases: list[dict], blocks: int) -> dict:
    by_id = {process["run_id"]: process for process in processes}
    expected = {
        f"block{block}-slot{slot}": label
        for block in range(blocks)
        for slot, label in enumerate(("before", "after", "after", "before"))
    }
    if len(processes) != len(expected) or set(by_id) != set(expected):
        raise ValueError("Missing or duplicate ABBA process slots")
    if any(by_id[run_id]["label"] != label for run_id, label in expected.items()):
        raise ValueError("ABBA revision order was not preserved")
    case_ids = {case["id"] for case in cases}
    if not cases or len(case_ids) != len(cases):
        raise ValueError("Cases must be nonempty and unique")
    for process in processes:
        if (
            len(process["rows"]) != len(cases)
            or {row["case"]["id"] for row in process["rows"]} != case_ids
        ):
            raise ValueError("Worker rows do not cover the selected cases exactly")
    result = {}
    for case in cases:
        rows = {
            process["run_id"]: next(
                row for row in process["rows"] if row["case"]["id"] == case["id"]
            )
            for process in processes
        }
        all_checks = [run["hashes"] for row in rows.values() for run in row["runs"]]
        required = {"fitted_model", "fit_state", "prediction", "forecast_state"}
        if not all_checks or not required.issubset(all_checks[0]):
            raise ValueError("Missing numerical fingerprints")
        if any(set(check) != set(all_checks[0]) for check in all_checks):
            raise ValueError("Numerical fingerprint schemas differ")
        agreement = {key: len({check[key] for check in all_checks}) == 1 for key in all_checks[0]}
        metrics = {}
        for metric in next(iter(rows.values()))["summary"]:
            ratios, block_ratios = [], []
            for block in range(blocks):
                # Pair nearest AB and BA process slots, never pool repeats as
                # if each were an independent process-level observation.
                paired = [
                    rows[f"block{block}-slot0"]["summary"][metric]["median"]
                    / rows[f"block{block}-slot1"]["summary"][metric]["median"],
                    rows[f"block{block}-slot3"]["summary"][metric]["median"]
                    / rows[f"block{block}-slot2"]["summary"][metric]["median"],
                ]
                ratios.extend(paired)
                block_ratios.append(statistics.median(paired))
            medians = {
                label: [
                    row["summary"][metric]["median"]
                    for run_id, row in rows.items()
                    if by_id[run_id]["label"] == label
                ]
                for label in ("before", "after")
            }
            metrics[metric] = {
                "process_medians_seconds": medians,
                "paired_process_ratios_before_over_after": ratios,
                "ratio_summary": _summary(ratios),
                "block_median_ratios": block_ratios,
                "block_spread": _summary(block_ratios),
            }
        result[case["id"]] = {
            "case": case,
            "exact_numerical_agreement": agreement,
            "all_hashes_equal": all(agreement.values()),
            "metrics": metrics,
        }
    return result


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    if args.worker:
        _worker(args)
        return 0
    directory = args.artifacts_dir.resolve()
    output = args.output.resolve() if args.output else directory / "results.json"
    reserved = {"protocol.json", "artifacts.json", "runner.py", "thread_policy.py"}
    if output == directory or (
        output.is_relative_to(directory) and (output.parent != directory or output.name in reserved)
    ):
        raise ValueError("Output inside the artifact directory must be a new top-level report file")
    if output.exists():
        raise FileExistsError(output)
    directory.mkdir(parents=True, exist_ok=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    (directory / "artifacts").mkdir()
    (directory / "processes").mkdir()
    (directory / "sources").mkdir()
    shutil.copyfile(Path(__file__), directory / "runner.py")
    shutil.copyfile(ROOT / "benchmarks/rc_bench/_threads.py", directory / "thread_policy.py")
    sources = {
        label: _snapshot(reference, directory / "sources" / label)
        for label, reference in (("before", args.before), ("after", args.after))
    }
    protocol = {
        "schema_version": 1,
        "profile": args.profile,
        "cases": _cases(args),
        "timestamp_utc": subprocess.check_output(
            ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True
        ).strip(),
        "blocks": args.blocks,
        "repeats": args.repeats,
        "warmups": args.warmups,
        "reservoir_forward": args.reservoir_forward,
        "sources": sources,
        "runner_sha256": _digest((directory / "runner.py").read_bytes()),
        "thread_policy_sha256": _digest((directory / "thread_policy.py").read_bytes()),
        "dependencies": {
            name: _digest((ROOT / name).read_bytes())
            for name in ("pyproject.toml", "uv.lock")
            if (ROOT / name).is_file()
        },
        "scheduling": "Serialized fresh-process ABBA blocks; all selected rows in each process",
        "timing_scope": {
            "fit_seconds": "rd.Model.fit(observations), including transactional graph staging, learned preprocessing, washout, features and solve; outer model construction/cloning excluded",
            "history_forecast_seconds": "rd.Model.forecast(history=history,horizon=...,compile=False), including original-unit transforms, reset/history, generated steps and output assembly",
            "reservoir_forward_seconds": "Optional public reservoir reset_state + forward on complete input prefix; no hooks or readout",
        },
        "data": "Baseline-generated synthetic sine/cosine only; input_steps includes washout; observations=input_steps+1; targets shift by one",
        "weight_construction": "Only baseline reference preparation applies spectral_radius=0.7; measurement placeholders use spectral_radius=None before strict checkpoint loading and logical-weight equality checks",
        "initial_weight_comparison": "Only initial logical recurrent weights normalize zero signs because CSR drops structural signed zeros; raw artifact bytes, fitted-model tensors, predictions and states remain exact",
        "inference": "float64 CPU, no_grad, eval, no execution hooks; one verified native thread",
        "claims": "Implementation timing and exact fixed-artifact regression only; no accuracy tuning, universal speed, cold-compile or common memory claim",
    }
    _write(directory / "protocol.json", protocol)

    def run(mode: str, label: str, run_id: str) -> dict:
        result_path = directory / "processes" / f"{run_id}.json"
        command = [
            sys.executable,
            str(directory / "runner.py"),
            "--worker",
            mode,
            "--label",
            label,
            "--run-id",
            run_id,
            "--artifacts-dir",
            str(directory),
            "--output",
            str(result_path),
        ]
        print(f"CPU workflows: {run_id} ({label}, {mode})", flush=True)
        with (
            (directory / "processes" / f"{run_id}.stdout.txt").open("x") as stdout,
            (directory / "processes" / f"{run_id}.stderr.txt").open("x") as stderr,
        ):
            subprocess.run(
                command,
                cwd=directory,
                check=True,
                timeout=args.timeout,
                stdout=stdout,
                stderr=stderr,
            )
        return json.loads(result_path.read_text())

    preparation = run("prepare", "before", "prepare")
    processes = [
        run("measure", label, f"block{block}-slot{slot}")
        for block in range(args.blocks)
        for slot, label in enumerate(("before", "after", "after", "before"))
    ]
    comparisons = _compare(processes, protocol["cases"], args.blocks)
    for label in sources:
        if _manifest(directory / "sources" / label) != sources[label]["files"]:
            raise RuntimeError(f"Source snapshot changed: {label}")
    if (
        _digest((directory / "runner.py").read_bytes()) != protocol["runner_sha256"]
        or _digest((directory / "thread_policy.py").read_bytes())
        != protocol["thread_policy_sha256"]
    ):
        raise RuntimeError("Frozen measurement tools changed during the experiment")
    report = {
        "protocol": protocol,
        "preparation": preparation,
        "processes": processes,
        "comparisons": comparisons,
        "storage_diagnostics": [
            {
                "run_id": process["run_id"],
                "label": process["label"],
                **row["storage_pair_diagnostic"],
            }
            for process in processes
            for row in process["rows"]
            if "storage_pair_diagnostic" in row
        ],
        "all_hashes_equal": all(row["all_hashes_equal"] for row in comparisons.values()),
    }
    _write(output, report)
    print(f"Wrote {output}; exact numerical agreement: {report['all_hashes_equal']}", flush=True)
    return 0 if report["all_hashes_equal"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
