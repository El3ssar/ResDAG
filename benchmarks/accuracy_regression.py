"""Measure matched forecast accuracy across two isolated Git revisions.

Run from the repository with ``resdag[hpo,dynamics]`` installed::

    python benchmarks/accuracy_regression.py --before 00ca741 --after HEAD \
        --artifacts-dir /tmp/resdag-accuracy --output accuracy.json

The frozen classical recipes are regression workloads, not tuned winners. Both
revisions receive identical raw trajectories and initialization seeds. Every
origin is scored separately over at least ten Lyapunov times; the headline is
its contiguous horizon at NRMSE < 0.3 using fixed training-label scales. The
0.5 threshold and the public truth-window-normalized metric are diagnostics.
Timing is recorded for context, not as a controlled speed comparison. All
predictions, tensor fingerprints and source/import manifests stay in the
artifact directory. This script does not change the working checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tarfile
from importlib.metadata import version
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "00ca741d5fadb0867e65e35a5dc03ffc52ba0117"
SYSTEMS = {
    "lorenz": {
        "dt": 0.02,
        "lambda_max": 0.9056,
        "warmup": 200,
        "train_rows": 2500,
        "raw_samples": 5000,
        "discard_samples": 2000,
        "families": ["classic", "ott"],
        "seeds": [12345, 7, 42],
    },
    "rossler": {
        "dt": 0.145,
        "lambda_max": 0.0714,
        "warmup": 500,
        "train_rows": 8000,
        "raw_samples": 11000,
        "discard_samples": 1000,
        "families": ["classic"],
        "seeds": [42, 101, 102, 103, 104, 105],
    },
}


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _recipe(system: str, seed: int) -> dict[str, Any]:
    options: dict[str, Any] = {
        "reservoir_size": 300 if system == "lorenz" else 400,
        "feedback_size": 3,
        "output_size": 3,
        "spectral_radius": 0.9 if system == "lorenz" else 0.21,
        "readout": "svd",
        "readout_alpha": 0.01 if system == "lorenz" else 3e-10,
        "seed": seed,
    }
    if system == "rossler":
        options.update(
            leak_rate=0.85,
            bias_scaling=0.11,
            feedback_initializer=("random", {"input_scaling": 0.35}),
            topology=("erdos_renyi", {"p": 6 / 400, "directed": True, "self_loops": True}),
        )
    return options


def _generate_data(directory: Path) -> dict[str, Any]:
    import numpy as np
    import tsdynamics as ts

    records = {}
    for name, settings in SYSTEMS.items():
        system = ts.systems.Lorenz() if name == "lorenz" else ts.systems.Rossler()
        count, discard, dt = (
            settings["raw_samples"],
            settings["discard_samples"],
            settings["dt"],
        )
        trajectory = system.run(
            final_time=(count + discard) * dt,
            dt=dt,
            ic=[1.0, 1.0, 1.0],
            solver="dop853",
            rtol=1e-10,
            atol=1e-12,
        )
        raw = np.asarray(trajectory.y[discard : discard + count], dtype=np.float64)
        if raw.shape != (count, 3) or not np.isfinite(raw).all():
            raise RuntimeError(f"Invalid generated {name} trajectory")
        path = directory / f"{name}.npy"
        np.save(path, raw)
        records[name] = {
            "dt": dt,
            "samples": count,
            "discard_samples": discard,
            "parameters": repr(system),
            "initial_condition": [1.0, 1.0, 1.0],
            "solver": "dop853",
            "rtol": 1e-10,
            "atol": 1e-12,
            "shape": list(raw.shape),
            "raw_sha256": _digest(raw.tobytes()),
            "file_sha256": _digest(path.read_bytes()),
            "tsdynamics_version": version("tsdynamics"),
        }
    _write_json(directory / "data.json", records)
    return records


def _archive(reference: str, directory: Path) -> str:
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{reference}^{{commit}}"], cwd=ROOT, text=True
    ).strip()
    payload = subprocess.check_output(["git", "archive", commit, "src/resdag"], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        archive.extractall(directory, filter="data")
    (directory / "commit.txt").write_text(commit + "\n")
    return commit


def _fingerprints(module: Any) -> dict[str, Any]:
    result = {}
    for name, value in list(module.named_parameters()) + list(module.named_buffers()):
        tensor = value.detach().cpu().contiguous()
        result[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": _digest(tensor.numpy().tobytes()),
        }
    return result


def _independent_horizon(prediction: Any, truth: Any, scale: Any, threshold: float) -> int:
    """Small NumPy first-crossing oracle, independent of library score helpers."""
    import numpy as np

    errors = np.sqrt(np.mean(((prediction - truth) / scale) ** 2, axis=-1))
    invalid = np.flatnonzero(~(errors < threshold))
    return int(invalid[0]) if len(invalid) else len(errors)


def _worker(directory: Path, label: str, quick: bool) -> None:
    import time

    import numpy as np
    import torch

    source = directory / label / "src"
    sys.path.insert(0, str(source))
    import resdag as rd
    from resdag.hpo._tuning_scores import score_forecast
    from resdag.metrics import valid_prediction_time

    if not Path(rd.__file__).resolve().is_relative_to(source):
        raise RuntimeError(f"Wrong ResDAG import: {rd.__file__}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rows, forecasts, states, data_hashes = [], {}, {}, {}
    for system, settings in SYSTEMS.items():
        raw = np.load(directory / f"{system}.npy")
        data_hashes[system] = _digest(raw.tobytes())
        data = torch.from_numpy(raw.copy())
        dt, exponent, warmup = settings["dt"], settings["lambda_max"], settings["warmup"]
        observed = warmup + settings["train_rows"] + 1
        horizon = math.ceil(10 / (dt * exponent))
        # Match common tuning's admitted one-step training labels, excluding
        # washout. The scale is fixed before seeing any forecast origin.
        scale = data[warmup + 1 : observed].std(dim=0, correction=0)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        scale = scale.clamp_min(torch.finfo(scale.dtype).eps)
        for family in settings["families"]:
            seeds = settings["seeds"][:1] if quick else settings["seeds"]
            for seed in seeds:
                torch.manual_seed(918273)
                recipe = _recipe(system, seed)
                builder = rd.classic_esn if family == "classic" else rd.ott_esn
                graph = builder(**recipe, dtype=torch.float64)
                model = rd.Model(graph=graph, washout=warmup).eval()
                case = f"{system}-{family}-seed{seed}"
                states[case] = {
                    "initial_graph": _fingerprints(graph),
                    "recipe": recipe,
                    "washout": warmup,
                    "observations": observed,
                }
                started = time.perf_counter()
                model.fit(data[:observed])
                fit_seconds = time.perf_counter() - started
                states[case]["fitted_model"] = _fingerprints(model)
                offsets = [0] if quick else [0, 600, 1200]
                for origin_id, offset in enumerate(offsets):
                    origin = observed + offset
                    truth = data[origin : origin + horizon]
                    if len(truth) != horizon:
                        raise RuntimeError("Ground truth does not cover the full forecast")
                    started = time.perf_counter()
                    prediction = model.forecast(
                        history=data[origin - 500 : origin], horizon=horizon
                    )
                    elapsed = time.perf_counter() - started
                    p, t = prediction.detach().numpy(), truth.numpy()
                    if p.shape != t.shape or not np.isfinite(p).all():
                        raise RuntimeError(f"Invalid forecast for {case}, origin {origin_id}")
                    identifier = f"{case}-origin{origin_id}"
                    forecasts[identifier] = p
                    scores = {}
                    for threshold in (0.3, 0.5):
                        measured = score_forecast(
                            {"prediction": prediction},
                            {"prediction": truth},
                            {"prediction": scale},
                            threshold=threshold,
                        )
                        independent = _independent_horizon(p, t, scale.numpy(), threshold)
                        if measured.score != independent:
                            raise RuntimeError(
                                "Fixed-scale score disagrees with independent oracle"
                            )
                        scores[str(threshold)] = {
                            "steps": measured.score,
                            "lyapunov_times": measured.score * dt * exponent,
                            "right_censored": measured.score == horizon,
                            "full_horizon_nrmse": measured.nrmse,
                        }
                    legacy_steps = valid_prediction_time(
                        prediction[None], truth[None], threshold=0.3
                    )
                    legacy_scale = t.std(axis=0)
                    legacy_scale = np.where(legacy_scale == 0, 1.0, legacy_scale)
                    if legacy_steps != _independent_horizon(p, t, legacy_scale, 0.3):
                        raise RuntimeError("Public metric disagrees with independent oracle")
                    rows.append(
                        {
                            "id": identifier,
                            "case": case,
                            "system": system,
                            "family": family,
                            "seed": seed,
                            "origin_id": origin_id,
                            "origin_index": origin,
                            "history_steps": 500,
                            "dt": dt,
                            "lambda_max": exponent,
                            "horizon_steps": horizon,
                            "horizon_lyapunov_times": horizon * dt * exponent,
                            "training_scales": scale.tolist(),
                            "fixed_training_scores": scores,
                            "public_metric_0.3_steps": legacy_steps,
                            "public_metric_0.3_lyapunov_times": legacy_steps * dt * exponent,
                            "fit_seconds": fit_seconds,
                            "forecast_seconds": elapsed,
                            "forecast_sha256": _digest(p.tobytes()),
                        }
                    )
    modules = {
        name: str(Path(module.__file__).resolve())
        for name, module in sys.modules.items()
        if (name == "resdag" or name.startswith("resdag.")) and getattr(module, "__file__", None)
    }
    if not all(Path(path).is_relative_to(source) for path in modules.values()):
        raise RuntimeError("ResDAG modules came from mixed revisions")
    np.savez_compressed(directory / f"{label}-predictions.npz", **forecasts)
    _write_json(directory / f"{label}-states.json", states)
    _write_json(
        directory / f"{label}-results.json",
        {
            "label": label,
            "commit": (directory / label / "commit.txt").read_text().strip(),
            "source": str(source),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
            "dtype": "float64",
            "threads": 1,
            "interop_threads": 1,
            "raw_data_sha256": data_hashes,
            "loaded_modules": modules,
            "rows": rows,
        },
    )


def _assert_matched(before: dict, after: dict, states_before: dict, states_after: dict) -> None:
    """Refuse a causal regression claim when the recipes/initial weights differ."""
    for field in (
        "torch",
        "numpy",
        "device",
        "dtype",
        "threads",
        "interop_threads",
        "raw_data_sha256",
    ):
        if before[field] != after[field]:
            raise ValueError(f"Unmatched environment field: {field}")
    if states_before.keys() != states_after.keys():
        raise ValueError("Unmatched model cases")
    for case in states_before:
        for field in ("recipe", "washout", "observations", "initial_graph"):
            if states_before[case][field] != states_after[case][field]:
                raise ValueError(f"Unmatched {field}: {case}")
    if [row["id"] for row in before["rows"]] != [row["id"] for row in after["rows"]]:
        raise ValueError("Unmatched forecast origins")
    for old, new in zip(before["rows"], after["rows"]):
        for field in ("origin_index", "history_steps", "dt", "lambda_max", "horizon_steps"):
            if old[field] != new[field]:
                raise ValueError(f"Unmatched {field}: {old['id']}")


def _compare(directory: Path) -> dict[str, Any]:
    import numpy as np

    before, after = (
        json.loads((directory / f"{label}-results.json").read_text())
        for label in ("before", "after")
    )
    old_states, new_states = (
        json.loads((directory / f"{label}-states.json").read_text())
        for label in ("before", "after")
    )
    _assert_matched(before, after, old_states, new_states)
    with (
        np.load(directory / "before-predictions.npz") as old,
        np.load(directory / "after-predictions.npz") as new,
    ):
        expected = {row["id"] for row in before["rows"]}
        if set(old.files) != expected or set(new.files) != expected:
            raise ValueError("Stored predictions do not match the reported forecast cases")
        if any(old[name].shape != new[name].shape for name in old.files):
            raise ValueError("Stored prediction shapes do not match")
        maximum = max(float(np.max(np.abs(old[name] - new[name]))) for name in old.files)
        bit_identical = all(old[name].tobytes() == new[name].tobytes() for name in old.files)
    summaries = []
    for system, family in sorted({(r["system"], r["family"]) for r in after["rows"]}):
        selected = [r for r in after["rows"] if (r["system"], r["family"]) == (system, family)]
        scores = {}
        for threshold in ("0.3", "0.5"):
            values = [r["fixed_training_scores"][threshold]["lyapunov_times"] for r in selected]
            scores[threshold] = {
                "median_lyapunov_times": statistics.median(values),
                "min_lyapunov_times": min(values),
                "max_lyapunov_times": max(values),
                "right_censored_count": sum(
                    r["fixed_training_scores"][threshold]["right_censored"] for r in selected
                ),
            }
        summaries.append(
            {"system": system, "family": family, "forecast_count": len(selected), "scores": scores}
        )
    return {
        "before_commit": before["commit"],
        "after_commit": after["commit"],
        "forecast_count": len(after["rows"]),
        "maximum_forecast_difference": maximum,
        "all_forecasts_bit_identical": bit_identical,
        "all_fitted_tensor_fingerprints_identical": all(
            old_states[key]["fitted_model"] == new_states[key]["fitted_model"] for key in old_states
        ),
        "all_fixed_training_scores_identical": all(
            old["fixed_training_scores"] == new["fixed_training_scores"]
            for old, new in zip(before["rows"], after["rows"])
        ),
        "summaries": summaries,
        "before": before,
        "after": after,
        "recipes_and_tensor_fingerprints": {"before": old_states, "after": new_states},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", default=BASELINE)
    parser.add_argument("--after", default="HEAD")
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true", help="One seed/origin per architecture")
    parser.add_argument("--worker", choices=("before", "after"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    directory = args.artifacts_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if args.worker:
        _worker(directory, args.worker, args.quick)
        return
    for label in ("before", "after"):
        if (directory / label).exists():
            raise ValueError("Choose a fresh artifacts-dir; source archives already exist")
    data = _generate_data(directory)
    for label, ref in (("before", args.before), ("after", args.after)):
        source = directory / label
        if source.exists():
            raise ValueError(
                f"Source directory already exists; choose a fresh artifacts-dir: {source}"
            )
        source.mkdir()
        _archive(ref, source)
        env = os.environ.copy()
        for variable in (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            env[variable] = "1"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            label,
            "--artifacts-dir",
            str(directory),
            "--output",
            str(args.output),
        ]
        if args.quick:
            command.append("--quick")
        with (directory / f"{label}.log").open("w") as log:
            subprocess.run(command, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
    report = _compare(directory)
    report.update(
        {
            "data": data,
            "quick": args.quick,
            "platform": platform.platform(),
            "python": sys.version,
            "runner_source_sha256": _digest(Path(__file__).read_bytes()),
            "headline": "Median per-origin/seed VPT, NRMSE < 0.3, fixed training-label scales",
            "scope": "Frozen classical regression recipes; no tuning or cross-library comparison",
            "timing_limit": "One sequential measurement per fit/forecast; not a speed benchmark",
            "target": "7–8 Lyapunov-time median is an objective, not established by these baselines",
            "artifacts_directory": str(directory),
            "artifact_files": {
                path.name: {"sha256": _digest(path.read_bytes()), "bytes": path.stat().st_size}
                for path in sorted(directory.iterdir())
                if path.is_file()
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("forecast_count", "maximum_forecast_difference", "summaries")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
