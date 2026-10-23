"""Compare vanilla, OTT and custom DAG forecasts on a TSDynamics trajectory.

Install ``resdag[dynamics,viz]`` and the Graphviz system executable. For a
dependency-light diagram use ``--diagram-format dot``. The defaults illustrate
the workflow; they are not tuned recipes or a claim of seven Lyapunov times.
Each run also writes ``experiment.json`` and ``replay-data.pt`` with resolved
settings, exact split data and artifact hashes. ``--verify`` replays saved
models without integration or fitting; recorded timings are not remeasured.

    python examples/17_forecast_experiment.py --system lorenz
    python examples/17_forecast_experiment.py --system rossler --reservoir-size 400
    python examples/17_forecast_experiment.py --verify forecast-experiment/experiment.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

import resdag as rd

MODEL_NAMES = ("vanilla", "ott", "custom")
SCHEMA_VERSION = 1
CLOCK_ATOL = 1e-10


def _json_text(value: object) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _identity(value: object) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_record(value: torch.Tensor) -> dict:
    value = value.detach().cpu().contiguous()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _json_record(value: object) -> object:
    """Keep resolved artifact/trajectory settings in standard, finite JSON."""
    if isinstance(value, torch.Tensor):
        return {"tensor": _tensor_record(value)}
    if isinstance(value, np.ndarray):
        return _json_record(value.tolist())
    if isinstance(value, dict):
        return {str(key): _json_record(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_record(item) for item in value]
    if isinstance(value, np.generic):
        return _json_record(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": str(value)}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Cannot record setting of type {type(value).__name__}")


def _model_configuration(payload: dict) -> dict:
    """Record the existing portable artifact's resolved component configuration."""
    return _json_record(
        {
            "format": payload["format"],
            "version": payload["version"],
            "root": payload["root"],
            "components": {
                key: {field: record[field] for field in ("name", "version", "config")}
                for key, record in payload["components"].items()
            },
        }
    )


def _dependencies() -> dict:
    result = {}
    for name in ("resdag", "torch", "numpy", "scipy", "networkx", "pytorch-symbolic", "tsdynamics"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def _metric_definitions() -> dict:
    return {
        "rmse": "sqrt(mean((prediction - truth)**2)) over time and signals; nonfinite is null",
        "scale": "training[washout + 1:].std(axis=0, ddof=0); zero scales become 1",
        "normalized_error": "sqrt(mean(((prediction - truth) / scale)**2, axis=-1))",
        "valid_prefix": "consecutive finite errors strictly below threshold, before first failure",
        "primary_threshold": 0.3,
        "secondary_threshold": 0.5,
        "valid_lyapunov_times": "valid_steps * actual_dt * lambda_max",
        "reached_horizon": "lower bound: all heldout steps pass; no extrapolated validity claim",
        "timings": "recorded wall seconds only; replay verifies correctness, not timing",
    }


def _score_record(*args: object, **kwargs: object) -> dict:
    result = forecast_scores(*args, **kwargs)
    if not math.isfinite(result["rmse"]):
        result["rmse"] = None
    return result


def _artifact_path(directory: Path, relative: str) -> Path:
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("artifact path must be relative to the manifest")
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("artifact path escapes the manifest directory")
    return path


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"nonstandard JSON constant: {value}")


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(), parse_constant=_reject_json_constant)


def _validate_replay_data(data: dict, recipe: dict, predictions: dict) -> None:
    """Check semantic split/clock alignment independently of the file hashes."""
    args, recorded = recipe["arguments"], recipe["data"]
    n, h, horizon = args["training"], args["history"], args["horizon"]
    washout, dt = args["washout"], recorded["actual_dt"]
    if not all(type(item) is int for item in (n, h, horizon, washout)):
        raise ValueError("split lengths and washout must be integers")
    if not (n > washout + 2 and washout >= 0 and 0 < h <= n and horizon > 0):
        raise ValueError("invalid training/history/heldout split")
    if type(dt) not in (int, float) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("actual_dt must be positive and finite")
    requested_dt = args["dt"]
    if (
        type(requested_dt) not in (int, float)
        or not math.isfinite(requested_dt)
        or abs(requested_dt - dt) > CLOCK_ATOL
    ):
        raise ValueError("resolved dt does not match the saved clock step")
    if recorded["splits"] != {"training": [0, n], "history": [n - h, n], "truth": [n, n + horizon]}:
        raise ValueError("split indices do not match resolved arguments")
    descriptors = {**recorded["tensors"], **predictions}
    expected = {
        f"{split}_{axis}" for split in ("training", "history", "truth") for axis in ("y", "t")
    }
    expected.update(f"{name}_prediction" for name in MODEL_NAMES)
    if not isinstance(data, dict) or set(data) != expected or set(descriptors) != expected:
        raise ValueError("replay tensor schema does not match experiment")
    if not isinstance(recorded["variables"], list):
        raise ValueError("state variables must be a list")
    width = len(recorded["variables"])
    if (
        width != 3
        or not all(isinstance(name, str) for name in recorded["variables"])
        or len(set(recorded["variables"])) != width
    ):
        raise ValueError("expected three distinct state variables")
    for key, tensor in data.items():
        length = n if key.startswith("training_") else h if key.startswith("history_") else horizon
        shape = (length,) if key.endswith("_t") else (length, width)
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            raise ValueError(f"invalid replay tensor: {key}")
        if (
            tensor.dtype != torch.float64
            or tensor.device.type != "cpu"
            or tuple(tensor.shape) != shape
        ):
            raise ValueError(f"invalid replay tensor shape/dtype/device: {key}")
        if _tensor_record(tensor) != descriptors[key]:
            raise ValueError(f"replay tensor hash mismatch: {key}")
        if not key.endswith("_prediction") and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"nonfinite replay data: {key}")
    if not torch.equal(data["history_y"], data["training_y"][-h:]) or not torch.equal(
        data["history_t"], data["training_t"][-h:]
    ):
        raise ValueError("history must equal the exact training tail")
    clocks = torch.cat((data["training_t"], data["truth_t"]))
    differences = clocks[1:] - clocks[:-1]
    if not bool((differences > 0).all()) or not torch.allclose(
        differences, torch.full_like(differences, dt), rtol=0, atol=CLOCK_ATOL
    ):
        raise ValueError("training and heldout clocks must be consecutive and uniform")
    expected_truth = data["history_t"][-1] + torch.arange(1, horizon + 1, dtype=torch.float64) * dt
    if not torch.allclose(data["truth_t"], expected_truth, rtol=0, atol=CLOCK_ATOL):
        raise ValueError("heldout clock does not align with forecast history")


def verify_experiment(path: Path) -> dict:
    """Replay saved forecasts; never integrate data, fit models, or time a run."""
    manifest = _read_json(path)
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError("unsupported experiment schema_version")
    if set(manifest) != {
        "schema_version",
        "experiment_id",
        "recipe",
        "models",
        "predictions",
        "scores",
        "artifacts",
    }:
        raise ValueError("experiment manifest schema mismatch")
    recipe = manifest["recipe"]
    if manifest["experiment_id"] != _identity(recipe) or recipe["metrics"] != _metric_definitions():
        raise ValueError("experiment identity or metric definitions mismatch")
    if [model["name"] for model in manifest["models"]] != list(MODEL_NAMES):
        raise ValueError("experiment model schema mismatch")
    artifacts = manifest["artifacts"]
    expected_artifacts = {"replay-data.pt", "scores.json"}
    for model in manifest["models"]:
        if model["artifact"] != f"{model['name']}.pt" or model["diagram"] not in (
            f"{model['name']}.dot",
            f"{model['name']}.svg",
        ):
            raise ValueError("model artifact names do not match the experiment schema")
        expected_artifacts.update((model["artifact"], model["diagram"]))
    if set(artifacts) != expected_artifacts:
        raise ValueError("experiment artifact schema mismatch")
    for relative, digest in artifacts.items():
        if _file_hash(_artifact_path(path.parent, relative)) != digest:
            raise ValueError(f"artifact hash mismatch: {relative}")
    data = torch.load(path.parent / "replay-data.pt", map_location="cpu", weights_only=True)
    _validate_replay_data(data, recipe, manifest["predictions"])
    if _read_json(path.parent / "scores.json") != manifest["scores"]:
        raise ValueError("scores.json does not match the manifest")
    if [row["model"] for row in manifest["scores"]] != list(MODEL_NAMES):
        raise ValueError("score model schema mismatch")
    args = recipe["arguments"]
    if type(args["threads"]) is not int or args["threads"] < 1:
        raise ValueError("threads must be positive")
    lambda_max = recipe["system"]["lambda_max"]
    if type(lambda_max) not in (int, float) or not math.isfinite(lambda_max) or lambda_max <= 0:
        raise ValueError("lambda_max must be positive and finite")
    for row in manifest["scores"]:
        expected_score = _score_record(
            data[f"{row['model']}_prediction"].numpy(),
            data["truth_y"].numpy(),
            data["training_y"].numpy(),
            washout=args["washout"],
            dt=recipe["data"]["actual_dt"],
            lambda_max=lambda_max,
        )
        if set(row) != {"model", "fit_seconds", "forecast_seconds", *expected_score} or any(
            type(row[field]) not in (int, float) or not math.isfinite(row[field]) or row[field] < 0
            for field in ("fit_seconds", "forecast_seconds")
        ):
            raise ValueError("score row schema mismatch")
        if expected_score != {key: row[key] for key in expected_score}:
            raise ValueError(f"recorded scores mismatch: {row['model']}")
    # Check every portable configuration before any model is constructed.
    for record in manifest["models"]:
        payload = torch.load(
            _artifact_path(path.parent, record["artifact"]), map_location="cpu", weights_only=True
        )
        expected_metadata = {
            "forecast_experiment": {
                "schema_version": SCHEMA_VERSION,
                "experiment_id": manifest["experiment_id"],
                "model": record["name"],
            }
        }
        if (
            payload["metadata"] != expected_metadata
            or _model_configuration(payload) != record["configuration"]
        ):
            raise ValueError(f"model metadata/configuration mismatch: {record['name']}")
    import tsdynamics as ts

    # Keep the original public trajectory adapter, including its fitted-clock
    # resampling. Passing raw values would bypass that preparation and can change
    # rounding even when the recorded observation clock is nearly uniform.
    history = ts.Trajectory(
        t=data["history_t"].numpy(),
        y=data["history_y"].numpy(),
        variables=recipe["data"]["variables"],
    )
    torch.set_num_threads(args["threads"])
    for record, recorded_score in zip(manifest["models"], manifest["scores"], strict=True):
        model = rd.load(_artifact_path(path.parent, record["artifact"]))
        prediction = model.forecast(history=history, horizon=args["horizon"], return_type="tensor")
        torch.testing.assert_close(
            prediction, data[f"{record['name']}_prediction"], rtol=0, atol=0, equal_nan=True
        )
        scores = _score_record(
            prediction.numpy(),
            data["truth_y"].numpy(),
            data["training_y"].numpy(),
            washout=args["washout"],
            dt=recipe["data"]["actual_dt"],
            lambda_max=lambda_max,
        )
        if scores != {key: recorded_score[key] for key in scores}:
            raise ValueError(f"replayed scores mismatch: {record['name']}")
    return {
        "experiment_id": manifest["experiment_id"],
        "verified_models": list(MODEL_NAMES),
        "forecasts": "exact replay",
        "scores": "recomputed",
        "timings": "recorded only; not verified",
        "recorded_dependencies": recipe["dependencies"],
        "replay_dependencies": _dependencies(),
    }


def build_models(reservoir_size: int, seed: int, washout: int) -> dict:
    """Keep the total reservoir width equal across three example architectures."""
    if reservoir_size < 2:
        raise ValueError("reservoir_size must be at least two")
    models = {
        "vanilla": rd.ESN(
            reservoir_size=reservoir_size,
            input_size=3,
            washout=washout,
            readout="svd",
            seed=seed,
            dtype=torch.float64,
        ),
        "ott": rd.Model(
            graph=rd.ott_esn(reservoir_size, 3, 3, readout="svd", seed=seed, dtype=torch.float64),
            washout=washout,
        ),
    }
    x = rd.reservoir_input(3, dtype=torch.float64)
    fast_width = reservoir_size // 2
    fast = rd.ESNLayer(
        fast_width,
        feedback_size=3,
        spectral_radius=0.9,
        leak_rate=1.0,
        seed=seed,
        dtype=torch.float64,
    )(x)
    slow = rd.ESNLayer(
        reservoir_size - fast_width,
        feedback_size=3,
        spectral_radius=0.9,
        leak_rate=0.3,
        seed=seed + 1,
        dtype=torch.float64,
    )(x)
    features = rd.Concatenate()(x, fast, slow)
    y = rd.SVDReadoutLayer(reservoir_size + 3, 3, alpha=1e-6).double()(features)
    models["custom"] = rd.Model(inputs={"state": x}, outputs={"prediction": y}, washout=washout)
    return models


def forecast_scores(
    prediction: np.ndarray,
    truth: np.ndarray,
    training: np.ndarray,
    *,
    washout: int,
    dt: float,
    lambda_max: float,
    threshold: float = 0.3,
) -> dict:
    """Score the uninterrupted valid prefix using training-only component scales."""
    if prediction.shape != truth.shape or prediction.ndim != 2 or not len(truth):
        raise ValueError("prediction and truth must share a nonempty (time, signals) shape")
    scale = training[washout + 1 :].std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    error = np.sqrt(np.mean(((prediction - truth) / scale) ** 2, axis=-1))
    valid = np.isfinite(error) & (error < threshold)
    steps = len(valid) if valid.all() else int(np.argmax(~valid))
    secondary_valid = np.isfinite(error) & (error < 0.5)
    secondary_steps = len(error) if secondary_valid.all() else int(np.argmax(~secondary_valid))
    return {
        "rmse": float(np.sqrt(np.mean((prediction - truth) ** 2))),
        "valid_steps": steps,
        "valid_lyapunov_times": steps * dt * lambda_max,
        "reached_horizon": bool(valid.all()),
        "secondary_valid_lyapunov_times_at_0_5": secondary_steps * dt * lambda_max,
        "secondary_reached_horizon": bool(secondary_valid.all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("lorenz", "rossler"), default="lorenz")
    parser.add_argument("--reservoir-size", type=int, default=192)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--training", type=int, default=4000)
    parser.add_argument("--washout", type=int, default=200)
    parser.add_argument("--history", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--diagram-format", choices=("svg", "dot"), default="svg")
    parser.add_argument("--output", type=Path, default=Path("forecast-experiment"))
    parser.add_argument(
        "--verify", type=Path, help="verify an existing experiment.json without fitting"
    )
    args = parser.parse_args()
    if args.verify is not None:
        print(_json_text(verify_experiment(args.verify)), end="")
        return
    if (
        args.training <= args.washout + 2
        or args.washout < 0
        or not 0 < args.history <= args.training
    ):
        parser.error("training must exceed washout + 2; history must be within training")
    if args.threads < 1:
        parser.error("threads must be positive")
    try:
        import tsdynamics as ts
    except ImportError as exc:
        raise ImportError("Install resdag[dynamics] on Python 3.12 or newer") from exc

    # These constants apply to the canonical parameter values below only.
    if args.system == "lorenz":
        system, default_dt, lambda_max = ts.systems.Lorenz(), 0.02, 0.9056
    else:
        system = ts.systems.Rossler(a=0.2, b=0.2, c=5.7)
        default_dt, lambda_max = 0.05, 0.0714
    dt = default_dt if args.dt is None else args.dt
    if not np.isfinite(dt) or dt <= 0:
        parser.error("dt must be positive and finite")
    horizon = math.ceil(10 / (dt * lambda_max)) if args.horizon is None else args.horizon
    if horizon < 1:
        parser.error("horizon must be positive")
    torch.set_num_threads(args.threads)
    trajectory = system.run(
        final_time=(args.training + horizon) * dt,
        dt=dt,
        transient=100.0,
        solver="dop853",
        rtol=1e-10,
        atol=1e-12,
    )
    training = trajectory[: args.training]
    truth = trajectory[args.training : args.training + horizon]
    actual_dt = float(training.t[1] - training.t[0])
    args.output.mkdir(parents=True, exist_ok=True)
    data = {}
    for name, split in (
        ("training", training),
        ("history", training[-args.history :]),
        ("truth", truth),
    ):
        data[f"{name}_y"] = torch.from_numpy(np.array(split.y, dtype=np.float64, copy=True))
        data[f"{name}_t"] = torch.from_numpy(np.array(split.t, dtype=np.float64, copy=True))
    resolved_args = {key: value for key, value in vars(args).items() if key != "verify"}
    resolved_args.update(dt=dt, horizon=horizon, output=str(args.output.resolve()))
    recipe = {
        "arguments": resolved_args,
        "system": {
            "name": type(system).__name__,
            "parameters": _json_record(system.params.as_dict()),
            "initial_condition": _json_record(system.ic),
            "lambda_max": lambda_max,
            "lambda_max_source": "canonical parameter constant, not estimated from this trajectory",
        },
        "integration": {
            "final_time": (args.training + horizon) * dt,
            "dt": dt,
            "t0": 0.0,
            "transient": 100.0,
            "solver": "dop853",
            "rtol": 1e-10,
            "atol": 1e-12,
            "max_step": None,
            "max_step_meaning": "unbounded",
            "resolved_provenance": _json_record(dict(trajectory.meta)),
        },
        "execution": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "device": "cpu",
            "dtype": "torch.float64",
            "model_backend": "pytorch",
            "forecast_compile": False,
            "forecast_execution": "fast",
            "history_input_kind": "tsdynamics.Trajectory",
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "dependencies": _dependencies(),
        "example_sha256": _file_hash(Path(__file__)),
        "data": {
            "variables": list(training.variables),
            "actual_dt": actual_dt,
            "splits": {
                "training": [0, args.training],
                "history": [args.training - args.history, args.training],
                "truth": [args.training, args.training + horizon],
            },
            "index_convention": "zero-based half-open indices of the post-transient trajectory",
            "tensor_hash_convention": "SHA-256 of contiguous CPU bytes; dtype and shape recorded separately",
            "tensors": {key: _tensor_record(value) for key, value in data.items()},
        },
        "metrics": _metric_definitions(),
    }
    experiment_id = _identity(recipe)
    model_records = []
    report = []
    for name, model in build_models(args.reservoir_size, args.seed, args.washout).items():
        start = perf_counter()
        model.fit(training)
        fit_seconds = perf_counter() - start
        start = perf_counter()
        prediction = model.forecast(history=training[-args.history :], horizon=horizon)
        forecast_seconds = perf_counter() - start
        np.testing.assert_allclose(prediction.t, truth.t, rtol=0, atol=1e-10)
        result = {
            "model": name,
            "fit_seconds": fit_seconds,
            "forecast_seconds": forecast_seconds,
            **_score_record(
                prediction.y,
                truth.y,
                training.y,
                washout=args.washout,
                dt=actual_dt,
                lambda_max=lambda_max,
            ),
        }
        report.append(result)
        print(json.dumps(result, allow_nan=False))
        model.save(args.output / f"{name}.{args.diagram_format}")
        model.save(
            args.output / f"{name}.pt",
            metadata={
                "forecast_experiment": {
                    "schema_version": SCHEMA_VERSION,
                    "experiment_id": experiment_id,
                    "model": name,
                }
            },
        )
        payload = torch.load(args.output / f"{name}.pt", map_location="cpu", weights_only=True)
        model_records.append(
            {
                "name": name,
                "artifact": f"{name}.pt",
                "diagram": f"{name}.{args.diagram_format}",
                "configuration": _model_configuration(payload),
            }
        )
        data[f"{name}_prediction"] = torch.from_numpy(
            np.array(prediction.y, dtype=np.float64, copy=True)
        )
        restored = rd.load(args.output / f"{name}.pt")
        restored_prediction = restored.forecast(history=training[-args.history :], horizon=horizon)
        np.testing.assert_allclose(restored_prediction.y, prediction.y, rtol=0, atol=0)
    (args.output / "scores.json").write_text(_json_text(report))
    torch.save(data, args.output / "replay-data.pt")
    predictions = {
        key: _tensor_record(value) for key, value in data.items() if key.endswith("_prediction")
    }
    _validate_replay_data(data, recipe, predictions)
    artifact_names = ["scores.json", "replay-data.pt"]
    for record in model_records:
        artifact_names.extend((record["artifact"], record["diagram"]))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "recipe": recipe,
        "models": model_records,
        "predictions": predictions,
        "scores": report,
        "artifacts": {name: _file_hash(args.output / name) for name in artifact_names},
    }
    (args.output / "experiment.json").write_text(_json_text(manifest))
    print("Untuned single-seed examples. reached_horizon means a lower bound, not the true limit.")
    print("Compare candidates on validation data; keep a separate final test segment untouched.")


if __name__ == "__main__":
    main()
