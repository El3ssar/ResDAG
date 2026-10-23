"""Bounded classical ESN tuning, untouched tests, and matched revision replay.

Uses the public ``Model.tune``/Optuna workflow, never a separate tuner. Run::

    python benchmarks/tuned_accuracy.py --before 00ca741 --after 0b546d6 \
        --artifacts-dir /tmp/resdag-tuned --output tuned-accuracy.json

Four balanced studies each attempt 24 configurations, three initializations and
three validation origins. Winners are frozen before evaluating five unseen
initializations at three later test origins. Both source revisions replay those
winners and unoptimized baselines on identical data; optional known Rössler
recipes are reported separately. All horizons span at least ten Lyapunov times.
This is a bounded capability experiment, not a guaranteed forecast-quality claim.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

_HELPER_SPEC = importlib.util.spec_from_file_location(
    "accuracy_regression_helpers", Path(__file__).with_name("accuracy_regression.py")
)
assert _HELPER_SPEC is not None and _HELPER_SPEC.loader is not None
HELPERS = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(HELPERS)
TRAIN_SEEDS = (7, 42, 12345)
TEST_SEEDS = (201, 202, 203, 204, 205)
SMOKE_SEEDS = (101,)
WIDTHS = [300, 400, 600, 1000]
WASHOUT = 500


def _protocol(system: str) -> dict[str, Any]:
    dt, exponent, train = (0.02, 0.9056, 10001) if system == "lorenz" else (0.145, 0.0714, 8501)
    horizon = math.ceil(10 / (dt * exponent))
    validation_end = train + 3 * horizon
    return {
        "dt": dt,
        "lambda_max": exponent,
        "washout": WASHOUT,
        "training": [0, train],
        "validation": [train, validation_end],
        "test": [validation_end, validation_end + 3 * horizon],
        "test_origins": [validation_end + index * horizon for index in range(3)],
        "horizon_steps": horizon,
        "horizon_lyapunov_times": horizon * dt * exponent,
    }


def _search(trial: Any) -> dict[str, Any]:
    return {
        "width": trial.suggest_categorical("width", WIDTHS),
        "rho": trial.suggest_float("rho", 0.05, 1.4),
        "leak": trial.suggest_float("leak", 0.1, 1.0),
        "input_scale": trial.suggest_float("input_scale", 0.01, 1.0, log=True),
        "bias_scale": trial.suggest_float("bias_scale", 0.01, 0.5, log=True),
        "alpha": trial.suggest_float("alpha", 1e-12, 1e-3, log=True),
        "degree": trial.suggest_categorical("degree", [3, 6, 12]),
    }


def _known() -> dict[str, Any]:
    return dict(
        width=400, rho=0.21, leak=0.85, input_scale=0.35, bias_scale=0.11, alpha=3e-10, degree=6
    )


def _baseline(system: str) -> dict[str, Any]:
    return dict(
        width=300 if system == "lorenz" else 400,
        rho=0.9,
        leak=1.0,
        input_scale=1.0,
        bias_scale=1.0,
        alpha=1e-6,
        degree=6,
    )


def _build(rd: Any, torch: Any, family: str, params: dict, seed: int) -> Any:
    builder = rd.classic_esn if family == "classic" else rd.ott_esn
    graph = builder(
        params["width"],
        3,
        3,
        spectral_radius=params["rho"],
        leak_rate=params["leak"],
        feedback_initializer=("random", {"input_scaling": params["input_scale"]}),
        bias_scaling=params["bias_scale"],
        topology=(
            "erdos_renyi",
            {"p": params["degree"] / params["width"], "directed": True, "self_loops": True},
        ),
        readout="svd",
        readout_alpha=params["alpha"],
        seed=seed,
        dtype=torch.float64,
    )
    return rd.Model(graph=graph, washout=WASHOUT).eval()


def _data(directory: Path) -> dict[str, Any]:
    from importlib.metadata import version

    import numpy as np
    import tsdynamics as ts

    records = {}
    for name in ("lorenz", "rossler"):
        protocol = _protocol(name)
        count, dt, discard = protocol["test"][1], protocol["dt"], 2000 if name == "lorenz" else 1000
        system = ts.systems.Lorenz() if name == "lorenz" else ts.systems.Rossler()
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
            raise RuntimeError(f"Invalid {name} source data")
        np.save(directory / f"{name}.npy", raw)
        records[name] = {
            "protocol": protocol,
            "shape": list(raw.shape),
            "parameters": repr(system),
            "initial_condition": [1.0, 1.0, 1.0],
            "discard_samples": discard,
            "solver": "dop853",
            "rtol": 1e-10,
            "atol": 1e-12,
            "tsdynamics": version("tsdynamics"),
            "raw_sha256": HELPERS._digest(raw.tobytes()),
        }
    HELPERS._write_json(directory / "data.json", records)
    return records


def _imports(directory: Path, label: str) -> tuple[Any, Any]:
    import torch

    source = directory / label / "src"
    sys.path.insert(0, str(source))
    import resdag as rd

    if not Path(rd.__file__).resolve().is_relative_to(source):
        raise RuntimeError(f"Wrong source import: {rd.__file__}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    return rd, torch


def _import_manifest(directory: Path, label: str) -> dict:
    source = directory / label / "src"
    modules = {
        name: str(Path(module.__file__).resolve())
        for name, module in sys.modules.items()
        if (name == "resdag" or name.startswith("resdag.")) and getattr(module, "__file__", None)
    }
    if not all(Path(path).is_relative_to(source) for path in modules.values()):
        raise RuntimeError("Mixed source revisions")
    return modules


def _new_study(optuna: Any, directory: Path, name: str) -> Any:
    return optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=20261023),
        pruner=optuna.pruners.NopPruner(),
        study_name=name,
        storage=f"sqlite:///{directory / (name + '.sqlite3')}",
    )


def _budget_evidence(study: Any, expected_trials: int, expected_runs: int, horizon: int) -> dict:
    from collections import Counter

    counts = Counter(trial.state.name for trial in study.trials)
    if (
        len(study.trials) != expected_trials
        or counts["PRUNED"]
        or counts["RUNNING"]
        or counts["WAITING"]
    ):
        raise ValueError("The declared full-check attempt budget was not completed without pruning")
    for trial in study.trials:
        if trial.state.name == "COMPLETE":
            runs = trial.user_attrs.get("runs", [])
            if len(runs) != expected_runs or any(
                not run.get("complete") or run.get("completed_steps") != horizon for run in runs
            ):
                raise ValueError(
                    "A completed trial did not evaluate every full seed/origin forecast"
                )
    return {
        "sampler": type(study.sampler).__name__,
        "pruner": type(study.pruner).__name__,
        "attempts": expected_trials,
        "state_counts": {
            state: counts[state] for state in ("COMPLETE", "FAIL", "PRUNED", "RUNNING", "WAITING")
        },
        "runs_per_complete_trial": expected_runs,
        "full_horizon_steps": horizon,
        "all_attempts_complete": counts["COMPLETE"] == expected_trials,
        "all_successful_trials_received_full_checks": True,
    }


def _tune(directory: Path, n_trials: int, quick: bool) -> None:
    import numpy as np
    import optuna

    rd, torch = _imports(directory, "after")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    winners = {}
    for system in ("lorenz", "rossler"):
        protocol = _protocol(system)
        train_stop, val_stop = protocol["training"][1], protocol["validation"][1]
        raw_array = np.load(directory / f"{system}.npy", mmap_mode="r")
        raw = torch.from_numpy(np.array(raw_array[:val_stop], copy=True))
        for family in ("classic", "ott"):

            def factory(params, seed):
                return _build(rd, torch, family, params, seed)

            model = factory(_known(), TRAIN_SEEDS[0])
            name = f"{system}-{family}"
            study = _new_study(optuna, directory, name)
            starting = (
                [_known()]
                if system == "rossler"
                else [
                    dict(
                        width=1000,
                        rho=0.1,
                        leak=1.0,
                        input_scale=0.1,
                        bias_scale=0.1,
                        alpha=1e-8,
                        degree=6,
                    )
                ]
            )
            for params in starting:
                study.enqueue_trial(params)
            model.tune(
                raw[:train_stop],
                validation=(raw[train_stop:val_stop], None),
                factory=factory,
                search_space=_search,
                preset="balanced",
                n_trials=n_trials,
                seeds=TRAIN_SEEDS[:1] if quick else TRAIN_SEEDS,
                n_origins=1 if quick else 3,
                horizon=protocol["horizon_steps"],
                threshold=0.3,
                study=study,
                seed=20261023,
                n_jobs=1,
                inner_max_num_threads=1,
                catch_exceptions=True,
            )
            report = model.tuning_report
            budget = _budget_evidence(study, n_trials, 1 if quick else 9, protocol["horizon_steps"])
            winners[name] = {
                "budget": budget,
                "params": report.best_params,
                "best_trial": report.best_trial,
                "validation_score_steps": report.score,
                "validation_score_lyapunov_times": report.score
                * protocol["dt"]
                * protocol["lambda_max"],
                "attempts": report.attempts,
                "elapsed_seconds": report.elapsed_seconds,
                "seeds": list(report.seeds),
                "validation": report.validation,
                "protocol_hash": report.protocol_hash,
                "runs": list(report.runs),
                "queued_candidates": starting,
                "trials": [
                    {
                        "number": t.number,
                        "state": t.state.name,
                        "params": t.params,
                        "value": t.value,
                        "user_attrs": t.user_attrs,
                    }
                    for t in study.trials
                ],
            }
            HELPERS._write_json(directory / f"{name}-tuning.json", winners[name])
            print(
                json.dumps(
                    {
                        "finished": name,
                        "best_params": report.best_params,
                        "validation_lt": winners[name]["validation_score_lyapunov_times"],
                        "seconds": report.elapsed_seconds,
                    }
                ),
                flush=True,
            )
    HELPERS._write_json(
        directory / "frozen-winners.json",
        {
            "winners": winners,
            "test_used_for_selection": False,
            "source_imports": _import_manifest(directory, "after"),
        },
    )


def _evaluate(directory: Path, label: str, quick: bool) -> None:
    import time

    import numpy as np

    rd, torch = _imports(directory, label)
    from resdag.hpo._tuning_scores import score_forecast

    winner_path = directory / "frozen-winners.json"
    frozen = winner_path.read_bytes()
    winners = json.loads(frozen)["winners"]
    rows, predictions, fingerprints = [], {}, {}
    for system in ("lorenz", "rossler"):
        protocol = _protocol(system)
        raw = torch.from_numpy(np.load(directory / f"{system}.npy").copy())
        training = raw[: protocol["training"][1]]
        scales = training[WASHOUT + 1 :].std(dim=0, correction=0)
        scales = torch.where(
            scales == 0, torch.ones_like(scales), scales.clamp_min(torch.finfo(scales.dtype).eps)
        )
        for family in ("classic", "ott"):
            variants = {
                "unoptimized": _baseline(system),
                "winner": winners[f"{system}-{family}"]["params"],
            }
            if system == "rossler":
                variants["known_recipe"] = _known()
            for variant, params in variants.items():
                for seed in SMOKE_SEEDS if quick else TEST_SEEDS:
                    torch.manual_seed(918273)
                    model = _build(rd, torch, family, params, seed)
                    case = f"{system}-{family}-{variant}-seed{seed}"
                    fingerprints[case] = {
                        "initial": HELPERS._fingerprints(model.graph),
                        "params": params,
                    }
                    started = time.perf_counter()
                    model.fit(training)
                    fit_seconds = time.perf_counter() - started
                    fingerprints[case]["fitted"] = HELPERS._fingerprints(model)
                    if (
                        label == "after"
                        and variant == "winner"
                        and seed == (SMOKE_SEEDS[0] if quick else TEST_SEEDS[0])
                    ):
                        model.save(directory / f"{system}-{family}-winner-seed{seed}.pt")
                    for origin_id, origin in enumerate(
                        protocol["test_origins"][:1] if quick else protocol["test_origins"]
                    ):
                        truth = raw[origin : origin + protocol["horizon_steps"]]
                        started = time.perf_counter()
                        pred = model.forecast(history=raw[:origin], horizon=len(truth))
                        forecast_seconds = time.perf_counter() - started
                        key = f"{case}-origin{origin_id}"
                        p, target = pred.detach().numpy(), truth.numpy()
                        if not np.isfinite(p).all():
                            raise RuntimeError(f"Nonfinite held-out prediction: {key}")
                        predictions[key] = p
                        scores = {}
                        for threshold in (0.3, 0.5):
                            result = score_forecast(
                                {"prediction": pred},
                                {"prediction": truth},
                                {"prediction": scales},
                                threshold=threshold,
                            )
                            if result.score != HELPERS._independent_horizon(
                                p, target, scales.numpy(), threshold
                            ):
                                raise RuntimeError("Independent metric oracle failed")
                            scores[str(threshold)] = {
                                "steps": result.score,
                                "lyapunov_times": result.score
                                * protocol["dt"]
                                * protocol["lambda_max"],
                                "right_censored": result.score == len(truth),
                                "full_horizon_nrmse": result.nrmse,
                            }
                        rows.append(
                            {
                                "id": key,
                                "system": system,
                                "family": family,
                                "variant": variant,
                                "seed": seed,
                                "origin_id": origin_id,
                                "origin_index": origin,
                                "history_steps": origin,
                                "scores": scores,
                                "training_scales": scales.tolist(),
                                "forecast_sha256": HELPERS._digest(p.tobytes()),
                                "fit_seconds": fit_seconds,
                                "forecast_seconds": forecast_seconds,
                            }
                        )
    if frozen != winner_path.read_bytes():
        raise RuntimeError("Winning parameters changed during test evaluation")
    np.savez_compressed(directory / f"{label}-test-predictions.npz", **predictions)
    HELPERS._write_json(
        directory / f"{label}-test.json",
        {
            "rows": rows,
            "fingerprints": fingerprints,
            "winners_sha256": HELPERS._digest(frozen),
            "source_imports": _import_manifest(directory, label),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    )


def _summaries(rows: list[dict]) -> list[dict]:
    summaries = []
    for system, family, variant in sorted({(r["system"], r["family"], r["variant"]) for r in rows}):
        chosen = [
            r for r in rows if (r["system"], r["family"], r["variant"]) == (system, family, variant)
        ]
        summary = {
            "system": system,
            "family": family,
            "variant": variant,
            "count": len(chosen),
            "scores": {},
        }
        for threshold in ("0.3", "0.5"):
            values = [r["scores"][threshold]["lyapunov_times"] for r in chosen]
            summary["scores"][threshold] = {
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
                "right_censored": sum(r["scores"][threshold]["right_censored"] for r in chosen),
                "per_seed_medians": {
                    str(seed): statistics.median(
                        r["scores"][threshold]["lyapunov_times"]
                        for r in chosen
                        if r["seed"] == seed
                    )
                    for seed in sorted({r["seed"] for r in chosen})
                },
            }
        summaries.append(summary)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", default=HELPERS.BASELINE)
    parser.add_argument("--after", default="0b546d6")
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-trials", type=int, default=24)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--worker", choices=("tune", "before", "after"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.n_trials <= 24:
        parser.error("n-trials must be between 1 and 24")
    directory = args.artifacts_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if args.worker == "tune":
        _tune(directory, args.n_trials, args.quick)
        return
    if args.worker:
        _evaluate(directory, args.worker, args.quick)
        return
    if any((directory / label).exists() for label in ("before", "after")):
        raise ValueError("Use a fresh artifacts directory")
    data = _data(directory)
    commits = {}
    for label, ref in (("before", args.before), ("after", args.after)):
        source = directory / label
        source.mkdir()
        commits[label] = HELPERS._archive(ref, source)
    for worker in ("tune", "before", "after"):
        env = os.environ.copy()
        for name in (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            env[name] = "1"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            worker,
            "--artifacts-dir",
            str(directory),
            "--output",
            str(args.output),
            "--n-trials",
            str(args.n_trials),
        ]
        if args.quick:
            command.append("--quick")
        with (directory / f"{worker}.log").open("w") as log:
            subprocess.run(command, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
    import numpy as np

    before, after = (
        json.loads((directory / f"{label}-test.json").read_text()) for label in ("before", "after")
    )
    if before["winners_sha256"] != after["winners_sha256"]:
        raise RuntimeError("Revision replay did not use the same frozen winner")
    if [r["id"] for r in before["rows"]] != [r["id"] for r in after["rows"]]:
        raise RuntimeError("Revision replay did not use the same cases")
    if any(
        before["fingerprints"][case]["initial"] != after["fingerprints"][case]["initial"]
        for case in before["fingerprints"]
    ):
        raise RuntimeError("Revision replay did not use the same initial weights")
    with (
        np.load(directory / "before-test-predictions.npz") as old,
        np.load(directory / "after-test-predictions.npz") as new,
    ):
        maximum = max(float(np.max(np.abs(old[key] - new[key]))) for key in old.files)
    report = {
        "commits": commits,
        "data": data,
        "quick": args.quick,
        "n_trials_per_study": args.n_trials,
        "train_seeds": list(TRAIN_SEEDS[:1] if args.quick else TRAIN_SEEDS),
        "test_seeds": list(SMOKE_SEEDS if args.quick else TEST_SEEDS),
        "protocol": "fixed training-label population std, threshold0.3 headline/0.5 secondary, median per unseen seed/origin; winners frozen before test",
        "timing_limit": "Recorded sequential fit/forecast observations are not a controlled speed benchmark",
        "platform": platform.platform(),
        "python": sys.version,
        "runner_source_sha256": HELPERS._digest(Path(__file__).read_bytes()),
        "helper_source_sha256": HELPERS._digest(Path(HELPERS.__file__).read_bytes()),
        "frozen_winners": json.loads((directory / "frozen-winners.json").read_text()),
        "maximum_revision_forecast_difference": maximum,
        "fitted_tensor_fingerprints_identical": all(
            before["fingerprints"][case]["fitted"] == after["fingerprints"][case]["fitted"]
            for case in before["fingerprints"]
        ),
        "scores_identical": all(
            old["scores"] == new["scores"] for old, new in zip(before["rows"], after["rows"])
        ),
        "summaries": _summaries(after["rows"]),
        "before": before,
        "after": after,
        "artifacts_directory": str(directory),
        "artifact_files": {
            path.name: {"sha256": HELPERS._digest(path.read_bytes()), "bytes": path.stat().st_size}
            for path in sorted(directory.iterdir())
            if path.is_file()
        },
    }
    report["validated_study_budgets"] = {
        name: winner["budget"] for name, winner in report["frozen_winners"]["winners"].items()
    }
    report["selection_integrity"] = {
        "smoke_seeds": list(SMOKE_SEEDS),
        "test_used_for_selection": False,
        "winning_params_frozen_before_test": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    HELPERS._write_json(args.output, report)
    print(
        json.dumps(
            {"summaries": report["summaries"], "max_revision_forecast_difference": maximum},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
