"""Validate vanilla ESN recipes on independent runs before opening final tests.

This experiment uses standard Optuna TPE and the public ESN fit/forecast API.
Every candidate is refitted separately on each validation trajectory. A batched
fit would instead share one readout, answering a different question. No inspected
phase-five test trajectory is admitted. Run with a new output directory::

    python benchmarks/vanilla_robustness.py --output /tmp/vanilla-robustness

``--quick`` is an execution smoke, not forecast-quality evidence. Full searches
attempt at most 24 configurations per system with a 600-second soft search
budget; an already-started candidate completes all seeds and origins. Both
winners are frozen before any final-test trajectory is generated.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import shutil
import statistics
import subprocess
import sys
import time
import warnings
import zlib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ("lorenz", "rossler")


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _identity(value: Any) -> str:
    return _hash(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _plan(quick: bool, trials: int, timeout: float, numpy: Any) -> dict[str, Any]:
    rng = numpy.random.default_rng(2026102407)
    systems = {}
    for name in SYSTEMS:
        dt, exponent, count = (0.02, 0.9056, 10001) if name == "lorenz" else (0.145, 0.0714, 8501)
        horizon = 24 if quick else math.ceil(10 / (dt * exponent))
        origins = 1 if quick else 3
        systems[name] = {
            "parameters": (
                {"sigma": 10.0, "rho": 28.0, "beta": 8 / 3}
                if name == "lorenz"
                else {"a": 0.2, "b": 0.2, "c": 5.7}
            ),
            "dt": dt,
            "lambda_max": exponent,
            "train_observations": 180 if quick else count,
            "washout": 20 if quick else 500,
            "horizon": horizon,
            "origin_count": origins,
            "discard": 20 if quick else (2000 if name == "lorenz" else 1000),
            "validation_ics": rng.uniform(0.5, 1.5, size=(1 if quick else 3, 3)).tolist(),
            "final_ics": rng.uniform(0.5, 1.5, size=(1 if quick else 5, 3)).tolist(),
        }
    return {
        "schema": 1,
        "kind": "execution smoke" if quick else "independent-trajectory recipe search",
        "systems": systems,
        "trials_per_system": trials,
        "search_timeout_seconds": timeout,
        "search_budget": "soft wall time; finish all checks of each admitted candidate",
        "search_sampler_seed": 2026102411,
        "initial_condition_rng_seed": 2026102407,
        "validation_seeds": [6849201] if quick else [6849201, 6849202, 6849203],
        "final_seeds": [7849201] if quick else [7849201, 7849202, 7849203],
        "solver": "Torch SVD, float64, unpenalized intercept",
        "preprocessing": "training-only standard scaling; original-unit forecasts",
        "history": "entire observed prefix; first target is immediately after its last sample",
        "metric": "first per-step training-standardized RMS error >= threshold",
        "metric_scale_rows": "raw[washout + 1 : train_observations], matching preceding primary protocol",
        "thresholds": [0.3, 0.5],
        "objective": "median of per-trajectory medians at 0.3",
        "nested_observations": "each trajectory median contains all seed/origin forecasts",
        "pruning": False,
        "thread_policy": "one Torch intra/inter-op thread and one verified native thread",
        "input_to_readout_choices": [True, False],
        "widths": [16, 24] if quick else [300, 400, 600, 1000],
        "other_ranges": {
            "spectral_radius": [0.05, 1.4],
            "leak_rate": [0.1, 1.0],
            "input_scaling_log": [0.01, 1.0],
            "bias_scaling_log": [0.01, 0.5],
            "alpha_log": [1e-12, 1e-3],
            "mean_degree": [3, 6, 12],
        },
        "final_rule": "freeze both winners before generating any final observations",
        "solver_parameters": {"method": "dop853", "rtol": 1e-10, "atol": 1e-12},
    }


def _search(trial: Any, plan: dict) -> dict:
    return {
        "reservoir_size": trial.suggest_categorical("reservoir_size", plan["widths"]),
        "spectral_radius": trial.suggest_float("spectral_radius", 0.05, 1.4),
        "leak_rate": trial.suggest_float("leak_rate", 0.1, 1.0),
        "input_scaling": trial.suggest_float("input_scaling", 0.01, 1.0, log=True),
        "bias_scaling": trial.suggest_float("bias_scaling", 0.01, 0.5, log=True),
        "alpha": trial.suggest_float("alpha", 1e-12, 1e-3, log=True),
        "mean_degree": trial.suggest_categorical("mean_degree", [3, 6, 12]),
        "input_to_readout": trial.suggest_categorical("input_to_readout", [True, False]),
    }


def _incumbents(path: Path, quick: bool) -> tuple[dict, dict]:
    """Admit old validation-selected primary recipes, never select by old tests."""
    source = json.loads(path.read_text())
    recipes = source["preregistration"]["protocol"]["recipes"]
    result = {}
    for name, recipe in recipes.items():
        if recipe["family"] != "classic" or recipe["panel"] != "primary-capability":
            continue
        old = recipe["params"]
        result[name.removesuffix("-classic")] = {
            "reservoir_size": 16 if quick else old["width"],
            "spectral_radius": old["rho"],
            "leak_rate": old["leak"],
            "input_scaling": old["input_scale"],
            "bias_scaling": old["bias_scale"],
            "alpha": old["alpha"],
            "mean_degree": old["degree"],
            "input_to_readout": True,
        }
    if set(result) != set(SYSTEMS):
        raise ValueError("Both preregistered primary vanilla recipes are required")
    return result, {"file": path.name, "sha256": _hash(path.read_bytes())}


def _generate(
    name: str, config: dict, conditions: list, directory: Path, ts: Any, numpy: Any
) -> list:
    if directory.name == "final" and not (directory.parent / "frozen-winners.json").is_file():
        raise RuntimeError("Freeze both winners before generating final observations")
    result = []
    count = config["train_observations"] + config["origin_count"] * config["horizon"]
    for index, ic in enumerate(conditions):
        start = time.perf_counter()
        constructor = ts.systems.Lorenz if name == "lorenz" else ts.systems.Rossler
        system = constructor(**config["parameters"])
        trajectory = system.run(
            final_time=(count + config["discard"]) * config["dt"],
            dt=config["dt"],
            ic=ic,
            solver="dop853",
            rtol=1e-10,
            atol=1e-12,
        )
        raw = numpy.asarray(
            trajectory.y[config["discard"] : config["discard"] + count], dtype="<f8"
        )
        if raw.shape != (count, 3) or not numpy.isfinite(raw).all():
            raise ValueError("Invalid generated trajectory")
        path = directory / f"{name}-{index}.npy"
        numpy.save(path, raw)
        result.append(
            {
                "path": path,
                "sha256": _hash(path.read_bytes()),
                "seconds": time.perf_counter() - start,
            }
        )
    return result


def _score(prediction: Any, truth: Any, scale: Any, config: dict, torch: Any) -> tuple[dict, dict]:
    if prediction.shape != truth.shape or prediction.ndim != 2 or scale.shape != (3,):
        raise ValueError("Forecast, truth and training scale shapes must agree")
    if (
        not bool(torch.isfinite(truth).all())
        or not bool(torch.isfinite(scale).all())
        or not bool((scale > 0).all())
    ):
        raise ValueError("Truth and training scales must be finite and nonconstant")
    errors = ((prediction - truth) / scale).square().mean(-1).sqrt()
    scores = {}
    for threshold in (0.3, 0.5):
        failed = (~torch.isfinite(errors) | (errors >= threshold)).nonzero()
        steps = int(failed[0, 0]) if len(failed) else len(errors)
        scores[str(threshold)] = {
            "steps": steps,
            "lyapunov_times": steps * config["dt"] * config["lambda_max"],
            "right_censored": steps == len(errors),
        }
    payload = errors.detach().contiguous().numpy().astype("<f8", copy=False).tobytes()
    return scores, {
        "encoding": "zlib-base64 little-endian float64",
        "length": len(errors),
        "sha256": _hash(payload),
        "data": base64.b64encode(zlib.compress(payload)).decode(),
    }


def _aggregate(rows: list[dict]) -> dict:
    trajectories = sorted({row["trajectory"] for row in rows})
    result = {}
    for threshold in ("0.3", "0.5"):
        medians = [
            statistics.median(
                row["scores"][threshold]["lyapunov_times"]
                for row in rows
                if row["trajectory"] == index
            )
            for index in trajectories
        ]
        values = [row["scores"][threshold]["lyapunov_times"] for row in rows]
        result[threshold] = {
            "median_of_trajectory_medians": statistics.median(medians),
            "trajectory_medians": medians,
            "individual_range": [min(values), max(values)],
            "right_censored": sum(row["scores"][threshold]["right_censored"] for row in rows),
            "forecasts": len(rows),
        }
    return result


def _reservoir_fingerprint(model: Any, rd: Any) -> str:
    """Hash the initialized reservoir, excluding the intentionally different head."""
    parameters = {}
    for index, layer in enumerate(
        module for module in model.graph.modules() if isinstance(module, rd.ESNLayer)
    ):
        for name, value in layer.named_parameters():
            tensor = value.detach().cpu().contiguous()
            parameters[f"{index}.{name}"] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": _hash(tensor.numpy().tobytes()),
            }
    if not parameters:
        raise RuntimeError("The experiment requires an identifiable ESN reservoir")
    return _identity(parameters)


def _paired_baselines(records: list[dict], states: dict[str, str]) -> dict:
    """Require both controlled feature choices before opening any final data."""
    if any(states.get(str(index)) != "COMPLETE" for index in (0, 1)):
        raise RuntimeError("Both controlled True/False baseline trials must complete")
    paired = {record["trial"]: record for record in records if record["trial"] in (0, 1)}
    left, right = paired[0], paired[1]
    if (
        left["params"]["input_to_readout"] is not True
        or right["params"]["input_to_readout"] is not False
    ):
        raise RuntimeError("Controlled baseline feature choices changed")

    def omit(params: dict) -> dict:
        return {key: value for key, value in params.items() if key != "input_to_readout"}

    if omit(left["params"]) != omit(right["params"]):
        raise RuntimeError("Controlled baselines must differ only in readout features")

    def fits(record: dict) -> list:
        return [
            (fit["trajectory"], fit["seed"], fit["reservoir_sha256"])
            for fit in record["evaluation"]["fits"]
        ]

    if not fits(left) or fits(left) != fits(right):
        raise RuntimeError("Controlled baselines must use identical reservoir weights")
    return {
        "input_plus_state": left["evaluation"]["summary"],
        "state_only": right["evaluation"]["summary"],
        "same_reservoir_weights": True,
        "changed_setting": "input_to_readout only",
    }


def _evaluate_impl(
    params: dict,
    datasets: list,
    config: dict,
    seeds: list,
    rd: Any,
    torch: Any,
    numpy: Any,
    result: dict | None = None,
) -> dict:
    rows, fits = [], []
    result = {} if result is None else result
    result.update({"fits": fits, "forecasts": rows})
    result["training_scales"] = []
    for index, dataset in enumerate(datasets):
        if _hash(dataset["path"].read_bytes()) != dataset["sha256"]:
            raise RuntimeError("Dataset changed after registration")
        raw = torch.from_numpy(numpy.load(dataset["path"]))
        train = raw[: config["train_observations"]]
        scale = train[config["washout"] + 1 :].std(dim=0, unbiased=False)
        result["training_scales"].append({"trajectory": index, "scale": scale.tolist()})
        for seed in seeds:
            result["in_progress"] = {"operation": "fit", "trajectory": index, "seed": seed}
            start = time.perf_counter()
            settings = {
                key: value
                for key, value in params.items()
                if key not in ("mean_degree", "input_scaling")
            }
            model = rd.ESN(
                input_size=3,
                washout=config["washout"],
                readout="svd",
                readout_bias=True,
                topology=(
                    "erdos_renyi",
                    {
                        "p": params["mean_degree"] / params["reservoir_size"],
                        "directed": True,
                        "self_loops": True,
                    },
                ),
                feedback_initializer=("random", {"input_scaling": params["input_scaling"]}),
                seed=seed,
                dtype=torch.float64,
                **settings,
            ).eval()
            fingerprint = _reservoir_fingerprint(model, rd)
            model.fit(train)
            if _reservoir_fingerprint(model, rd) != fingerprint:
                raise RuntimeError("Readout fitting unexpectedly changed reservoir weights")
            fits.append(
                {
                    "trajectory": index,
                    "seed": seed,
                    "seconds": time.perf_counter() - start,
                    "reservoir_sha256": fingerprint,
                }
            )
            for origin in range(config["origin_count"]):
                result["in_progress"] = {
                    "operation": "forecast",
                    "trajectory": index,
                    "seed": seed,
                    "origin": origin,
                }
                offset = config["train_observations"] + origin * config["horizon"]
                start = time.perf_counter()
                prediction = model.forecast(raw[:offset], horizon=config["horizon"])
                seconds = time.perf_counter() - start
                scores, trace = _score(
                    prediction, raw[offset : offset + config["horizon"]], scale, config, torch
                )
                rows.append(
                    {
                        "trajectory": index,
                        "seed": seed,
                        "origin": origin,
                        "origin_index": offset,
                        "history_observations": offset,
                        "seconds": seconds,
                        "prediction_sha256": _hash(
                            prediction.detach().contiguous().numpy().tobytes()
                        ),
                        "finite": bool(torch.isfinite(prediction).all()),
                        "scores": scores,
                        "error_trace": trace,
                    }
                )
    result.pop("in_progress")
    result["summary"] = _aggregate(rows)
    return result


def _evaluate(
    params: dict,
    datasets: list,
    config: dict,
    seeds: list,
    rd: Any,
    torch: Any,
    numpy: Any,
    result: dict | None = None,
) -> dict:
    result = {} if result is None else result
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            return _evaluate_impl(params, datasets, config, seeds, rd, torch, numpy, result)
        finally:
            result["warnings"] = [
                {
                    "category": warning.category.__name__,
                    "message": str(warning.message),
                    "file": Path(warning.filename).name,
                    "line": warning.lineno,
                }
                for warning in captured
            ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.trials < 2 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("at least two trials and a finite positive timeout are required")
    if args.output.exists():
        parser.error("output already exists; choose a new exclusive directory")
    threads = _load("vanilla_threads", ROOT / "benchmarks/rc_bench/_threads.py")
    threads.bootstrap_cli()
    import numpy as np
    import optuna
    import torch
    import tsdynamics as ts

    args.output.mkdir(parents=True, exist_ok=False)
    source = args.output / "source"
    manifest = {}
    for path in sorted((ROOT / "src/resdag").rglob("*")):
        if path.is_file() and (path.suffix == ".py" or path.name == "py.typed"):
            relative = path.relative_to(ROOT)
            target = source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            manifest[str(relative)] = _hash(path.read_bytes())
    sys.path.insert(0, str(source / "src"))
    import resdag as rd

    if not Path(rd.__file__).resolve().is_relative_to(source.resolve()):
        raise RuntimeError("Experiment must import the frozen source snapshot")
    script = Path(__file__).read_bytes()
    (args.output / "runner.py").write_bytes(script)
    helper = ROOT / "benchmarks/rc_bench/_threads.py"
    (args.output / "thread-policy.py").write_bytes(helper.read_bytes())
    previous = ROOT / "docs/assets/data/phase-five-independent-trajectories-2026-10-23.json"
    (args.output / "incumbent-provenance.json").write_bytes(previous.read_bytes())
    trials = min(args.trials, 2) if args.quick else args.trials
    plan = _plan(args.quick, trials, args.timeout, np)
    incumbents, predecessor = _incumbents(previous, args.quick)
    plan.update(
        {
            "source_files": manifest,
            "source_manifest_sha256": _identity(manifest),
            "runner_sha256": _hash(script),
            "thread_helper_sha256": _hash(helper.read_bytes()),
            "incumbents": incumbents,
            "incumbent_provenance": predecessor,
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "versions": {
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "numpy",
                    "scipy",
                    "optuna",
                    "tsdynamics",
                    "pytorch-symbolic",
                    "threadpoolctl",
                )
            },
            "python": platform.python_version(),
        }
    )
    _write(args.output / "preregistered.json", plan)
    plan_hash = _hash((args.output / "preregistered.json").read_bytes())
    validation = args.output / "validation"
    final = args.output / "final"
    validation.mkdir()
    final.mkdir()
    results: dict[str, Any] = {
        "plan_sha256": plan_hash,
        "validation": {},
        "winners": {},
        "final": {},
    }
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with threads.single_thread_policy() as policy:
        results["native_policy"] = policy
        for name in SYSTEMS:
            config = plan["systems"][name]
            datasets = _generate(name, config, config["validation_ics"], validation, ts, np)
            records = []
            results["validation"][name] = {"trials": records}
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=plan["search_sampler_seed"]),
                pruner=optuna.pruners.NopPruner(),
            )
            for flag in (True, False):
                study.enqueue_trial({**incumbents[name], "input_to_readout": flag})

            def objective(trial: Any) -> float:
                params = _search(trial, plan)
                record: dict[str, Any] = {"trial": trial.number, "params": params, "evaluation": {}}
                records.append(record)
                try:
                    _evaluate(
                        params,
                        datasets,
                        config,
                        plan["validation_seeds"],
                        rd,
                        torch,
                        np,
                        record["evaluation"],
                    )
                    record["status"] = "complete"
                    print(
                        json.dumps(
                            {
                                "stage": "validation trial",
                                "system": name,
                                "trial": trial.number,
                                "score": record["evaluation"]["summary"]["0.3"][
                                    "median_of_trajectory_medians"
                                ],
                            }
                        ),
                        flush=True,
                    )
                    return record["evaluation"]["summary"]["0.3"]["median_of_trajectory_medians"]
                except (ArithmeticError, RuntimeError, ValueError) as error:
                    record["status"] = "failed"
                    record["error"] = f"{type(error).__name__}: {error}"
                    raise
                finally:
                    _write(args.output / "validation-progress.json", results)

            started = time.perf_counter()
            study.optimize(
                objective,
                n_trials=trials,
                timeout=args.timeout,
                catch=(ArithmeticError, RuntimeError, ValueError),
            )
            states = {str(trial.number): trial.state.name for trial in study.trials}
            paired = _paired_baselines(records, states)
            results["validation"][name] = {
                "datasets": [
                    {**item, "path": str(item["path"].relative_to(args.output))}
                    for item in datasets
                ],
                "trials": records,
                "states": states,
                "controlled_feature_comparison": paired,
                "search_seconds": time.perf_counter() - started,
            }
            results["winners"][name] = {
                "trial": study.best_trial.number,
                "params": study.best_params,
                "validation_score": study.best_value,
            }
            _write(args.output / "validation-results.json", results)
            print(
                json.dumps(
                    {
                        "stage": "validation complete",
                        "system": name,
                        "trials": len(study.trials),
                        "winner": results["winners"][name],
                    }
                ),
                flush=True,
            )
        frozen = {
            "plan_sha256": plan_hash,
            "winners": results["winners"],
            "test_used_for_selection": False,
        }
        _write(args.output / "frozen-winners.json", frozen)
        winners_hash = _hash((args.output / "frozen-winners.json").read_bytes())
        for name in SYSTEMS:
            config = plan["systems"][name]
            datasets = _generate(name, config, config["final_ics"], final, ts, np)
            results["final"][name] = {
                "datasets": [
                    {**item, "path": str(item["path"].relative_to(args.output))}
                    for item in datasets
                ]
            }
            for variant, params in (
                ("incumbent", incumbents[name]),
                ("winner", results["winners"][name]["params"]),
            ):
                evaluation = results["final"][name][variant] = {}
                try:
                    _evaluate(
                        params, datasets, config, plan["final_seeds"], rd, torch, np, evaluation
                    )
                except Exception as error:
                    evaluation["error"] = f"{type(error).__name__}: {error}"
                    _write(args.output / "failure-report.json", results)
                    raise
                _write(args.output / "final-progress.json", results)
            print(json.dumps({"stage": "final complete", "system": name}), flush=True)
        if (
            _hash((args.output / "frozen-winners.json").read_bytes()) != winners_hash
            or _hash((args.output / "preregistered.json").read_bytes()) != plan_hash
        ):
            raise RuntimeError("Frozen protocol or winners changed")
        for relative, expected in manifest.items():
            if _hash((source / relative).read_bytes()) != expected:
                raise RuntimeError("Frozen source snapshot changed")
        for name, module in tuple(sys.modules.items()):
            if name == "resdag" or name.startswith("resdag."):
                location = getattr(module, "__file__", None)
                if location and not Path(location).resolve().is_relative_to(source.resolve()):
                    raise RuntimeError(f"Mixed source import: {name}")
        results["source_imports_verified"] = True
    results["winners_sha256"] = winners_hash
    results["scope"] = "recipe robustness; timing observations are not a controlled speed ranking"
    _write(args.output / "report.json", results)


if __name__ == "__main__":
    main()
