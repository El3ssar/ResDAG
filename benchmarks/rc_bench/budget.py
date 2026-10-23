"""Equal wall-clock tuning budgets with Optuna and untouched final tests.

Run ``python -m benchmarks.rc_bench.budget --help``. The normal protocol gives
each library/system 600 seconds, three training seeds and three validation
origins. A separate small ``--smoke`` protocol checks plumbing, never rankings.
"""

from __future__ import annotations

# ruff: noqa: E402 -- CLI thread limits must precede numerical package imports.
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

# Enforce the declared CPU resource policy before importing NumPy/BLAS/Torch.
if __name__ == "__main__":
    from ._threads import bootstrap_cli

    bootstrap_cli()
    os.environ["TQDM_DISABLE"] = "1"

import numpy as np
import optuna

from ._threads import ThreadPolicyError, single_thread_policy, verify_thread_policy
from .budget_adapters import CommonParams, julia_storage_pilot, prepare_adapter
from .config import Point
from .data import prepare_point
from .metrics import valid_prediction_time

TRAIN_SEEDS = (7, 42, 12345)
TEST_SEEDS = (301, 302, 303, 304, 305)
LIBRARIES = ("resdag-cpu", "reservoirpy", "rcjl")
SYSTEMS = ("lorenz", "rossler")


class BudgetExpiredError(RuntimeError):
    """Candidate work reached the common wall-clock deadline; never eligible."""


class CandidateFailureError(RuntimeError):
    """A recorded candidate failed; Optuna may continue within the same budget."""


@dataclass(frozen=True)
class Protocol:
    system: str
    train_stop: int
    washout: int
    horizon: int
    dt: float
    lambda_max: float
    validation_origins: tuple[int, ...]
    test_origins: tuple[int, ...]

    @classmethod
    def canonical(cls, system: str, smoke: bool = False) -> Protocol:
        if system not in SYSTEMS:
            raise ValueError(f"Unknown system: {system}")
        dt, exponent, train = (0.02, 0.9056, 10001) if system == "lorenz" else (0.145, 0.0714, 8501)
        horizon = math.ceil(10 / (dt * exponent))
        washout = 500
        if smoke:
            train, horizon, washout = 61, 8, 5
        validation = tuple(train + index * horizon for index in range(3))
        test = tuple(train + (3 + index) * horizon for index in range(3))
        return cls(system, train, washout, horizon, dt, exponent, validation, test)

    @property
    def validation_stop(self) -> int:
        return self.validation_origins[-1] + self.horizon

    @property
    def test_stop(self) -> int:
        return self.test_origins[-1] + self.horizon


@dataclass
class Deadline:
    seconds: float
    clock: Callable[[], float] = time.perf_counter

    def __post_init__(self):
        if not math.isfinite(self.seconds) or self.seconds <= 0:
            raise ValueError("budget seconds must be finite and positive")
        self.started = self.clock()

    @property
    def elapsed(self) -> float:
        return self.clock() - self.started

    def check(self) -> None:
        if self.elapsed >= self.seconds:
            raise BudgetExpiredError("candidate reached the equal wall-clock deadline")


def search(trial: Any, smoke: bool = False) -> CommonParams:
    return CommonParams(
        reservoir_size=trial.suggest_categorical(
            "reservoir_size", [8, 16] if smoke else [300, 400, 600, 1000]
        ),
        spectral_radius=trial.suggest_float("spectral_radius", 0.05, 1.4),
        leak_rate=trial.suggest_float("leak_rate", 0.1, 1.0),
        input_scaling=trial.suggest_float("input_scaling", 0.01, 1.0, log=True),
        bias_scaling=trial.suggest_float("bias_scaling", 0.01, 0.5, log=True),
        mean_degree=trial.suggest_categorical("mean_degree", [3, 6] if smoke else [3, 6, 12]),
        alpha=trial.suggest_float("alpha", 1e-12, 1e-3, log=True),
    )


def score(prediction: np.ndarray, truth: np.ndarray, protocol: Protocol) -> dict:
    if prediction.shape != truth.shape:
        raise ValueError(f"Forecast shape {prediction.shape} does not match {truth.shape}")
    result = {"steps": len(prediction), "diverged": bool(not np.isfinite(prediction).all())}
    for threshold in (0.3, 0.5):
        steps = valid_prediction_time(prediction, truth, threshold, scale=np.ones(truth.shape[1]))
        result[str(threshold)] = {
            "valid_steps": steps,
            "lyapunov_times": steps * protocol.dt * protocol.lambda_max,
            "censored": steps == len(prediction),
        }
    return result


def evaluate_candidate(
    adapter,
    values: np.ndarray,
    params: CommonParams,
    protocol: Protocol,
    seeds: tuple[int, ...],
    origins: tuple[int, ...],
    *,
    deadline: Deadline | None = None,
    ledger: dict | None = None,
    keep_errors: bool = False,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Fit once per seed; only observed history reaches the fitted model."""
    ledger = ledger if ledger is not None else {"fits": [], "forecasts": [], "runs": []}
    predictions = {}
    check = deadline.check if deadline is not None else lambda: None
    training = np.ascontiguousarray(values[: protocol.train_stop])
    for seed in seeds:
        verify_thread_policy()
        check()
        fitted = {"seed": seed, "started_elapsed": deadline.elapsed if deadline else None}
        ledger["fits"].append(fitted)
        started = time.perf_counter()
        try:
            candidate, timing = adapter.fit(training, params, seed, protocol.washout)
            fitted.update(timing, complete=True)
        except ThreadPolicyError:
            raise
        except Exception as exception:
            fitted.update(complete=False, failure=f"{type(exception).__name__}: {exception}")
            if not keep_errors:
                raise
            for origin in origins:
                ledger["runs"].append(
                    {
                        "seed": seed,
                        "origin": origin,
                        "status": "failed",
                        "failure": "fit failed: " + fitted["failure"],
                    }
                )
            continue
        finally:
            fitted["wall_seconds"] = time.perf_counter() - started
            verify_thread_policy()
        check()
        for origin in origins:
            check()
            operation = {"seed": seed, "origin": origin, "horizon": protocol.horizon}
            ledger["forecasts"].append(operation)
            # The model cannot inspect validation/test truth: its only input is
            # the observed washout-length context ending before the origin.
            history = np.ascontiguousarray(values[origin - protocol.washout : origin])
            started = time.perf_counter()
            try:
                prediction, timing = candidate.forecast(history, protocol.horizon)
                result = score(prediction, values[origin : origin + protocol.horizon], protocol)
                operation.update(timing, complete=True)
            except ThreadPolicyError:
                raise
            except Exception as exception:
                operation.update(complete=False, failure=f"{type(exception).__name__}: {exception}")
                if not keep_errors:
                    raise
                ledger["runs"].append(
                    {
                        "seed": seed,
                        "origin": origin,
                        "status": "failed",
                        "failure": operation["failure"],
                    }
                )
                continue
            finally:
                operation["wall_seconds"] = time.perf_counter() - started
                verify_thread_policy()
            check()
            ledger["runs"].append({"seed": seed, "origin": origin, "status": "complete", **result})
            predictions[f"seed{seed}-origin{origin}"] = prediction
    check()
    return ledger, predictions


def _summary(runs: list[dict]) -> dict:
    result = {}
    for threshold in ("0.3", "0.5"):
        values = [row[threshold]["lyapunov_times"] for row in runs]
        result[threshold] = {
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "all_values": values,
            "censored_count": sum(row[threshold]["censored"] for row in runs),
        }
    return result | {"runs": len(runs), "diverged_count": sum(row["diverged"] for row in runs)}


def tune_track(
    adapter,
    values: np.ndarray,
    protocol: Protocol,
    study,
    seconds: float,
    *,
    smoke: bool = False,
    max_trials: int | None = None,
    deadline: Deadline | None = None,
) -> dict:
    """Standard Optuna optimization; unfinished/late trials cannot win."""
    if len(values) != protocol.validation_stop:
        raise ValueError(
            "tuning receives training and validation only, never final test observations"
        )
    deadline = deadline or Deadline(seconds)

    def objective(trial):
        ledger = {"fits": [], "forecasts": [], "runs": []}
        trial.set_user_attr("started_elapsed", deadline.elapsed)
        try:
            deadline.check()
            params = search(trial, smoke)
            evaluate_candidate(
                adapter,
                values,
                params,
                protocol,
                TRAIN_SEEDS,
                protocol.validation_origins,
                deadline=deadline,
                ledger=ledger,
            )
            summary = _summary(ledger["runs"])
            trial.set_user_attr("work", ledger)
            trial.set_user_attr("summary", summary)
            deadline.check()
            trial.set_user_attr("finished_elapsed", deadline.elapsed)
            trial.set_user_attr("eligible", True)
            deadline.check()
            return summary["0.3"]["median"]
        except ThreadPolicyError:
            raise
        except Exception as exception:
            elapsed = deadline.elapsed
            trial.set_user_attr("work", ledger)
            trial.set_user_attr("eligible", False)
            trial.set_user_attr("finished_elapsed", elapsed)
            trial.set_user_attr("overrun_seconds", max(0.0, elapsed - seconds))
            trial.set_user_attr("failure", f"{type(exception).__name__}: {exception}")
            if isinstance(exception, BudgetExpiredError):
                trial.study.stop()
                raise
            raise CandidateFailureError(str(exception)) from exception

    study.optimize(
        objective,
        timeout=max(0.0, seconds - deadline.elapsed),
        n_trials=max_trials,
        n_jobs=1,
        catch=(CandidateFailureError, BudgetExpiredError),
    )
    elapsed = deadline.elapsed
    eligible = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.user_attrs.get("eligible")
        and trial.user_attrs["finished_elapsed"] < seconds
    ]
    winner = max(eligible, key=lambda trial: (trial.value, -trial.number)) if eligible else None
    counts = Counter(trial.state.name for trial in study.trials)
    return {
        "budget_seconds": seconds,
        "elapsed_seconds": elapsed,
        "overrun_seconds": max(0.0, elapsed - seconds),
        "max_trials": max_trials,
        "attempts": len(study.trials),
        "state_counts": dict(counts),
        "eligible_complete_trials": len(eligible),
        "fit_attempts": sum(
            len(t.user_attrs.get("work", {}).get("fits", [])) for t in study.trials
        ),
        "forecast_attempts": sum(
            len(t.user_attrs.get("work", {}).get("forecasts", [])) for t in study.trials
        ),
        "winner": (
            None
            if winner is None
            else {
                "number": winner.number,
                "params": winner.params,
                "validation": winner.user_attrs["summary"],
            }
        ),
        "trials": [
            {
                "number": t.number,
                "state": t.state.name,
                "value": t.value,
                "params": t.params,
                **t.user_attrs,
            }
            for t in study.trials
        ],
    }


def _write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _scaled(values: np.ndarray, protocol: Protocol) -> tuple[np.ndarray, dict]:
    point = Point(
        reservoir_size=8,
        train_len=protocol.train_stop - 1,
        warmup=protocol.washout,
        horizon=protocol.horizon,
    )
    return prepare_point(values, point)


def _versions() -> dict:
    import resdag

    result = {}
    for package in ("resdag", "torch", "numpy", "scipy", "optuna", "reservoirpy", "tsdynamics"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    result["resdag_distribution_metadata"] = result["resdag"]
    result["resdag"] = resdag.__version__
    return result


def _source_manifest() -> dict:
    import resdag

    root = Path(__file__).resolve().parents[2]
    runtime = root / "src/resdag"
    imports = {
        name: str(Path(module.__file__).resolve())
        for name, module in sys.modules.items()
        if (name == "resdag" or name.startswith("resdag.")) and getattr(module, "__file__", None)
    }
    if not all(Path(origin).is_relative_to(runtime) for origin in imports.values()):
        raise RuntimeError(
            "Benchmark imports must come from this source checkout; install it editable first"
        )
    files = [
        *root.glob("src/resdag/**/*.py"),
        *root.glob("benchmarks/rc_bench/**/*.py"),
        *root.glob("benchmarks/julia/*.jl"),
        *root.glob("benchmarks/julia/*.toml"),
    ]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
    ).stdout
    return {
        "commit": commit,
        "dirty": bool(status),
        "platform": platform.platform(),
        "runtime_version": resdag.__version__,
        "runtime_import_origins": imports,
        "files_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files)
        },
    }


def render(report: dict) -> str:
    lines = [
        "# Equal tuning time: vanilla reservoir forecasting",
        "",
        "Each library/system has its own wall-clock budget. Startup and one fixed synthetic warmup are separate. "
        "Construction, fitting once per seed, all synchronization/rollout, scoring and IPC count toward tuning. "
        "Incomplete and deadline-overrun candidates cannot win. Final tests use five unseen seeds and three later origins.",
        "",
        "| System / library | Eligible / attempts | Search seconds / overrun | Test LT at 0.3: median [min,max] | Test LT at 0.5 |",
        "|---|---:|---:|---|---|",
    ]
    if report.get("smoke"):
        lines[1:1] = [
            "",
            "**Correctness smoke only: tiny models/short windows; these values are not a performance comparison.**",
        ]
    for key, track in report["tracks"].items():
        tuning = track["tuning"]
        result = track.get("test", {}).get("summary")
        cells = []
        for threshold in ("0.3", "0.5"):
            if result is None:
                cells.append("unavailable")
            else:
                metric = result[threshold]
                prefix = "≥" if metric["censored_count"] > result["runs"] // 2 else ""
                low_prefix = "≥" if metric["censored_count"] == result["runs"] else ""
                high_prefix = "≥" if metric["censored_count"] else ""
                cells.append(
                    f"{prefix}{metric['median']:.3f} [{low_prefix}{metric['min']:.3f}, {high_prefix}{metric['max']:.3f}]"
                )
        lines.append(
            f"| {key} | {tuning['eligible_complete_trials']}/{tuning['attempts']} | "
            f"{tuning['elapsed_seconds']:.2f} / {tuning['overrun_seconds']:.2f} | {' | '.join(cells)} |"
        )
    lines.extend(
        [
            "",
            "Scores use a fixed per-signal population standard deviation from admitted training labels, "
            "stopping at the first normalized RMS error ≥0.3 (secondary0.5). ≥ denotes right censoring. "
            "Full per-seed/origin values, divergence flags, failures and work accounting are in JSON and Optuna SQLite.",
            "",
            "Common model: Float64 CPU, tanh leaky state, zero initial memory, Gaussian recurrent weights, "
            "mean-degree/width directed density with self-loops, dense uniform input weights, random uniform reservoir bias; "
            "state-only linear readout, no input skip or readout intercept. Native RNG, sparse masks/storage and direct solvers differ. "
            "This tuned-winner comparison is separate from matched-configuration throughput tests.",
            "",
            report.get("evaluation_boundary", ""),
            "",
            "API/build failures make a final track incomplete and its scientific headline unavailable. "
            "They are never silently dropped or imputed as forecast horizons. Model-storage observations, "
            "retained-feature estimates and cumulative process RSS (including startup) are in the raw work records; "
            "they are not isolated per-trial peak-memory comparisons.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    with single_thread_policy() as policy:
        return _main(argv, policy)


def _main(argv: list[str] | None, thread_policy: dict) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=float, default=600.0)
    parser.add_argument("--libraries", nargs="+", choices=LIBRARIES, default=LIBRARIES)
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=SYSTEMS)
    parser.add_argument("--max-trials", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--julia-storage", choices=("auto", "dense", "sparse"), default="auto")
    args = parser.parse_args(argv)
    if not math.isfinite(args.budget_seconds) or args.budget_seconds <= 0:
        parser.error("--budget-seconds must be positive and finite")
    if args.max_trials is not None and args.max_trials < 1:
        parser.error("--max-trials must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "winners.json").exists():
        parser.error("output already contains frozen winners; choose a new output directory")
    if not (args.data_dir / "data.json").exists():
        from benchmarks.tuned_accuracy import _data

        args.data_dir.mkdir(parents=True, exist_ok=True)
        _data(args.data_dir)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pilot = None
    julia_storage = args.julia_storage
    if "rcjl" in args.libraries and julia_storage == "auto":
        pilot = julia_storage_pilot()
        julia_storage = pilot["selection"]
    report = {
        "protocol": "equal-wall-time-v2",
        "smoke": args.smoke,
        "thread_policy": thread_policy,
        "versions": _versions(),
        "source_data": json.loads((args.data_dir / "data.json").read_text()),
        "source": _source_manifest(),
        "julia_storage": julia_storage if "rcjl" in args.libraries else None,
        "julia_storage_pilot": pilot,
        "train_seeds": list(TRAIN_SEEDS),
        "reserved_test_seeds": list(TEST_SEEDS),
        "evaluation_boundary": "Fresh reserved initialization seeds301–305. Reused raw trajectories/origins are held out from this study, not historically unseen. One trajectory per system; no independent-trajectory generalization claim.",
        "tracks": {},
    }
    # No final-test values are read until every library/system winner is frozen.
    for system in args.systems:
        protocol = Protocol.canonical(system, args.smoke)
        raw = np.load(args.data_dir / f"{system}.npy", mmap_mode="r")
        if len(raw) < protocol.test_stop:
            raise ValueError(f"{system} requires {protocol.test_stop} raw observations")
        validation_raw = np.array(raw[: protocol.validation_stop], copy=True)
        values, scaling = _scaled(validation_raw, protocol)
        for library in args.libraries:
            key = f"{system}-{library}"
            adapter, startup = prepare_adapter(library, julia_storage=julia_storage)
            verify_thread_policy()
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=20261023),
                pruner=optuna.pruners.NopPruner(),
                storage=f"sqlite:///{(args.output_dir / (key + '.sqlite3')).resolve()}",
                study_name=key,
            )
            try:
                tuning = tune_track(
                    adapter,
                    values,
                    protocol,
                    study,
                    args.budget_seconds,
                    smoke=args.smoke,
                    max_trials=args.max_trials,
                )
            finally:
                adapter.close()
            report["tracks"][key] = {
                "protocol": asdict(protocol),
                "normalization": scaling["normalization"],
                "validation_data_sha256": hashlib.sha256(validation_raw.tobytes()).hexdigest(),
                "startup": startup,
                "tuning": tuning,
            }
            _write(args.output_dir / "tuning-progress.json", report)
    _write(args.output_dir / "winners.json", report)
    frozen_sha = hashlib.sha256((args.output_dir / "winners.json").read_bytes()).hexdigest()
    report["frozen_winners_sha256"] = frozen_sha
    for key, track in report["tracks"].items():
        winner = track["tuning"]["winner"]
        if winner is None:
            continue
        system, library = key.split("-", 1)
        protocol = Protocol.canonical(system, args.smoke)
        raw = np.load(args.data_dir / f"{system}.npy", mmap_mode="r")
        values, _ = _scaled(np.array(raw[: protocol.test_stop], copy=True), protocol)
        track["test_data_sha256"] = hashlib.sha256(
            np.array(raw[protocol.validation_stop : protocol.test_stop]).tobytes()
        ).hexdigest()
        adapter, startup = prepare_adapter(library, julia_storage=julia_storage)
        verify_thread_policy()
        ledger = {"fits": [], "forecasts": [], "runs": []}
        try:
            ledger, predictions = evaluate_candidate(
                adapter,
                values,
                CommonParams(**winner["params"]),
                protocol,
                TEST_SEEDS,
                protocol.test_origins,
                ledger=ledger,
                keep_errors=True,
            )
            np.savez_compressed(args.output_dir / f"{key}-test-predictions.npz", **predictions)
            track["test"] = {
                "startup": startup,
                "work": ledger,
                "complete": all(row["status"] == "complete" for row in ledger["runs"]),
                "summary": (
                    _summary(ledger["runs"])
                    if all(row["status"] == "complete" for row in ledger["runs"])
                    else None
                ),
            }
        except ThreadPolicyError:
            raise
        except Exception as exception:
            track["test"] = {
                "failure": f"{type(exception).__name__}: {exception}",
                "complete": False,
                "work": ledger,
                "summary": None,
            }
        finally:
            adapter.close()
        assert (
            hashlib.sha256((args.output_dir / "winners.json").read_bytes()).hexdigest()
            == frozen_sha
        )
        _write(args.output_dir / "results.json", report)
    _write(args.output_dir / "results.json", report)
    (args.output_dir / "report.md").write_text(render(report))
    print(args.output_dir / "report.md")
    if report["source"]["files_sha256"] != _source_manifest()["files_sha256"]:
        raise RuntimeError(
            "Benchmark source changed during the experiment; results are not accepted"
        )
    return (
        0
        if all(track.get("test", {}).get("complete") for track in report["tracks"].values())
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
