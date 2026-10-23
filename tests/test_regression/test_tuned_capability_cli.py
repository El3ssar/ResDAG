"""Capability CLI guards without training models or invoking tuning workers."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shlex
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "tuned_accuracy.py"
SEEDS = (401, 402, 403, 404, 405)
COMMIT = "f" * 40


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location("tuned_capability_test", PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path, value):
    path.write_text(json.dumps(value))


def _argv(tmp_path):
    return [
        str(PATH),
        "--mode",
        "capability",
        "--after",
        "chosen-source",
        "--test-seeds",
        *(str(seed) for seed in SEEDS),
        "--n-trials",
        "2",
        "--artifacts-dir",
        str(tmp_path / "artifacts"),
        "--output",
        str(tmp_path / "report.json"),
    ]


@pytest.fixture
def fake_experiment(tmp_path, monkeypatch, runner):
    state = SimpleNamespace(
        actions=[],
        bootstrap=0,
        policies=0,
        mutate=None,
        dependency_change_stage=None,
        dependencies={
            package: f"observed-{package}"
            for package in (
                "torch",
                "numpy",
                "scipy",
                "optuna",
                "joblib",
                "tsdynamics",
                "threadpoolctl",
            )
        },
    )
    policy_path = tmp_path / "synthetic_threads.py"
    policy_path.write_text("# copied stdlib policy placeholder\n")

    class Threads:
        __file__ = str(policy_path)
        THREAD_VARIABLES = (
            "VECLIB_MAXIMUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OMP_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "JULIA_NUM_THREADS",
        )

        @staticmethod
        def bootstrap_cli():
            state.bootstrap += 1

        @staticmethod
        @contextmanager
        def single_thread_policy():
            state.policies += 1
            yield {"synthetic": True}

    def archive(reference, directory):
        state.reference = reference
        module = directory / "src" / "resdag" / "__init__.py"
        module.parent.mkdir(parents=True)
        module.write_text("# immutable source placeholder\n")
        (directory / "commit.txt").write_text(COMMIT)
        return COMMIT

    def data(directory, supplied=None):
        (directory / "data.json").write_text("{}")
        for system in ("lorenz", "rossler"):
            (directory / f"{system}.npy").write_bytes(b"immutable observations")
        return {"synthetic": True}

    def run(command, *, env, check, stdout, stderr):
        assert check is True
        assert all(env[name] == "1" for name in Threads.THREAD_VARIABLES)
        assert env["PYTHONPATH"] == ""
        action = command[command.index("--worker") + 1]
        assert action in ("tune", "after")
        assert command[command.index("--after") + 1] == COMMIT
        index = command.index("--test-seeds") + 1
        assert tuple(int(seed) for seed in command[index : index + len(SEEDS)]) == SEEDS
        state.actions.append(action)
        directory = Path(command[command.index("--artifacts-dir") + 1])
        if action == "tune":
            budget = {
                "attempts": 2,
                "runs_per_complete_trial": 9,
                "state_counts": {"COMPLETE": 2, "FAIL": 0, "PRUNED": 0, "RUNNING": 0, "WAITING": 0},
            }
            winners = {
                f"{system}-{family}": {"budget": budget, "params": {"width": 8}}
                for system in ("lorenz", "rossler")
                for family in ("classic", "ott")
            }
            if state.mutate == "budget":
                budget["runs_per_complete_trial"] = 1
            _write(
                directory / "frozen-winners.json",
                {"winners": winners, "test_used_for_selection": False},
            )
        else:
            rows = []
            for system in ("lorenz", "rossler"):
                variants = (
                    ("unoptimized", "winner", "known_recipe")
                    if system == "rossler"
                    else ("unoptimized", "winner")
                )
                for family in ("classic", "ott"):
                    for variant in variants:
                        for seed in SEEDS:
                            for origin in range(3):
                                rows.append(
                                    {
                                        "system": system,
                                        "family": family,
                                        "variant": variant,
                                        "seed": seed,
                                        "origin_id": origin,
                                        "origin_index": runner._protocol(system)["test_origins"][
                                            origin
                                        ],
                                        "history_steps": runner._protocol(system)["test_origins"][
                                            origin
                                        ],
                                        "scores": {
                                            threshold: {
                                                "lyapunov_times": float(origin + 1),
                                                "right_censored": False,
                                            }
                                            for threshold in ("0.3", "0.5")
                                        },
                                    }
                                )
            if state.mutate == "cases":
                rows[-1] = rows[0]
            elif state.mutate == "clock":
                rows[-1]["history_steps"] -= 1
            _write(
                directory / "after-test.json",
                {
                    "rows": rows,
                    "winners_sha256": runner.HELPERS._digest(
                        (directory / "frozen-winners.json").read_bytes()
                    ),
                },
            )
            if state.mutate == "winner":
                with (directory / "frozen-winners.json").open("a") as file:
                    file.write(" ")
            if state.mutate == "input":
                (directory / "lorenz.npy").write_bytes(b"changed observations")
            elif state.mutate == "worker-source":
                (directory / "runner" / "tuned_accuracy.py").write_text("# changed worker")
            elif state.mutate == "library-source":
                (directory / "after" / "src" / "resdag" / "__init__.py").write_text(
                    "# changed library"
                )
            elif state.mutate == "output":
                Path(command[command.index("--output") + 1]).write_text("keep late output")
        _write(directory / f"{action}-thread-policy.json", {"synthetic": True})
        if state.dependency_change_stage == action:
            state.dependencies["numpy"] = "changed-during-worker"

    monkeypatch.setattr(runner, "_thread_helpers", lambda: Threads)
    monkeypatch.setattr(runner, "_capability_data", data)
    monkeypatch.setattr(runner, "_data", data)
    monkeypatch.setattr(runner, "_dependency_versions", lambda: dict(state.dependencies))
    monkeypatch.setattr(runner.platform, "platform", lambda: "synthetic-platform")
    monkeypatch.setattr(runner.HELPERS, "_archive", archive)
    monkeypatch.setattr(runner.subprocess, "run", run)
    state.runner = runner
    return state


def test_capability_uses_existing_stages_and_reports_no_revision_parity(
    tmp_path, monkeypatch, fake_experiment
):
    state = fake_experiment
    monkeypatch.setattr(sys, "argv", _argv(tmp_path))
    state.runner.main()
    assert state.actions == ["tune", "after"]
    assert state.reference == "chosen-source"
    assert state.bootstrap == 1 and state.policies == 1
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["source_commit"] == COMMIT
    assert report["test_seeds"] == list(SEEDS)
    assert report["dependency_versions"] == state.dependencies
    manifest = json.loads((tmp_path / "artifacts" / "capability-manifest.json").read_text())
    assert manifest["dependency_versions"] == state.dependencies
    assert len(report["after"]["rows"]) == 150
    assert len(report["validated_study_budgets"]) == 4
    for field in (
        "before",
        "maximum_revision_forecast_difference",
        "scores_identical",
        "fitted_tensor_fingerprints_identical",
    ):
        assert field not in report
    command = shlex.split(report["reproduce_command"])
    assert command[:2] == ["python", "benchmarks/tuned_accuracy.py"]
    assert command[command.index("--after") + 1] == COMMIT
    assert "FROZEN_DATA_DIRECTORY" in command
    inputs = report["prepared_input_sha256"]
    assert {
        "runner/tuned_accuracy.py",
        "runner/accuracy_regression.py",
        "runner/rc_bench/_threads.py",
    } <= inputs.keys()


def test_distribution_versions_are_observed_without_importing_packages(runner, monkeypatch):
    calls = []

    def installed_version(package):
        calls.append(package)
        if package == "tsdynamics":
            raise runner.importlib.metadata.PackageNotFoundError(package)
        return f"installed-{package}"

    monkeypatch.setattr(runner.importlib.metadata, "version", installed_version)
    observed = runner._dependency_versions()
    assert calls == ["torch", "numpy", "scipy", "optuna", "joblib", "tsdynamics", "threadpoolctl"]
    assert observed == {
        package: None if package == "tsdynamics" else f"installed-{package}" for package in calls
    }


@pytest.mark.parametrize("stage", ["tune", "after"])
def test_dependency_version_change_stops_execution_without_report(
    tmp_path, monkeypatch, fake_experiment, stage
):
    fake_experiment.dependency_change_stage = stage
    monkeypatch.setattr(sys, "argv", _argv(tmp_path))
    with pytest.raises(RuntimeError, match="dependency versions changed"):
        fake_experiment.runner.main()
    assert fake_experiment.actions == (["tune"] if stage == "tune" else ["tune", "after"])
    assert not (tmp_path / "report.json").exists()


@pytest.mark.parametrize(
    "mutation", ["winner", "input", "worker-source", "library-source", "budget", "cases", "clock"]
)
def test_changed_inputs_winner_budget_or_test_cases_cannot_produce_report(
    tmp_path, monkeypatch, fake_experiment, mutation
):
    fake_experiment.mutate = mutation
    monkeypatch.setattr(sys, "argv", _argv(tmp_path))
    with pytest.raises(RuntimeError):
        fake_experiment.runner.main()
    assert not (tmp_path / "report.json").exists()


def test_nonempty_artifact_directory_is_rejected_before_replacing_data(
    tmp_path, monkeypatch, fake_experiment
):
    directory = tmp_path / "artifacts"
    directory.mkdir()
    sentinel = directory / "lorenz.npy"
    sentinel.write_bytes(b"keep existing observations")
    monkeypatch.setattr(sys, "argv", _argv(tmp_path))
    with pytest.raises(ValueError, match="fresh artifacts"):
        fake_experiment.runner.main()
    assert sentinel.read_bytes() == b"keep existing observations"
    assert not fake_experiment.actions


@pytest.mark.parametrize(
    "relative",
    [
        ".",
        "after-test.json",
        "after-test.json/report.json",
        "data.json",
        "frozen-winners.json",
        "capability-manifest.json",
        "runner/tuned_accuracy.py",
        "after/src/resdag/report.json",
        "tune.log",
        "after-thread-policy.json",
        "lorenz-classic.sqlite3",
        "lorenz-classic.sqlite3-wal",
        "rossler-ott-tuning.json",
        "rossler-ott-winner-seed401.pt",
    ],
)
def test_report_cannot_replace_reserved_evidence(tmp_path, monkeypatch, fake_experiment, relative):
    arguments = _argv(tmp_path)
    arguments[arguments.index("--output") + 1] = str(tmp_path / "artifacts" / relative)
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(ValueError, match="output report"):
        fake_experiment.runner.main()
    assert not fake_experiment.actions
    assert not (tmp_path / "artifacts").exists()


def test_report_cannot_be_a_fresh_ancestor_of_the_artifact_directory(
    tmp_path, monkeypatch, fake_experiment
):
    arguments = _argv(tmp_path)
    arguments[arguments.index("--artifacts-dir") + 1] = str(tmp_path / "fresh" / "artifacts")
    arguments[arguments.index("--output") + 1] = str(tmp_path / "fresh")
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(ValueError, match="output report"):
        fake_experiment.runner.main()
    assert not fake_experiment.actions
    assert not (tmp_path / "fresh").exists()


def test_report_parent_file_is_rejected_before_workers(tmp_path, monkeypatch, fake_experiment):
    blocked = tmp_path / "blocked"
    blocked.write_text("keep parent file")
    arguments = _argv(tmp_path)
    arguments[arguments.index("--output") + 1] = str(blocked / "report.json")
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(ValueError, match="parent must be a directory"):
        fake_experiment.runner.main()
    assert not fake_experiment.actions
    assert blocked.read_text() == "keep parent file"


def test_safe_report_can_be_saved_inside_artifacts(tmp_path, monkeypatch, fake_experiment):
    report_path = tmp_path / "artifacts" / "reports" / "capability.json"
    arguments = _argv(tmp_path)
    arguments[arguments.index("--output") + 1] = str(report_path)
    monkeypatch.setattr(sys, "argv", arguments)
    fake_experiment.runner.main()
    report = json.loads(report_path.read_text())
    assert fake_experiment.actions == ["tune", "after"]
    assert "reports/capability.json" not in report["artifact_files"]
    for name, evidence in report["artifact_files"].items():
        assert (
            hashlib.sha256((tmp_path / "artifacts" / name).read_bytes()).hexdigest()
            == evidence["sha256"]
        )


def test_report_created_during_workers_is_preserved(tmp_path, monkeypatch, fake_experiment):
    fake_experiment.mutate = "output"
    monkeypatch.setattr(sys, "argv", _argv(tmp_path))
    with pytest.raises(ValueError, match="appeared during execution"):
        fake_experiment.runner.main()
    assert (tmp_path / "report.json").read_text() == "keep late output"


@pytest.mark.parametrize(
    "values", [None, [], [401, 401], [7], [42], [12345], [101], [-1], [2**63 - 1]]
)
def test_final_seed_bank_rejects_ambiguous_invalid_or_previously_used_roles(runner, values):
    with pytest.raises(ValueError):
        runner._capability_seeds(values, quick=False)
    assert runner._capability_seeds(list(SEEDS), quick=False) == SEEDS


def test_quick_smoke_cannot_consume_the_reserved_final_bank(runner):
    assert runner._capability_seeds(None, quick=True) == (101,)
    assert runner._capability_seeds([101], quick=True) == (101,)
    with pytest.raises(ValueError, match="smoke seed"):
        runner._capability_seeds(list(SEEDS), quick=True)


@pytest.mark.parametrize(
    "options",
    [
        ["--mode", "capability", "--test-seeds", "401"],
        ["--mode", "capability", "--after", "source"],
        ["--mode", "capability", "--after", "source", "--before", "other", "--test-seeds", "401"],
        ["--mode", "capability", "--after", "source", "--worker", "before", "--test-seeds", "401"],
        ["--test-seeds", "401"],
    ],
)
def test_ambiguous_source_or_mode_flags_fail_before_bootstrap(
    tmp_path, monkeypatch, fake_experiment, options
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(PATH),
            *options,
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--output",
            str(tmp_path / "report.json"),
        ],
    )
    with pytest.raises(SystemExit) as error:
        fake_experiment.runner.main()
    assert error.value.code == 2
    assert not fake_experiment.actions and fake_experiment.bootstrap == 0


@pytest.mark.parametrize("action", ["tune", "after"])
def test_worker_receives_explicit_seeds_and_verified_policy(
    tmp_path, monkeypatch, fake_experiment, action
):
    calls = []
    runner = fake_experiment.runner
    monkeypatch.setattr(
        runner,
        "_tune",
        lambda directory, n_trials, quick: calls.append(("tune", directory, n_trials, quick)),
    )
    monkeypatch.setattr(
        runner,
        "_evaluate",
        lambda directory, label, quick, *, test_seeds=None: calls.append(
            (label, directory, quick, test_seeds)
        ),
    )
    directory = tmp_path / "artifacts"
    directory.mkdir()
    monkeypatch.setattr(sys, "argv", _argv(tmp_path) + ["--worker", action])
    runner.main()
    assert fake_experiment.policies == 1
    if action == "after":
        assert calls == [("after", directory, False, SEEDS)]
    else:
        assert calls == [("tune", directory, 2, False)]
    assert json.loads((directory / f"{action}-thread-policy.json").read_text()) == {
        "synthetic": True
    }


def test_imports_does_not_reset_already_verified_thread_settings(tmp_path, monkeypatch, runner):
    source = tmp_path / "after" / "src" / "resdag" / "__init__.py"
    torch = SimpleNamespace(get_num_threads=lambda: 1, get_num_interop_threads=lambda: 1)
    rd = SimpleNamespace(__file__=str(source))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "resdag", rd)
    monkeypatch.setattr(sys, "path", list(sys.path))
    assert runner._imports(tmp_path, "after") == (rd, torch)


@pytest.mark.parametrize("corruption", [None, "hash", "shape", "clock", "dtype", "nonfinite"])
def test_frozen_data_admission_checks_actual_values_and_recorded_protocol(
    tmp_path, monkeypatch, runner, corruption
):
    supplied, output = tmp_path / "supplied", tmp_path / "out"
    supplied.mkdir()
    output.mkdir()
    protocol = {"test": [0, 4], "dt": 0.1}
    monkeypatch.setattr(runner, "_protocol", lambda system: protocol)
    data = {}
    for system in ("lorenz", "rossler"):
        raw = np.arange(12, dtype=np.float64).reshape(4, 3)
        if system == "lorenz":
            if corruption == "shape":
                raw = raw[:3]
            elif corruption == "dtype":
                raw = raw.astype(np.float32)
            elif corruption == "nonfinite":
                raw[0, 0] = np.nan
        np.save(supplied / f"{system}.npy", raw)
        data[system] = {
            "protocol": protocol,
            "shape": list(raw.shape),
            "raw_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
        }
    if corruption == "hash":
        data["lorenz"]["raw_sha256"] = "incorrect"
    elif corruption == "clock":
        data["lorenz"]["protocol"] = {"test": [0, 4], "dt": 0.2}
    _write(supplied / "data.json", data)
    if corruption is None:
        assert runner._capability_data(output, supplied) == data
        for name in ("data.json", "lorenz.npy", "rossler.npy"):
            assert (output / name).read_bytes() == (supplied / name).read_bytes()
    else:
        with pytest.raises(ValueError, match="Frozen lorenz data"):
            runner._capability_data(output, supplied)


def test_matched_mode_keeps_initial_weight_parity_guard(tmp_path, monkeypatch, fake_experiment):
    state = fake_experiment

    def matched_worker(command, **kwargs):
        action = command[command.index("--worker") + 1]
        state.actions.append(action)
        if action != "tune":
            directory = Path(command[command.index("--artifacts-dir") + 1])
            _write(
                directory / f"{action}-test.json",
                {
                    "winners_sha256": "same-winner",
                    "rows": [{"id": "same-case"}],
                    "fingerprints": {"same-case": {"initial": action}},
                },
            )

    monkeypatch.setattr(state.runner.subprocess, "run", matched_worker)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(PATH),
            "--before",
            "old-source",
            "--after",
            "new-source",
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--output",
            str(tmp_path / "report.json"),
        ],
    )
    with pytest.raises(RuntimeError, match="same initial weights"):
        state.runner.main()
    assert state.actions == ["tune", "before", "after"]
    assert state.bootstrap == state.policies == 0
