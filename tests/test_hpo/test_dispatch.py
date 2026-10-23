"""Parent-owned studies, finite joblib windows and interruption cleanup."""

import importlib
import os
import threading
from contextlib import contextmanager

import optuna
import pytest
import torch

from resdag.hpo import run_hpo
from resdag.hpo.dispatch import _fail_active, run_trials
from resdag.hpo.objective import build_objective
from resdag.models import classic_esn


def _data():
    values = torch.sin(torch.arange(55, dtype=torch.float32).reshape(1, -1, 1) / 7)
    return dict(
        warmup=values[:, :5],
        train=values[:, 5:30],
        target=values[:, 6:31],
        f_warmup=values[:, 30:35],
        val=values[:, 35:45],
    )


def _runner(**options):
    return build_objective(
        model_creator=lambda size=8: classic_esn(
            reservoir_size=size, feedback_size=1, output_size=1
        ),
        search_space=lambda trial: {"size": 8},
        data_loader=lambda trial: _data(),
        loss_fn=lambda predicted, truth: float(((predicted - truth) ** 2).mean()),
        error_policy="fail",
        **options,
    )


@pytest.mark.parametrize("workers", [1, 2])
def test_notebook_closures_parent_callbacks_and_exact_attempt_budget(workers, tmp_path):
    parent = os.getpid()
    events = []
    path = str(tmp_path)
    offset = 1
    parent_only_lock = threading.Lock()

    def factory(size, fixed=0):
        from pathlib import Path

        Path(path, f"worker-{os.getpid()}").touch()
        assert fixed == offset
        return classic_esn(reservoir_size=size, feedback_size=1, output_size=1)

    def space(trial):
        assert os.getpid() == parent
        assert isinstance(trial, optuna.Trial)
        events.append(("space", trial.number))
        return {"size": trial.suggest_int("size", 7, 9), "fixed": offset}

    def loader(trial):
        assert os.getpid() == parent
        with parent_only_lock:
            events.append(("data", trial.number))
        return _data()

    def callback(trial, context):
        assert os.getpid() == parent
        assert isinstance(trial, optuna.Trial)
        assert context["params"]["fixed"] == offset
        with parent_only_lock:
            events.append(("callback", trial.number))

    study = run_hpo(
        factory,
        space,
        loader,
        n_trials=3,
        n_workers=workers,
        trial_callbacks=[callback],
        loss=lambda predicted, truth: float(((predicted - truth) ** 2).mean()),
        verbosity=0,
        seed=11,
    )
    assert len(study.trials) == 3
    assert all(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
    assert len(events) == 9
    pids = {int(file.name.split("-")[1]) for file in tmp_path.glob("worker-*")}
    assert pids and (pids == {parent} if workers == 1 else parent not in pids)
    assert all(trial.intermediate_values for trial in study.trials)


class _FirstCheckpoint(optuna.pruners.BasePruner):
    def __init__(self):
        self.parent = os.getpid()
        self.calls = []

    def prune(self, study, trial):
        assert os.getpid() == self.parent
        self.calls.append(trial.number)
        return bool(trial.intermediate_values)


@pytest.mark.parametrize("workers", [1, 2])
def test_in_memory_parent_pruner_releases_every_reservation(workers):
    pruner = _FirstCheckpoint()
    study = optuna.create_study(pruner=pruner)
    run_trials(study, _runner(n_checkpoints=4), n_trials=3, n_workers=workers, verbosity=0)
    assert len(study.trials) == 3
    assert all(trial.state == optuna.trial.TrialState.PRUNED for trial in study.trials)
    assert all(set(trial.intermediate_values) == {1} for trial in study.trials)
    assert sorted(pruner.calls) == [0, 1, 2]


@pytest.mark.parametrize("failure", [RuntimeError("executor startup"), KeyboardInterrupt()])
def test_executor_startup_or_interrupt_marks_only_owned_reservations_failed(monkeypatch, failure):
    dispatch = importlib.import_module("resdag.hpo.dispatch")
    study = optuna.create_study()
    foreign = study.ask()

    @contextmanager
    def fail(*args, **kwargs):
        raise failure
        yield

    monkeypatch.setattr(dispatch, "iter_jobs", fail)
    with pytest.raises(type(failure)) as caught:
        run_trials(study, _runner(), n_trials=5, n_workers=2, verbosity=0)
    assert caught.value is failure
    assert study.trials[foreign.number].state == optuna.trial.TrialState.RUNNING
    assert len(study.trials) == 3
    assert all(trial.state == optuna.trial.TrialState.FAIL for trial in study.trials[1:])


def test_fail_cleanup_attempts_every_trial_and_preserves_primary(monkeypatch):
    study = optuna.create_study()
    trials = {trial.number: trial for trial in (study.ask(), study.ask(), study.ask())}
    calls = []

    def fail(number, **kwargs):
        calls.append(number)
        raise OSError(f"cleanup {number}")

    monkeypatch.setattr(study, "tell", fail)
    primary = ValueError("primary")
    _fail_active(study, trials, primary)
    assert calls == [0, 1, 2]
    assert len(primary.__notes__) == 3
    with pytest.raises(OSError, match="cleanup 0") as caught:
        _fail_active(study, trials, None)
    assert calls == [0, 1, 2, 0, 1, 2]
    assert len(caught.value.__notes__) == 2


def test_failed_numerical_worker_leaves_no_running_and_next_run_works(monkeypatch):
    runner = _runner(catch_exceptions=False)

    def broken(**kwargs):
        raise ArithmeticError("numerical failure")

    runner.model_creator = broken
    study = optuna.create_study()
    with pytest.raises(ArithmeticError, match="numerical failure"):
        run_trials(study, runner, n_trials=3, n_workers=2, verbosity=0)
    assert all(trial.state != optuna.trial.TrialState.RUNNING for trial in study.trials)
    next_study = optuna.create_study()
    run_trials(next_study, _runner(), n_trials=1, n_workers=2, verbosity=0)
    assert next_study.best_value >= 0


def test_failed_preparation_consumes_exact_new_attempts():
    runner = _runner()

    def bad_data(trial):
        raise ValueError("bad prepared data")

    runner.data_loader = bad_data
    study = optuna.create_study()
    run_trials(study, runner, n_trials=5, n_workers=2, verbosity=0)
    assert len(study.trials) == 5
    assert all(trial.state == optuna.trial.TrialState.FAIL for trial in study.trials)


def test_unserializable_factory_releases_reserved_trials():
    lock = threading.Lock()

    def factory(size=8):
        with lock:
            return classic_esn(reservoir_size=size, feedback_size=1, output_size=1)

    runner = _runner()
    runner.model_creator = factory
    study = optuna.create_study()
    with pytest.raises(Exception, match="pickle|serializ"):
        run_trials(study, runner, n_trials=3, n_workers=2, verbosity=0)
    assert len(study.trials) == 2
    assert all(trial.state == optuna.trial.TrialState.FAIL for trial in study.trials)


@pytest.mark.parametrize("failure", [ArithmeticError("numerical error"), KeyboardInterrupt()])
def test_numerical_cleanup_preserves_primary_exception(monkeypatch, failure):
    module = importlib.import_module("resdag.hpo.trial_runner")
    runner = _runner(catch_exceptions=False)
    study = optuna.create_study()
    prepared = runner.prepare(study.ask())

    def fail(**kwargs):
        raise failure

    def cleanup():
        raise OSError("cleanup failed")

    runner.model_creator = fail
    monkeypatch.setattr(module, "_cleanup", cleanup)
    with pytest.raises(type(failure)) as caught:
        runner.evaluate(prepared)
    assert caught.value is failure
    assert any("cleanup failed" in note for note in failure.__notes__)


@pytest.mark.parametrize(
    "failure,state",
    [(ArithmeticError("numeric"), "fail"), (optuna.TrialPruned("pruned"), "pruned")],
)
def test_numerical_cleanup_preserves_recorded_failure_or_pruning(monkeypatch, failure, state):
    module = importlib.import_module("resdag.hpo.trial_runner")
    runner = _runner()
    study = optuna.create_study()
    prepared = runner.prepare(study.ask())

    def fail(**kwargs):
        raise failure

    def cleanup():
        raise OSError("cleanup failed")

    runner.model_creator = fail
    monkeypatch.setattr(module, "_cleanup", cleanup)
    outcome = runner.evaluate(prepared)
    assert outcome.state == state
    assert outcome.error == str(failure)
    assert outcome.attrs["cleanup_error"] == "cleanup failed"


def test_successful_numeric_cleanup_failure_still_raises(monkeypatch):
    module = importlib.import_module("resdag.hpo.trial_runner")
    runner = _runner()
    study = optuna.create_study()
    prepared = runner.prepare(study.ask())

    def cleanup():
        raise OSError("cleanup failed")

    monkeypatch.setattr(module, "_cleanup", cleanup)
    with pytest.raises(OSError, match="cleanup failed"):
        runner.evaluate(prepared)


def test_direct_runner_performs_numerical_cleanup_once(monkeypatch):
    module = importlib.import_module("resdag.hpo.trial_runner")
    calls = []
    monkeypatch.setattr(module, "_cleanup", lambda: calls.append("cleanup"))
    study = optuna.create_study()
    study.optimize(_runner(), n_trials=1)
    assert len(calls) == 1
    assert study.best_value >= 0
