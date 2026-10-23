"""Raw factories share estimation policy and skip rather than fabricate winners."""

import importlib
from contextlib import contextmanager

import optuna
import pytest
import torch

from resdag.hpo import run_hpo
from resdag.hpo.resources import MemoryBudget, MemoryEstimate
from resdag.hpo.trial_runner import TrialResult


def _factory(size=8):
    raise AssertionError("Oversized or mocked numerical candidates must never construct a model")


def _data(trial):
    values = torch.arange(30, dtype=torch.float32).reshape(1, -1, 1)
    return dict(
        warmup=values[:, :3],
        train=values[:, 3:16],
        target=values[:, 4:17],
        f_warmup=values[:, 16:20],
        val=values[:, 20:25],
    )


def _space(trial):
    return {"size": 8}


def test_raw_budget_requires_custom_estimator_before_any_study_or_data():
    with pytest.raises(ValueError, match="require memory_estimator"):
        run_hpo(_factory, _space, _data, 1, memory_budget="1MiB")
    with pytest.raises(ValueError, match="requires memory_budget"):
        run_hpo(_factory, _space, _data, 1, memory_estimator=lambda params, context: 10)


def test_oversized_candidates_are_pruned_before_construction_and_have_no_best():
    def estimate(params, context):
        assert context.max_horizon == 5
        assert context.retained_forecast_bytes == 20
        assert context.readout_features == ()
        with pytest.raises(TypeError):
            context.input_shapes["train"] = (1, 1, 1)
        return MemoryEstimate(10000, details={"custom": "peak"})

    study = run_hpo(
        _factory,
        _space,
        _data,
        n_trials=3,
        n_workers=2,
        memory_budget=MemoryBudget(9999, worker_overhead=0),
        memory_estimator=estimate,
        verbosity=0,
    )
    assert len(study.trials) == 3
    assert all(trial.state == optuna.trial.TrialState.PRUNED for trial in study.trials)
    assert all("memory_budget" in trial.user_attrs["pruned_reason"] for trial in study.trials)
    assert all(trial.user_attrs["memory_estimate"]["host_bytes"] == 10000 for trial in study.trials)
    with pytest.raises(ValueError):
        _ = study.best_trial


@pytest.mark.parametrize("limit,expected", [(2200, [2, 2, 1]), (1500, [1, 1, 1, 1, 1])])
def test_admission_reduces_numeric_window_without_losing_attempts(monkeypatch, limit, expected):
    dispatch = importlib.import_module("resdag.hpo.dispatch")
    windows = []

    @contextmanager
    def evaluate(function, jobs, **kwargs):
        windows.append(len(jobs))
        yield iter(
            TrialResult(item.number, item.params, "complete", 1.0, item.attrs) for item, _ in jobs
        )

    monkeypatch.setattr(dispatch, "iter_jobs", evaluate)
    study = run_hpo(
        _factory,
        _space,
        _data,
        n_trials=5,
        n_workers=2,
        memory_budget=MemoryBudget(limit, worker_overhead=100),
        memory_estimator=lambda params, context: MemoryEstimate(1000),
        verbosity=0,
    )
    assert windows == expected
    assert len(study.trials) == 5
    assert all(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
    assert all(trial.user_attrs["memory_estimate"]["host_bytes"] == 1000 for trial in study.trials)


def test_estimator_failure_is_configuration_error_and_releases_reserved_trial(monkeypatch):
    run = importlib.import_module("resdag.hpo.run")
    study = optuna.create_study()
    monkeypatch.setattr(run.optuna, "create_study", lambda **kwargs: study)
    primary = RuntimeError("estimator cannot estimate custom workspace")

    def bad_estimate(params, context):
        raise primary

    with pytest.raises(RuntimeError) as caught:
        run_hpo(
            _factory,
            _space,
            _data,
            n_trials=2,
            memory_budget=MemoryBudget(1000, worker_overhead=0),
            memory_estimator=bad_estimate,
            verbosity=0,
        )
    assert caught.value is primary
    assert len(study.trials) == 1
    assert study.trials[0].state == optuna.trial.TrialState.FAIL
