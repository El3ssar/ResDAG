"""Public study, attempt-budget and failure-ownership tuning contracts."""

import copy
import importlib
import random

import numpy as np
import pytest
import torch

optuna = pytest.importorskip("optuna")

import resdag as rd  # noqa: E402


def _affine():
    x = rd.reservoir_input(1, dtype=torch.float64)
    y = rd.SVDReadoutLayer(1, 1, alpha=0).double()(x)
    return rd.Model(inputs={"state": x}, outputs={"prediction": y}, washout=2)


def _data():
    return torch.arange(40, dtype=torch.float64)[:, None]


def _same(left, right):
    assert left.keys() == right.keys()
    for name in left:
        if isinstance(left[name], torch.Tensor):
            torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)
        else:
            assert left[name] == right[name]


@pytest.mark.parametrize(
    "options, error",
    [
        ({"preset": "unknown"}, ValueError),
        ({"n_trials": 0}, ValueError),
        ({"n_trials": True}, ValueError),
        ({"horizon": 0}, ValueError),
        ({"n_origins": 0}, ValueError),
        ({"n_jobs": 0}, ValueError),
        ({"n_jobs": True}, ValueError),
        ({"inner_max_num_threads": 0}, ValueError),
        ({"threshold": float("nan")}, ValueError),
        ({"threshold": -1}, ValueError),
        ({"timeout": 0}, ValueError),
        ({"timeout": float("inf")}, ValueError),
        ({"aggregation": "unknown"}, ValueError),
        ({"execution": "unknown"}, ValueError),
        ({"seeds": []}, ValueError),
        ({"seeds": 3}, ValueError),
        ({"seeds": [-1]}, ValueError),
        ({"seeds": [True]}, ValueError),
        ({"seeds": [1, 2]}, ValueError),
        ({"seed": True}, ValueError),
        ({"seed": -1}, ValueError),
        ({"direction": "minimize"}, ValueError),
        ({"search_space": 3}, TypeError),
        ({"objective": 3}, TypeError),
        ({"factory": 3}, TypeError),
        ({"catch_exceptions": 3}, TypeError),
    ],
)
def test_bad_options_leave_the_fitted_model_unchanged(options, error):
    model = _affine().fit(_data()[:20])
    before = copy.deepcopy(model.state_dict())
    arguments = {"preset": "quick", "n_trials": 1, "horizon": 2, **options}
    with pytest.raises(error):
        model.tune(_data(), **arguments)
    _same(model.state_dict(), before)
    assert "tuning_report" not in model.__dict__


def test_resolved_fixed_values_are_saved_and_search_is_called_once_per_attempt():
    calls = []

    def space(trial):
        calls.append(trial.number)
        return {"alpha": 0.0, "washout": 3}

    model = _affine().tune(_data(), preset="quick", n_trials=3, horizon=3, search_space=space)
    assert calls == [0, 1, 2]
    report = model.tuning_report
    assert report.best_params == {"alpha": 0.0, "washout": 3}
    assert report.attempts == 3
    assert model.washout == 3
    for trial in report.study.trials:
        assert trial.params == {}  # Fixed values are still part of reconstruction.
        assert trial.user_attrs["resolved_params"] == report.best_params
    torch.testing.assert_close(model.forecast(horizon=2), torch.tensor([[40.0], [41.0]]).double())


def test_all_failed_candidates_consume_the_budget_without_committing():
    model = _affine().fit(_data()[:20])
    before = copy.deepcopy(model.state_dict())
    study = optuna.create_study(direction="maximize")
    with pytest.raises(RuntimeError, match="No valid tuning candidates.*alpha"):
        model.tune(
            _data(),
            preset="quick",
            n_trials=3,
            horizon=2,
            study=study,
            search_space=lambda trial: {"alpha": -1.0},
        )
    assert len(study.trials) == 3
    assert all(trial.state == optuna.trial.TrialState.FAIL for trial in study.trials)
    _same(model.state_dict(), before)


def test_search_interrupt_marks_attempt_failed_and_preserves_all_caller_rng():
    model = _affine()
    study = optuna.create_study(direction="maximize")
    interruption = KeyboardInterrupt()

    def space(trial):
        random.random()
        np.random.random()
        torch.rand(1)
        raise interruption

    python_before, numpy_before, torch_before = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    with pytest.raises(KeyboardInterrupt) as caught:
        model.tune(_data(), preset="quick", n_trials=3, horizon=2, study=study, search_space=space)
    assert caught.value is interruption
    assert len(study.trials) == 1 and study.trials[0].state == optuna.trial.TrialState.FAIL
    assert random.getstate() == python_before
    np.testing.assert_equal(np.random.get_state(), numpy_before)
    torch.testing.assert_close(torch.get_rng_state(), torch_before, rtol=0, atol=0)
    study.set_user_attr("caller_can_continue", True)


def test_owned_storage_cleanup_finishes_before_model_commit(tmp_path, monkeypatch):
    tuning = importlib.import_module("resdag.hpo.tuning")
    model = _affine().fit(_data()[:20])
    before = copy.deepcopy(model.state_dict())
    storage = optuna.storages.RDBStorage(f"sqlite:///{tmp_path / 'study.db'}")
    monkeypatch.setattr(tuning, "resolve_storage", lambda *args, **kwargs: storage)
    original_remove = storage.remove_session

    def fail_cleanup():
        original_remove()
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(storage, "remove_session", fail_cleanup)
    try:
        with pytest.raises(RuntimeError, match="cleanup failed"):
            model.tune(
                _data(),
                preset="quick",
                n_trials=1,
                horizon=2,
                search_space=lambda trial: {"alpha": 0.0},
            )
        _same(model.state_dict(), before)
        assert "tuning_report" not in model.__dict__
    finally:
        original_remove()
        storage.engine.dispose()


def test_changed_model_cannot_reuse_its_old_study():
    model = _affine().tune(
        _data(), preset="quick", n_trials=1, horizon=2, search_space=lambda trial: {"alpha": 0.0}
    )
    study = model.tuning_report.study
    model.washout = 5
    with pytest.raises(ValueError, match="model changed"):
        model.tune(_data(), preset="quick", n_trials=1, horizon=2, study=study)
    assert len(study.trials) == 1


def test_different_data_does_not_append_trials_to_an_existing_protocol():
    model = _affine()

    def space(trial):
        return {"alpha": 0.0}

    model.tune(_data(), preset="quick", n_trials=1, horizon=2, search_space=space)
    study = model.tuning_report.study
    with pytest.raises(ValueError, match="protocol differs"):
        model.tune(
            _data() + 100, preset="quick", n_trials=1, horizon=2, search_space=space, study=study
        )
    assert len(study.trials) == 1


def test_existing_foreign_score_cannot_be_used_as_a_tuning_score():
    study = optuna.create_study(direction="maximize")
    study.add_trial(optuna.trial.create_trial(value=10000.0))
    model = _affine()
    with pytest.raises(ValueError, match="protocol differs"):
        model.tune(_data(), preset="quick", n_trials=1, horizon=2, study=study)
    assert len(study.trials) == 1


def test_fixed_tuple_configuration_is_preserved_for_candidates_and_refit():
    def build(params, seed):
        assert params["options"] == ("cycle", {"degree": 2})
        assert isinstance(params["options"], tuple)
        return _affine()

    model = _affine().tune(
        _data(),
        preset="quick",
        n_trials=1,
        horizon=2,
        factory=build,
        search_space=lambda trial: {"options": ("cycle", {"degree": 2})},
    )
    assert model.tuning_report.best_params["options"] == ("cycle", {"degree": 2})


def test_tuned_alpha_overrides_preset_readout_kwargs_and_report_matches():
    model = rd.ESN(4, washout=2, readout_kwargs={"alpha": 1000.0})
    model.tune(
        _data().float(),
        preset="quick",
        n_trials=1,
        horizon=2,
        search_space=lambda trial: {"alpha": 0.0},
    )
    head = next(
        module for module in model.graph.modules() if isinstance(module, rd.AutoReadoutLayer)
    )
    assert head.alpha == 0.0
    assert head.last_solve is not None and head.last_solve.alpha == 0.0
    assert model.tuning_report.best_params == {"alpha": 0.0}


def test_recipe_returning_a_caller_model_is_staged_instead_of_mutating_it():
    shared = _affine().fit(_data()[:15])
    original = copy.deepcopy(shared.state_dict())
    target = _affine()
    target.tune(
        _data(),
        preset="quick",
        n_trials=1,
        horizon=2,
        factory=lambda params, seed: shared,
        search_space=lambda trial: {},
    )
    _same(shared.state_dict(), original)
    torch.testing.assert_close(target.forecast(horizon=2), torch.tensor([[40.0], [41.0]]).double())
