"""A failed search cannot masquerade as a trained best model."""

import optuna
import pytest
import torch

from resdag import classic_esn
from resdag.hpo import run_hpo
from resdag.hpo.transfer import transfer_trials


def _creator(reservoir_size=6, seed=None):
    return classic_esn(reservoir_size, 1, 1, seed=seed)


def _space(trial):
    return {"reservoir_size": trial.suggest_int("reservoir_size", 6, 8)}


def _data(trial):
    series = torch.sin(torch.arange(100, dtype=torch.float32) * 0.1).reshape(1, -1, 1)
    return {
        "warmup": series[:, :10],
        "train": series[:, 10:60],
        "target": series[:, 11:61],
        "f_warmup": series[:, 51:61],
        "val": series[:, 61:71],
    }


def _invalid_data(trial):
    return {}


def _sometimes_invalid(trial):
    return {} if trial.number == 0 else _data(trial)


def _nan_loss(truth, prediction):
    return float("nan")


@pytest.mark.parametrize("workers", [1, 2])
def test_all_failed_trials_raise_with_original_error(workers, tmp_path):
    storage = str(tmp_path / "invalid.log")
    with pytest.raises(RuntimeError, match="No valid HPO trials.*missing required keys"):
        run_hpo(
            _creator,
            _space,
            _invalid_data,
            n_trials=2,
            n_workers=workers,
            storage=storage,
            study_name="all_invalid",
            seed=3,
            verbosity=0,
        )
    from resdag.hpo.storage import resolve_storage

    study = optuna.load_study(study_name="all_invalid", storage=resolve_storage(storage, 1))
    assert study.trials
    assert all(t.state == optuna.trial.TrialState.FAIL for t in study.trials)
    assert all("missing required keys" in t.user_attrs["error"] for t in study.trials)


def test_failed_configuration_is_excluded_from_winner():
    study = run_hpo(_creator, _space, _sometimes_invalid, n_trials=3, seed=9, verbosity=0)
    assert study.trials[0].state == optuna.trial.TrialState.FAIL
    assert study.best_trial.number != 0
    assert study.best_trial.user_attrs["model_seed"] == 9 + study.best_trial.number


def test_nonfinite_objective_is_a_failed_evaluation(tmp_path):
    storage = str(tmp_path / "nan.log")
    with pytest.raises(RuntimeError, match="No valid HPO trials.*non-finite"):
        run_hpo(
            _creator,
            _space,
            _data,
            n_trials=1,
            loss=_nan_loss,
            storage=storage,
            study_name="nan",
            verbosity=0,
        )
    from resdag.hpo.storage import resolve_storage

    study = optuna.load_study(study_name="nan", storage=resolve_storage(storage, 1))
    assert study.trials[0].state == optuna.trial.TrialState.FAIL
    assert study.trials[0].user_attrs["nonfinite_loss"] is True


def test_explicit_legacy_penalty_cannot_return_all_invalid_winner():
    with pytest.raises(RuntimeError, match="No valid HPO trials"):
        run_hpo(
            _creator,
            _space,
            _invalid_data,
            n_trials=1,
            error_policy="penalty",
            verbosity=0,
        )


def test_transfer_rejects_different_directions_before_mutation():
    source = optuna.create_study(direction="maximize")
    source.add_trial(
        optuna.trial.create_trial(
            value=100.0,
            params={"x": 0.5},
            distributions={"x": optuna.distributions.FloatDistribution(0, 1)},
        )
    )
    destination = optuna.create_study(direction="minimize")
    for study in (source, destination):
        study.set_user_attr("resdag_tuning_protocol_v1", "identical")
    with pytest.raises(ValueError, match="matching objective directions"):
        transfer_trials(destination, source, mode="scores", protocol_hash="identical")
    assert destination.trials == []


def test_transfer_skips_legacy_failed_scores():
    source = optuna.create_study()
    source.add_trial(
        optuna.trial.create_trial(
            value=-100.0,
            params={"x": 0.5},
            distributions={"x": optuna.distributions.FloatDistribution(0, 1)},
            user_attrs={"error": "invalid old trial"},
        )
    )
    destination = optuna.create_study()
    assert transfer_trials(destination, source) == 0
    assert destination.trials == []


def test_explicit_model_seed_is_recorded_instead_of_trial_rng_seed():
    def space(trial):
        return {"reservoir_size": 6, "seed": 123}

    study = run_hpo(_creator, space, _data, n_trials=1, seed=9, verbosity=0)
    assert study.best_trial.user_attrs["model_seed"] == 123
    assert study.best_trial.user_attrs["trial_seed"] == 9


def test_hpo_forecast_uses_eval_and_restores_mode(monkeypatch):
    from contextlib import contextmanager

    import resdag.core._forecast_stream as streaming

    original = streaming.raw_forecast_stream
    observed = []

    @contextmanager
    def forecast(model, *args, **kwargs):
        def observe(chunks):
            for chunk in chunks:
                observed.append(model.training)
                assert not any(module.training for module in model.modules())
                yield chunk

        with original(model, *args, **kwargs) as chunks:
            yield observe(chunks)

    monkeypatch.setattr(streaming, "raw_forecast_stream", forecast)
    run_hpo(_creator, _space, _data, n_trials=1, seed=9, verbosity=0)
    assert observed and not any(observed)
