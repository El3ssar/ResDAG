"""Complete settings reuse, exact-protocol score import and provenance."""

import copy

import optuna
import pytest
import torch
from optuna.distributions import FloatDistribution, IntDistribution
from optuna.trial import TrialState

from resdag.hpo import export_best_config, run_hpo
from resdag.hpo._provenance import (
    PROTOCOL_KEY,
    RESOLVED_SETTINGS_KEY,
    REUSED_SETTINGS_KEY,
    bind_protocol,
    resolve_trial_settings,
    trial_settings,
)
from resdag.hpo.transfer import apply_warm_start, transfer_trials
from resdag.models import classic_esn


def _prior(*, value=1.0, attrs=None):
    source = optuna.create_study(study_name="source")
    source.add_trial(
        optuna.trial.create_trial(
            value=value,
            params={"size": 8, "data_scale": 2.0},
            distributions={"size": IntDistribution(5, 10), "data_scale": FloatDistribution(1, 3)},
            user_attrs={RESOLVED_SETTINGS_KEY: repr({"size": 8, "alpha": 0.04}), **(attrs or {})},
        )
    )
    return source


def _data(trial):
    scale = trial.suggest_float("data_scale", 1, 3)
    values = scale * torch.sin(torch.arange(55, dtype=torch.float32).reshape(1, -1, 1) / 7)
    return dict(
        warmup=values[:, :5],
        train=values[:, 5:30],
        target=values[:, 6:31],
        f_warmup=values[:, 30:35],
        val=values[:, 35:45],
    )


def _space(trial):
    return {"size": trial.suggest_int("size", 5, 10)}


def _factory(size=8, alpha=0.1):
    return classic_esn(reservoir_size=size, feedback_size=1, output_size=1, readout_alpha=alpha)


def test_default_transfer_queues_full_settings_without_copying_score():
    source = _prior()
    source.set_user_attr(PROTOCOL_KEY, "source-data")
    destination = optuna.create_study(direction="maximize")
    destination.set_user_attr(PROTOCOL_KEY, "different-data")
    before = copy.deepcopy(source.trials)
    assert transfer_trials(destination, source) == 1
    queued = destination.trials[0]
    assert queued.state == TrialState.WAITING
    assert queued.value is None
    assert queued.user_attrs["resdag_transfer"] == {
        "mode": "reevaluate",
        "study_name": "source",
        "trial_number": 0,
        "protocol_hash": "source-data",
        "resolved_settings_literal": repr({"size": 8, "alpha": 0.04}),
    }
    trial = destination.ask()
    settings = resolve_trial_settings(trial, _space(trial))
    assert settings == {"size": 8, "alpha": 0.04}
    assert _data(trial)["train"].shape == (1, 25, 1)
    assert trial.params["data_scale"] == 2.0
    assert source.trials == before


@pytest.mark.parametrize("case", ["unbound", "different", "direction"])
def test_score_transfer_requires_exact_bound_task_and_direction_before_mutation(case):
    source, destination = _prior(), optuna.create_study(
        direction="maximize" if case == "direction" else "minimize"
    )
    if case != "unbound":
        source.set_user_attr(PROTOCOL_KEY, "source")
        destination.set_user_attr(PROTOCOL_KEY, "different" if case == "different" else "source")
    with pytest.raises(ValueError):
        transfer_trials(destination, source, mode="scores", protocol_hash="source")
    assert destination.trials == []


def test_score_transfer_preserves_every_parameter_distribution_and_full_settings():
    source, destination = _prior(), optuna.create_study()
    for study in (source, destination):
        study.set_user_attr(PROTOCOL_KEY, "same")
    assert transfer_trials(destination, source, mode="scores", protocol_hash="same") == 1
    copied = destination.trials[0]
    assert copied.params == source.trials[0].params
    assert copied.distributions == source.trials[0].distributions
    assert copied.value == source.best_value
    assert trial_settings(copied) == {"size": 8, "alpha": 0.04}
    assert copied.user_attrs["resdag_transfer"]["mode"] == "scores"


@pytest.mark.parametrize(
    "attrs,value",
    [({"error": "invalid"}, 1.0), ({"nonfinite_loss": True}, 1.0), ({}, float("inf"))],
)
def test_invalid_source_evaluations_cannot_seed_or_export(attrs, value):
    source, destination = _prior(value=value, attrs=attrs), optuna.create_study()
    assert transfer_trials(destination, source) == 0
    with pytest.raises(ValueError, match="invalid|finite"):
        export_best_config(source)


def test_malformed_recorded_settings_fail_before_enqueuing():
    source = _prior(attrs={RESOLVED_SETTINGS_KEY: "__import__('os')"})
    destination = optuna.create_study()
    with pytest.raises(ValueError, match="settings"):
        transfer_trials(destination, source)
    assert destination.trials == []


def test_warm_start_preserves_fixed_unsuggested_settings_and_export():
    study = optuna.create_study()
    assert apply_warm_start(study, [{}, {"size": 8, "alpha": 0.04}]) == 1
    trial = study.ask()
    assert resolve_trial_settings(trial, _space(trial)) == {"size": 8, "alpha": 0.04}
    assert REUSED_SETTINGS_KEY in trial.user_attrs
    source = _prior()
    assert export_best_config(source) == {"size": 8, "data_scale": 2.0, "alpha": 0.04}


@pytest.mark.parametrize("workers", [1, 2])
def test_raw_transfer_reevaluates_full_settings_with_data_only_suggestions(workers):
    prior = _prior(value=-99999)
    study = run_hpo(
        _factory,
        _space,
        _data,
        n_trials=2,
        transfer_from=prior,
        n_workers=workers,
        seed=4,
        loss=lambda predicted, truth: float(((predicted - truth) ** 2).mean()),
        verbosity=0,
    )
    assert len(study.trials) == 2
    assert all(trial.state == TrialState.COMPLETE for trial in study.trials)
    first = study.trials[0]
    assert first.params == {"size": 8, "data_scale": 2.0}
    assert trial_settings(first) == {"size": 8, "alpha": 0.04}
    assert first.value != -99999
    assert first.user_attrs["resdag_raw_data_params"] == {"data_scale": 2.0}
    assert first.user_attrs["resdag_transfer"]["trial_number"] == 0


def test_protocol_binding_rejects_old_evaluated_unbound_study_and_changed_hash():
    source = _prior()
    with pytest.raises(ValueError):
        bind_protocol(source, "new")
    source.set_user_attr(PROTOCOL_KEY, "old")
    with pytest.raises(ValueError):
        bind_protocol(source, "new")
    assert source.user_attrs[PROTOCOL_KEY] == "old"


@pytest.mark.parametrize(
    "direction,expected", [("minimize", [2, 0, 1, 3]), ("maximize", [3, 0, 1, 2])]
)
def test_reused_settings_best_first_with_stable_source_ties(direction, expected):
    source = optuna.create_study(direction=direction)
    for number, value in enumerate([2.0, 2.0, -1.0, 5.0]):
        source.add_trial(
            optuna.trial.create_trial(
                value=value,
                params={"x": number},
                distributions={"x": IntDistribution(0, 3)},
            )
        )
    destination = optuna.create_study()
    assert transfer_trials(destination, source) == 4
    assert [
        trial.user_attrs["resdag_transfer"]["trial_number"] for trial in destination.trials
    ] == expected
    assert all(trial.value is None for trial in destination.trials)


def test_multiobjective_settings_keep_stable_source_order():
    source = optuna.create_study(directions=["minimize", "maximize"])
    for number, values in enumerate(([2.0, 8.0], [1.0, 3.0], [5.0, 10.0])):
        source.add_trial(
            optuna.trial.create_trial(
                values=values,
                params={"x": number},
                distributions={"x": IntDistribution(0, 2)},
            )
        )
    destination = optuna.create_study()
    assert transfer_trials(destination, source) == 3
    assert [
        trial.user_attrs["resdag_transfer"]["trial_number"] for trial in destination.trials
    ] == [0, 1, 2]


def test_later_malformed_settings_reject_entire_import_before_mutation():
    source = _prior()
    source.add_trial(
        optuna.trial.create_trial(
            value=2.0,
            params={"size": 9},
            distributions={"size": IntDistribution(5, 10)},
            user_attrs={RESOLVED_SETTINGS_KEY: "not a literal"},
        )
    )
    destination = optuna.create_study()
    with pytest.raises(ValueError, match="settings"):
        transfer_trials(destination, source)
    assert destination.trials == []
