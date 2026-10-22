"""Independent winner/ordering checks for externally created Optuna studies."""

import copy
import math
import re
import sys

import numpy as np
import pytest

optuna = pytest.importorskip("optuna")

from resdag.hpo.transfer import transfer_trials  # noqa: E402
from resdag.hpo.utils import get_study_summary  # noqa: E402


def _trial(values, *, parameter=0.25):
    return optuna.trial.create_trial(
        values=values,
        params={"x": parameter},
        distributions={"x": optuna.distributions.FloatDistribution(0.0, 1.0)},
        user_attrs={"external_metadata": {"labels": ["preserve"]}},
    )


@pytest.mark.parametrize("directions", [("minimize",), ("maximize",), ("minimize", "maximize")])
@pytest.mark.parametrize("bad_value", [float("inf"), float("-inf"), float("nan")])
def test_transfer_omits_external_nonfinite_scores_without_diagnostic_flags(
    directions, bad_value, monkeypatch
):
    source = optuna.create_study(directions=directions)
    destination = optuna.create_study(directions=directions)
    width = len(directions)
    valid_values = [3.0] * width
    invalid_values = [3.0] * (width - 1) + [bad_value]
    source.add_trial(_trial(valid_values))
    if math.isnan(bad_value):
        # Public Optuna writes reject NaN. Model a corrupt/external storage read
        # through its public Study getter, without reaching into storage internals.
        external = _trial([0.0] * width, parameter=0.75)
        external.values = invalid_values
        valid = source.trials
        monkeypatch.setattr(source, "get_trials", lambda *args, **kwargs: valid + [external])
    else:
        # Optuna permits +/- infinity as completed values, with no error flags.
        source.add_trial(_trial(invalid_values, parameter=0.75))
    destination.add_trial(_trial([0.0] * width, parameter=0.5))
    source.set_user_attr("caller_owned", {"keep": True})
    original = copy.deepcopy(source.trials)

    assert transfer_trials(destination, source) == 1
    assert len(destination.trials) == 2
    assert destination.trials[-1].values == valid_values
    assert destination.trials[-1].params == {"x": 0.25}
    assert destination.trials[-1].user_attrs == original[0].user_attrs
    assert source.user_attrs == {"caller_owned": {"keep": True}}
    for before, after in zip(original, source.trials, strict=True):
        np.testing.assert_equal(before.values, after.values)
        assert before.params == after.params
        assert before.user_attrs == after.user_attrs
    # Caller-owned studies remain usable after transfer.
    source.set_user_attr("still_open", True)
    destination.set_user_attr("still_open", True)


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
@pytest.mark.parametrize("value", [0.0, -3.0, sys.float_info.max, -sys.float_info.max])
def test_transfer_keeps_finite_scores_even_at_float_limits(direction, value):
    source = optuna.create_study(direction=direction)
    destination = optuna.create_study(direction=direction)
    source.add_trial(_trial([value]))

    assert transfer_trials(destination, source) == 1
    assert destination.best_value == value
    assert destination.best_params == {"x": 0.25}
    assert source.best_value == value


@pytest.mark.parametrize("bad_slot", [0, 1])
@pytest.mark.parametrize("bad_value", [float("inf"), float("-inf")])
def test_multiobjective_transfer_checks_every_component_before_mutating(bad_slot, bad_value):
    directions = ["minimize", "maximize"]
    source = optuna.create_study(directions=directions)
    destination = optuna.create_study(directions=directions)
    values = [2.0, 7.0]
    values[bad_slot] = bad_value
    source.add_trial(_trial(values))
    destination.add_trial(_trial([8.0, 1.0], parameter=0.5))
    before = copy.deepcopy(destination.trials)

    assert transfer_trials(destination, source) == 0
    assert destination.trials == before
    assert source.trials[0].values == values


def test_matching_multiobjective_transfer_preserves_values_and_directions():
    source = optuna.create_study(directions=["minimize", "maximize"])
    destination = optuna.create_study(directions=["minimize", "maximize"])
    source.add_trial(_trial([2.0, 7.0]))

    assert transfer_trials(destination, source) == 1
    assert destination.directions == source.directions
    assert destination.trials[0].values == [2.0, 7.0]
    assert destination.trials[0].user_attrs == source.trials[0].user_attrs


def test_direction_mismatch_is_rejected_before_reading_or_mutating_trials(monkeypatch):
    source = optuna.create_study(directions=["minimize", "maximize"])
    destination = optuna.create_study(directions=["maximize", "minimize"])
    destination.add_trial(_trial([2.0, 7.0]))
    before = copy.deepcopy(destination.trials)

    def forbidden_read(*args, **kwargs):
        raise AssertionError("Source trials read before validating objective directions")

    monkeypatch.setattr(source, "get_trials", forbidden_read)
    with pytest.raises(ValueError, match="different objective directions"):
        transfer_trials(destination, source)
    assert destination.trials == before


@pytest.mark.parametrize(
    ("direction", "expected"),
    [("minimize", [2, 0, 3, 1]), ("maximize", [1, 3, 0, 2])],
)
@pytest.mark.parametrize("top_n", [1, 3, 10])
def test_summary_top_trials_follow_study_direction(direction, expected, top_n):
    study = optuna.create_study(direction=direction)
    for value in [2.0, 5.0, -1.0, 4.0]:
        study.add_trial(optuna.trial.create_trial(value=value))
    study.add_trial(optuna.trial.create_trial(state=optuna.trial.TrialState.FAIL))
    study.add_trial(optuna.trial.create_trial(state=optuna.trial.TrialState.PRUNED))
    before = copy.deepcopy(study.trials)

    summary = get_study_summary(study, top_n=top_n)
    entries = re.findall(r"^\d+\. Trial (\d+):", summary, re.MULTILINE)
    assert [int(number) for number in entries] == expected[:top_n]
    assert int(entries[0]) == study.best_trial.number
    assert f"Top {min(top_n, 4)} Trials" in summary
    assert study.trials == before


def test_maximizing_summary_keeps_trial_order_for_tied_values():
    study = optuna.create_study(direction="maximize")
    for value in [2.0, 5.0, 5.0, 1.0]:
        study.add_trial(optuna.trial.create_trial(value=value))
    summary = get_study_summary(study, top_n=3)
    assert re.findall(r"^\d+\. Trial (\d+):", summary, re.MULTILINE) == ["1", "2", "0"]
