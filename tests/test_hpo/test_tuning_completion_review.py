"""Independent acceptance oracles for completed common tuning protocols."""

import copy
from types import SimpleNamespace

import optuna
import pytest
import torch

import resdag as rd
from resdag.hpo import Screening


def _affine(*, lead=1, dt=None):
    x = rd.reservoir_input(1, dtype=torch.float64)
    y = rd.SVDReadoutLayer(1, 1, alpha=0).double()(x)
    return rd.Model(
        inputs={"state": x},
        outputs={"prediction": y},
        washout=1,
        prediction_lead=lead,
        resample_dt=dt,
    )


def _empty_space(trial):
    return {}


class _ConstantHead(rd.ReadoutLayer):
    supports_framewise = True

    def __init__(self, value):
        super().__init__(1, 1)
        self.value = value

    def solve(self, states, targets):
        return states.new_zeros(1, 1), states.new_full((1,), self.value)


def _constant(value):
    x = rd.reservoir_input(1, dtype=torch.float64)
    y = _ConstantHead(value).double()(x)
    return rd.Model(inputs={"state": x}, outputs={"prediction": y}, washout=0, preprocessing="off")


def test_screening_uses_full_horizon_and_final_ranking_reevaluates_all_seed_origins():
    data = torch.zeros(100, 1, dtype=torch.float64)
    study = optuna.create_study(direction="maximize")
    for value in [1, 2, 3]:
        study.enqueue_trial({"value": value})
    evaluations = []
    builds = []

    def factory(params, seed):
        builds.append((params["value"], seed))
        return _constant(params["value"])

    def space(trial):
        return {"value": trial.suggest_int("value", 1, 3)}

    def objective(context):
        value = float(context.predictions["prediction"][0, 0])
        evaluations.append(
            (value, context.seed, context.origin, context.predictions["prediction"].shape[0])
        )
        assert context.truth["prediction"].shape == (7, 1)
        assert context.scales["prediction"].item() == 1
        # Value3 wins the cheap first case, but value2 wins the full protocol.
        return value if (context.seed, context.origin) == (11, 0) else 10 - value

    model = _constant(0)
    model.tune(
        data,
        factory=factory,
        search_space=space,
        objective=objective,
        study=study,
        n_trials=3,
        seeds=[11, 12, 13],
        n_origins=3,
        horizon=7,
        checkpoint_interval=3,
        screening=Screening(keep_fraction=2 / 3),
    )
    report = model.tuning_report
    assert report.best_params == {"value": 2}
    assert report.score == 8
    assert len(report.runs) == 9 and all(run["complete"] for run in report.runs)
    assert report.resources["screened"] == 3 and report.resources["promoted"] == 2
    assert len(evaluations) == 3 + 2 * 9
    assert all(length == 7 for _, _, _, length in evaluations)
    assert [trial.state for trial in study.trials] == [
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.COMPLETE,
    ]
    assert study.trials[0].user_attrs["resdag_tune_valid"] is False
    assert study.trials[1].value == 8 and study.trials[2].value == 7
    assert len(builds) == 3 + 2 * 3 + 1
    torch.testing.assert_close(model.forecast(horizon=2), torch.full((2, 1), 2.0).double())


def _constant_factory(params, seed):
    return _constant(params["value"])


def _constant_space(trial):
    return {"value": trial.suggest_int("value", 1, 3)}


def _constant_objective(context):
    return float(context.predictions["prediction"][0, 0])


class _RecordingPruner(optuna.pruners.BasePruner):
    def __init__(self, prune_first=False):
        self.seen = []
        self.prune_first = prune_first

    def prune(self, study, trial):
        step = max(trial.intermediate_values)
        self.seen.append((trial.number, step, len(trial.intermediate_values)))
        return self.prune_first and trial.number == 0 and step == 7


def test_streamed_steps_are_globally_unique_over_several_seed_origin_runs():
    data = torch.arange(100, dtype=torch.float64)[:, None]
    pruner = _RecordingPruner()

    def factory(params, seed):
        return _affine()

    model = _affine()
    model.tune(
        data,
        factory=factory,
        search_space=_empty_space,
        n_trials=1,
        seeds=[2, 3],
        n_origins=3,
        horizon=7,
        checkpoint_interval=3,
        pruner=pruner,
    )
    expected = [case * 7 + prefix for case in range(6) for prefix in [3, 6, 7]]
    assert [step for _, step, _ in pruner.seen] == expected
    assert [count for _, _, count in pruner.seen] == list(range(1, 19))
    assert len(model.tuning_report.runs) == 6
    assert model.tuning_report.score == 7


def test_pruning_at_the_last_sample_still_disqualifies_that_candidate():
    data = torch.arange(100, dtype=torch.float64)[:, None]
    pruner = _RecordingPruner(prune_first=True)
    model = _affine()
    model.tune(
        data,
        search_space=_empty_space,
        n_trials=2,
        n_origins=1,
        horizon=7,
        checkpoint_interval=3,
        pruner=pruner,
    )
    report = model.tuning_report
    assert report.best_trial == 1 and report.score == 7
    assert report.study.trials[0].state == optuna.trial.TrialState.PRUNED
    assert report.study.trials[0].user_attrs["resdag_tune_valid"] is False
    assert report.study.trials[0].user_attrs["runs"][0]["complete"] is False
    assert report.study.trials[1].user_attrs["evaluation_complete"] is True


def test_common_threshold_preset_does_not_prune_an_exact_positive_horizon():
    model = _affine()
    model.tune(
        torch.arange(100, dtype=torch.float64)[:, None],
        search_space=_empty_space,
        n_trials=1,
        n_origins=1,
        horizon=7,
        checkpoint_interval=3,
        pruner="threshold",
    )
    assert model.tuning_report.score == 7


def test_common_threshold_preset_requires_an_explicit_bound_for_custom_scores():
    with pytest.raises(ValueError, match="configured|explicit"):
        _affine().tune(
            torch.arange(100, dtype=torch.float64)[:, None],
            search_space=_empty_space,
            objective=lambda context: 0.5,
            n_trials=1,
            n_origins=1,
            horizon=7,
            pruner="threshold",
        )


def test_warm_configuration_transfer_retests_on_the_new_task_without_old_score_copy():
    source_study = optuna.create_study(direction="maximize")
    for value in [1, 3]:
        source_study.enqueue_trial({"value": value})
    options = dict(
        factory=_constant_factory,
        search_space=_constant_space,
        seeds=[7],
        n_origins=1,
        n_trials=2,
        horizon=3,
    )
    _constant(0).tune(
        torch.zeros(40, 1).double(),
        study=source_study,
        objective=_constant_objective,
        **options,
    )
    seen = []

    def changed_objective(context):
        value = float(context.predictions["prediction"][0, 0])
        seen.append(value)
        assert bool((context.truth["prediction"] == 1).all())
        return -value

    target = _constant(0)
    target.tune(
        torch.ones(40, 1).double(),
        objective=changed_objective,
        warm_start=source_study,
        **options,
    )
    assert seen == [3, 1]
    assert target.tuning_report.best_params == {"value": 1}
    assert target.tuning_report.score == -1
    assert [trial.value for trial in target.tuning_report.study.trials] == [-3, -1]
    assert [trial.value for trial in source_study.trials] == [1, 3]
    assert all(
        trial.user_attrs["resdag_transfer"]["mode"] == "reevaluate"
        for trial in target.tuning_report.study.trials
    )


def test_identical_protocol_can_copy_full_scores_and_still_refits_the_correct_winner():
    data = torch.zeros(40, 1).double()
    source_study = optuna.create_study(direction="maximize")
    for value in [1, 3]:
        source_study.enqueue_trial({"value": value})
    options = dict(
        factory=_constant_factory,
        search_space=_constant_space,
        objective=_constant_objective,
        seeds=[7],
        n_origins=1,
        horizon=3,
    )
    _constant(0).tune(data, study=source_study, n_trials=2, **options)
    target_study = optuna.create_study(direction="maximize")
    target_study.enqueue_trial({"value": 1})
    target = _constant(0)
    target.tune(
        data,
        study=target_study,
        n_trials=1,
        warm_start=source_study,
        transfer_mode="scores",
        **options,
    )
    assert target.tuning_report.attempts == 1
    assert target.tuning_report.best_params == {"value": 3}
    assert target.tuning_report.score == 3
    assert len(target_study.trials) == 3
    assert len(source_study.trials) == 2
    torch.testing.assert_close(target.forecast(horizon=2), torch.full((2, 1), 3.0).double())


@pytest.mark.parametrize("factory", [None, _constant_factory])
def test_changed_custom_readout_implementation_cannot_resume_old_scores(monkeypatch, factory):
    data = torch.zeros(40, 1).double()
    study = optuna.create_study(direction="maximize")
    if factory is not None:
        study.enqueue_trial({"value": 1})
    options = dict(
        factory=factory,
        search_space=_empty_space if factory is None else _constant_space,
        objective=_constant_objective,
        seeds=[7],
        n_origins=1,
        horizon=3,
        n_trials=1,
        study=study,
    )
    model = _constant(1)
    model.tune(data, **options)

    def changed_solve(self, states, targets):
        return states.new_zeros(1, 1), states.new_full((1,), -self.value)

    monkeypatch.setattr(_ConstantHead, "solve", changed_solve)
    with pytest.raises(ValueError, match="protocol|changed"):
        model.tune(data, **options)


def test_screening_deadline_does_not_promote_incomplete_work_or_modify_fitted_model(monkeypatch):
    from resdag.hpo import tuning

    data = torch.arange(100, dtype=torch.float64)[:, None]
    model = _affine().fit(data)
    before = copy.deepcopy(model.state_dict())
    study = optuna.create_study(direction="maximize")
    clock = SimpleNamespace(now=0)
    monkeypatch.setattr(tuning, "time", SimpleNamespace(perf_counter=lambda: clock.now))

    def objective(context):
        clock.now = 10
        return 7.0

    with pytest.raises(RuntimeError, match="No valid tuning candidates"):
        model.tune(
            data,
            search_space=_empty_space,
            objective=objective,
            n_trials=3,
            seeds=[0],
            n_origins=1,
            horizon=7,
            study=study,
            screening=True,
            timeout=1,
        )
    assert len(study.trials) == 3
    assert all(trial.state == optuna.trial.TrialState.PRUNED for trial in study.trials)
    assert all("timeout" in trial.user_attrs["pruning_reason"] for trial in study.trials)
    assert model.state_dict().keys() == before.keys()
    for name, value in before.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)


def test_sparse_trajectory_preroll_streams_full_original_unit_objective():
    import numpy as np

    trajectory = pytest.importorskip("tsdynamics").Trajectory
    times = np.arange(0.0, 100.0, 3.0)
    data = trajectory(times, times[:, None], variables=("state",))
    calls = []
    pruner = _RecordingPruner()

    def objective(context):
        predicted, truth = context.predictions["prediction"], context.truth["prediction"]
        assert predicted.shape == truth.shape == (7, 1)
        torch.testing.assert_close(predicted, truth, rtol=1e-11, atol=1e-11)
        assert float(truth[0, 0]) >= 79
        calls.append(context.origin)
        return 7.0

    def prefix_objective(context):
        assert context.completed_steps in (3, 6, 7)
        assert context.horizon == 7
        assert context.truth["prediction"].shape == (context.completed_steps, 1)
        torch.testing.assert_close(
            context.predictions["prediction"], context.truth["prediction"], rtol=1e-11, atol=1e-11
        )
        return float(context.completed_steps)

    model = _affine(lead=2, dt=1.0)
    model.tune(
        data,
        search_space=_empty_space,
        objective=objective,
        pruning_objective=prefix_objective,
        pruner=pruner,
        n_trials=1,
        n_origins=3,
        horizon=7,
        checkpoint_interval=3,
    )
    assert calls == [0, 1, 2]
    assert any(record["skip"] > 0 for record in model.tuning_report.validation["origins"])
    assert model.tuning_report.score == 7
    assert [step for _, step, _ in pruner.seen] == [3, 6, 7, 10, 13, 14, 17, 20, 21]


@pytest.mark.parametrize("batch", [None, 3])
def test_memory_plan_reserves_known_long_validation_warmup_clone(batch):
    from resdag.hpo._tuning_data import prepare_tuning_data
    from resdag.hpo.resources import estimate_memory, memory_context

    model = _affine()
    train = torch.arange(20, dtype=torch.float64)[:, None]
    validation = torch.arange(20, 4020, dtype=torch.float64)[:, None]
    if batch is not None:
        train = train.unsqueeze(0).repeat(batch, 1, 1)
        validation = validation.unsqueeze(0).repeat(batch, 1, 1)
    prepared = prepare_tuning_data(
        model, train, validation=(validation, None), horizon=1, n_origins=2
    )
    context = memory_context(model, prepared)
    estimate = estimate_memory({}, context)
    # Each Origin.stream clones every observed input before warming the model.
    # This is a known live allocation beyond the resident protocol storage,
    # even with a one-sample forecast and an entirely framewise graph.
    required_clone = max(
        sum(value.numel() * value.element_size() for value in origin.history.values())
        for origin in prepared.origins
    )
    assert required_clone > 100 * train.numel() * train.element_size()
    assert (
        estimate.host_bytes
        >= context.dataset_host_bytes + context.model_host_bytes + required_clone
    )


def test_history_memory_context_covers_feature_traces_for_the_grown_replay_prefix():
    from resdag.hpo import MemoryBudget, MemoryEstimate

    train = torch.arange(20, dtype=torch.float64)[:, None]
    validation = torch.arange(20, 4020, dtype=torch.float64)[:, None]
    seen = []

    def estimate(params, context):
        seen.append(context)
        # Reject before any expensive replay. The assertion below inspects
        # facts supplied to the estimator, independent of its chosen estimate.
        return MemoryEstimate(10_000_000)

    with pytest.raises(RuntimeError, match="No valid tuning candidates"):
        _affine().tune(
            train,
            validation=(validation, None),
            search_space=_empty_space,
            n_trials=1,
            n_origins=1,
            horizon=3999,
            execution="history",
            memory_budget=MemoryBudget("1MB", worker_overhead=0),
            memory_estimator=estimate,
        )
    assert len(seen) == 1
    assert seen[0].forecast_samples >= 20 + 3999
    assert seen[0].refit_samples == 20
