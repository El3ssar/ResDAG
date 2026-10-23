"""Common-workflow resource, streaming and failure acceptance oracles."""

import copy
import importlib
import os

import pytest
import torch

import resdag as rd

optuna = pytest.importorskip("optuna")

from resdag.hpo import MemoryBudget, MemoryEstimate, Screening  # noqa: E402


class _ConstantHead(rd.ReadoutLayer):
    supports_framewise = True
    steps = 0

    def __init__(self, value=0.0):
        super().__init__(1, 1)
        self.value = value

    def solve(self, states, targets):
        return states.new_zeros(1, 1), states.new_full((1,), self.value)

    def forward(self, inputs):
        if self.is_fitted and inputs.shape[-2] == 1:
            type(self).steps += 1
        return super().forward(inputs)


def _constant(value=0.0):
    x = rd.reservoir_input(1, dtype=torch.float64)
    y = _ConstantHead(value).double()(x)
    return rd.Model(inputs={"state": x}, outputs={"prediction": y}, preprocessing="off", washout=0)


def _empty(trial):
    return {}


class _PruneSecond(optuna.pruners.BasePruner):
    def __init__(self):
        self.parent = os.getpid()
        self.seen = []

    def prune(self, study, trial):
        assert os.getpid() == self.parent
        self.seen.append((trial.number, max(trial.intermediate_values)))
        return trial.number == 1


def test_pruning_stops_actual_graph_execution_at_first_checkpoint():
    _ConstantHead.steps = 0
    pruner = _PruneSecond()
    model = _constant()
    model.tune(
        torch.zeros(100, 1).double(),
        search_space=_empty,
        n_trials=2,
        n_origins=1,
        horizon=8,
        checkpoint_interval=3,
        pruner=pruner,
    )
    # Each origin's first prediction comes from history warmup; only further
    # steps call the graph with a one-frame input. Refitting has a long input.
    assert _ConstantHead.steps == (8 - 1) + (3 - 1)
    assert pruner.seen == [(0, 3), (0, 6), (0, 8), (1, 3)]
    first, second = model.tuning_report.study.trials
    assert first.state == optuna.trial.TrialState.COMPLETE
    assert second.state == optuna.trial.TrialState.PRUNED
    assert second.user_attrs["runs"][0]["completed_steps"] == 3


def test_real_parallel_pruning_uses_caller_owned_in_memory_study():
    pruner = _PruneSecond()
    study = optuna.create_study(direction="maximize", pruner=pruner)
    model = _constant()
    model.tune(
        torch.zeros(100, 1).double(),
        search_space=_empty,
        n_trials=2,
        n_jobs=2,
        n_origins=1,
        horizon=8,
        checkpoint_interval=3,
        study=study,
    )
    assert model.tuning_report.study is study
    assert sorted(pruner.seen) == [(0, 3), (0, 6), (0, 8), (1, 3)]
    assert [trial.state for trial in study.trials] == [
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
    ]
    assert model.tuning_report.best_trial == 0


def test_explicit_prefix_objective_is_isolated_and_full_objective_remains_complete():
    prefixes, full = [], []

    def prefix(context):
        prefixes.append((context.completed_steps, context.horizon))
        assert context.predictions["prediction"].shape == (context.completed_steps, 1)
        assert context.truth["prediction"].shape == (context.completed_steps, 1)
        context.truth["prediction"].fill_(100)
        context.scales["prediction"].fill_(100)
        return 1.0

    def objective(context):
        full.append(context.predictions["prediction"].shape)
        assert context.truth["prediction"].count_nonzero() == 0
        torch.testing.assert_close(context.scales["prediction"], torch.ones(1).double())
        return 1.0

    model = _constant()
    model.tune(
        torch.zeros(100, 1).double(),
        search_space=_empty,
        n_trials=1,
        n_origins=1,
        horizon=8,
        checkpoint_interval=3,
        objective=objective,
        pruning_objective=prefix,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )
    assert prefixes == [(3, 8), (6, 8), (8, 8)]
    assert full == [(8, 1)]


@pytest.mark.parametrize("jobs", [1, 2])
def test_memory_admission_skips_before_factory_and_reduces_actual_parallel_work(jobs):
    builds = []

    def factory(params, seed):
        assert params["estimated"] != 200_000
        builds.append(params["estimated"])
        return _constant()

    def space(trial):
        return {"estimated": trial.suggest_categorical("estimated", [20_000, 90_000, 200_000])}

    study = optuna.create_study(direction="maximize")
    for estimate in [20_000, 90_000, 200_000]:
        study.enqueue_trial({"estimated": estimate})
    model = _constant()
    model.tune(
        torch.zeros(100, 1).double(),
        factory=factory,
        search_space=space,
        n_trials=3,
        seeds=[0],
        n_origins=1,
        horizon=8,
        n_jobs=jobs,
        study=study,
        memory_budget=MemoryBudget(100_000, worker_overhead=0),
        memory_estimator=lambda params, context: MemoryEstimate(params["estimated"]),
    )
    assert builds == [20_000, 90_000, 20_000]
    assert model.tuning_report.resources["max_running_jobs"] == 1
    assert model.tuning_report.resources["memory_skipped"] == 1
    assert model.tuning_report.attempts == 3
    assert [trial.state for trial in study.trials] == [
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
    ]
    assert "memory_budget" in study.trials[2].user_attrs["pruning_reason"]


def test_automatic_esn_estimate_never_constructs_oversized_candidate(monkeypatch):
    tuning = importlib.import_module("resdag.hpo.tuning")
    original = rd.ESN
    built = []

    class CheckedESN(original):
        def __init__(self, *args, **kwargs):
            assert kwargs.get("reservoir_size") != 100_000
            built.append(kwargs.get("reservoir_size"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(tuning, "ESN", CheckedESN)
    model = CheckedESN(reservoir_size=2, washout=0)
    study = optuna.create_study(direction="maximize")
    for size in [100_000, 2]:
        study.enqueue_trial({"reservoir_size": size})
    model.tune(
        torch.zeros(100, 1),
        search_space=lambda trial: {
            "reservoir_size": trial.suggest_categorical("reservoir_size", [100_000, 2])
        },
        study=study,
        n_trials=2,
        n_origins=1,
        seeds=[0],
        horizon=8,
        memory_budget=MemoryBudget("8MiB", worker_overhead=0),
    )
    assert built == [2, 2, 2]
    assert study.trials[0].state == optuna.trial.TrialState.PRUNED
    assert model.tuning_report.best_params == {"reservoir_size": 2}


def test_lazy_base_memory_preflight_precedes_dense_construction(monkeypatch):
    model = rd.ESN(reservoir_size=100_000)
    monkeypatch.setattr(model, "build", lambda *args, **kwargs: pytest.fail("allocated base"))
    with pytest.raises(ValueError, match="before construction"):
        model.tune(torch.zeros(100, 1), memory_budget="1MiB")
    assert not model.is_built


def test_no_candidate_fits_reports_memory_reason_and_preserves_fitted_model():
    data = torch.zeros(100, 1).double()
    model = _constant().fit(data)
    before = copy.deepcopy(model.state_dict())
    study = optuna.create_study(direction="maximize")
    with pytest.raises(RuntimeError, match="memory_budget"):
        model.tune(
            data,
            search_space=_empty,
            study=study,
            n_trials=3,
            n_origins=1,
            horizon=8,
            memory_budget=MemoryBudget(1, worker_overhead=0),
        )
    assert len(study.trials) == 3
    assert all(trial.state == optuna.trial.TrialState.PRUNED for trial in study.trials)
    for name, value in before.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize(
    "options, error, message",
    [
        ({"screening": "yes"}, TypeError, "Screening"),
        ({"screening": Screening(seed_count=2)}, ValueError, "counts"),
        ({"checkpoint_interval": 0}, ValueError, "positive integer"),
        ({"pruning_objective": 3}, TypeError, "callable"),
        ({"memory_estimator": lambda p, c: 1}, ValueError, "requires memory_budget"),
        (
            {"factory": lambda p, s: _constant(), "memory_budget": "1MiB"},
            ValueError,
            "requires memory_estimator",
        ),
        ({"warm_start": "study"}, TypeError, "Optuna study"),
        ({"transfer_mode": "all"}, ValueError, "transfer_mode"),
    ],
)
def test_invalid_advanced_options_fail_before_trial_allocation(options, error, message):
    with pytest.raises(error, match=message):
        _constant().tune(
            torch.zeros(100, 1).double(),
            search_space=_empty,
            n_trials=1,
            seeds=[0],
            n_origins=1,
            horizon=8,
            **options,
        )


def test_estimator_error_remains_primary_and_reserved_trial_is_failed():
    failure = RuntimeError("custom workspace cannot be estimated")

    def estimate(params, context):
        raise failure

    study = optuna.create_study(direction="maximize")
    with pytest.raises(RuntimeError) as caught:
        _constant().tune(
            torch.zeros(100, 1).double(),
            search_space=_empty,
            n_trials=2,
            seeds=[0],
            n_origins=1,
            study=study,
            horizon=8,
            memory_budget="1MiB",
            memory_estimator=estimate,
        )
    assert caught.value is failure
    assert len(study.trials) == 1 and study.trials[0].state == optuna.trial.TrialState.FAIL


def test_cleanup_attempts_every_reserved_trial_and_preserves_numerical_error(monkeypatch):
    primary = ArithmeticError("candidate computation failed")

    def factory(params, seed):
        raise primary

    study = optuna.create_study(direction="maximize")
    tell = study.tell
    attempted = []

    def failing_storage(trial, *args, **kwargs):
        attempted.append(trial.number)
        if trial.number == 0:
            raise OSError("storage unavailable for first trial")
        return tell(trial, *args, **kwargs)

    monkeypatch.setattr(study, "tell", failing_storage)
    with pytest.raises(ArithmeticError) as caught:
        _constant().tune(
            torch.zeros(100, 1).double(),
            search_space=_empty,
            factory=factory,
            n_trials=3,
            seeds=[0],
            n_origins=1,
            horizon=8,
            screening=True,
            study=study,
            catch_exceptions=False,
        )
    assert caught.value is primary
    assert attempted == [0, 1, 2]
    assert "storage unavailable" in " ".join(primary.__notes__)
    assert all(trial.state == optuna.trial.TrialState.FAIL for trial in study.trials[1:])


def test_lazy_base_preflight_uses_inferred_training_precision(monkeypatch):
    model = rd.ESN(reservoir_size=100)
    monkeypatch.setattr(model, "build", lambda *args, **kwargs: pytest.fail("allocated base"))
    # Float32 recurrence would fit; the inferred float64 recurrence would not.
    with pytest.raises(ValueError, match="before construction"):
        model.tune(
            torch.zeros(100, 1).double(),
            memory_budget=MemoryBudget(60_000, worker_overhead=0),
        )


def test_history_memory_context_reserves_grown_inputs_and_grown_feature_traces():
    contexts = []

    def estimate(params, context):
        contexts.append(context)
        return MemoryEstimate(1_000_000)

    model = _constant()
    model.tune(
        torch.zeros(100, 1).double(),
        search_space=_empty,
        n_trials=1,
        n_origins=1,
        horizon=8,
        execution="history",
        memory_budget=MemoryBudget("2MiB", worker_overhead=0),
        memory_estimator=estimate,
    )
    assert len(contexts) == 1
    context = contexts[0]
    # First forecast origin follows 80 observations; replay appends 8 values.
    assert context.forecast_samples == 88
    assert context.replay_history_bytes == 3 * 88 * 8
