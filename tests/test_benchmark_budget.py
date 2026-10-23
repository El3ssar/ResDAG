"""Independent equal-budget, work-count and held-out-data oracles."""

# ruff: noqa: E402 -- optional Optuna must be checked before importing its harness.

import json

import numpy as np
import pytest

optuna = pytest.importorskip("optuna")

from benchmarks.rc_bench import budget
from benchmarks.rc_bench.budget_adapters import CommonParams, PythonBudgetAdapter
from benchmarks.rc_bench.config import Point
from benchmarks.rc_bench.data import prepare_point


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


class Candidate:
    def __init__(self, adapter):
        self.adapter = adapter

    def forecast(self, history, horizon):
        self.adapter.histories.append(history.copy())
        self.adapter.clock.value += self.adapter.cost
        delta = history[-1] - history[-2]
        prediction = history[-1] + np.arange(1, horizon + 1)[:, None] * delta
        return prediction, {"forecast_seconds": self.adapter.cost}


class Adapter:
    def __init__(self, clock=None, cost=1, failures=0):
        self.clock = clock or Clock()
        self.cost, self.failures = cost, failures
        self.training, self.histories = [], []

    def fit(self, training, params, seed, washout):
        self.training.append((training.copy(), seed, washout))
        self.clock.value += self.cost
        if self.failures:
            self.failures -= 1
            raise RuntimeError("deliberate candidate failure")
        return Candidate(self), {"fit_calls": 1}

    def close(self):
        pass


def protocol():
    return budget.Protocol("lorenz", 10, 2, 2, 0.02, 0.9056, (10, 12, 14), (16, 18, 20))


def values(count=22):
    return np.repeat(np.arange(count, dtype=float)[:, None], 3, axis=1)


def study():
    return optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(seed=7),
        pruner=optuna.pruners.NopPruner(),
    )


def test_exact_deadline_excludes_even_a_fully_emitted_but_late_candidate():
    clock = Clock()
    adapter = Adapter(clock)
    result = budget.tune_track(
        adapter,
        values(16),
        protocol(),
        study(),
        12,
        smoke=True,
        max_trials=3,
        deadline=budget.Deadline(12, clock),
    )
    assert result["winner"] is None and result["eligible_complete_trials"] == 0
    assert result["state_counts"] == {"FAIL": 1}
    assert result["fit_attempts"] == 3 and result["forecast_attempts"] == 9
    assert result["trials"][0]["eligible"] is False
    assert "BudgetExpiredError" in result["trials"][0]["failure"]


def test_only_finished_candidate_can_win_and_later_partial_work_is_charged():
    clock = Clock()
    result = budget.tune_track(
        Adapter(clock),
        values(16),
        protocol(),
        study(),
        13,
        smoke=True,
        max_trials=3,
        deadline=budget.Deadline(13, clock),
    )
    assert result["winner"]["number"] == 0
    assert result["state_counts"] == {"COMPLETE": 1, "FAIL": 1}
    assert result["fit_attempts"] == 4 and result["forecast_attempts"] == 9
    assert result["elapsed_seconds"] == 13
    assert result["overrun_seconds"] == 0
    assert len(result["trials"][1]["work"]["fits"]) == 1


def test_single_operation_overrun_is_reported_without_a_retry_or_winner():
    clock = Clock()
    adapter = Adapter(clock, cost=2)
    result = budget.tune_track(
        adapter,
        values(16),
        protocol(),
        study(),
        1,
        smoke=True,
        max_trials=3,
        deadline=budget.Deadline(1, clock),
    )
    assert len(adapter.training) == 1 and not adapter.histories
    assert result["elapsed_seconds"] == 2 and result["overrun_seconds"] == 1
    assert result["trials"][0]["overrun_seconds"] == 1
    assert result["winner"] is None


def test_failed_attempt_is_preserved_and_success_has_exactly_three_fits_nine_forecasts():
    clock = Clock()
    adapter = Adapter(clock, failures=1)
    result = budget.tune_track(
        adapter,
        values(16),
        protocol(),
        study(),
        100,
        smoke=True,
        max_trials=2,
        deadline=budget.Deadline(100, clock),
    )
    assert result["state_counts"] == {"FAIL": 1, "COMPLETE": 1}
    assert result["winner"]["number"] == 1
    assert result["fit_attempts"] == 4 and result["forecast_attempts"] == 9
    assert len(result["trials"][1]["work"]["runs"]) == 9
    assert "deliberate candidate failure" in result["trials"][0]["failure"]
    assert len(adapter.training) == 4  # one failed construction/fit, then three seeds


def test_each_seed_fits_only_training_and_receives_only_observed_origin_history():
    adapter = Adapter(cost=0)
    params = CommonParams(8, 0.9, 0.3, 1.0, 0.2, 3, 1e-6)
    ledger, _ = budget.evaluate_candidate(
        adapter, values(), params, protocol(), budget.TRAIN_SEEDS, protocol().test_origins
    )
    assert len(adapter.training) == 3 and len(adapter.histories) == 9
    assert len(ledger["runs"]) == 9
    for training, _, washout in adapter.training:
        np.testing.assert_array_equal(training, values(10))
        assert washout == 2
    for index, history in enumerate(adapter.histories):
        origin = protocol().test_origins[index % 3]
        np.testing.assert_array_equal(history[:, 0], [origin - 2, origin - 1])


def test_tuning_rejects_any_final_test_rows():
    with pytest.raises(ValueError, match="never final test"):
        budget.tune_track(Adapter(), values(22), protocol(), study(), 1, max_trials=1)


def test_final_fit_error_attempts_all_seeds_and_does_not_invent_forecast_scores():
    adapter = Adapter(cost=0, failures=1)
    params = CommonParams(8, 0.9, 0.3, 1.0, 0.2, 3, 1e-6)
    ledger, predictions = budget.evaluate_candidate(
        adapter,
        values(),
        params,
        protocol(),
        budget.TEST_SEEDS,
        protocol().test_origins,
        keep_errors=True,
    )
    assert len(adapter.training) == 5
    assert len(ledger["runs"]) == 15
    failed = [row for row in ledger["runs"] if row["status"] == "failed"]
    assert len(failed) == 3 and len(predictions) == 12
    assert all("0.3" not in row and "0.5" not in row for row in failed)
    assert len(adapter.histories) == 12  # failed fit is never retried


def test_final_forecast_error_keeps_other_origins_and_seeds_without_zero_imputation():
    adapter = Adapter(cost=0)
    original_fit = adapter.fit

    def fit(*args):
        candidate, timing = original_fit(*args)
        original = candidate.forecast
        called = 0

        def forecast(history, horizon):
            nonlocal called
            called += 1
            if called == 2:
                raise RuntimeError("forecast API failed")
            return original(history, horizon)

        candidate.forecast = forecast
        return candidate, timing

    adapter.fit = fit
    params = CommonParams(8, 0.9, 0.3, 1.0, 0.2, 3, 1e-6)
    ledger, _ = budget.evaluate_candidate(
        adapter,
        values(),
        params,
        protocol(),
        budget.TEST_SEEDS,
        protocol().test_origins,
        keep_errors=True,
    )
    assert len(ledger["fits"]) == 5 and len(ledger["forecasts"]) == 15
    assert sum(row["status"] == "failed" for row in ledger["runs"]) == 5
    assert all("0.3" not in row for row in ledger["runs"] if row["status"] == "failed")


def test_later_forecast_origin_does_not_expand_training_targets_or_scaling():
    point = Point(reservoir_size=8, train_len=9, warmup=2, horizon=2, origin=20)
    raw = values(22)
    transformed, meta = prepare_point(raw, point)
    changed = raw.copy()
    changed[10:] = 1e9
    other, other_meta = prepare_point(changed, point)
    assert meta == other_meta
    assert meta["normalization"]["fit_rows"] == [3, 10]
    assert meta["train_targets"] == [1, 10]
    assert meta["forecast_truth"] == [20, 22]
    np.testing.assert_array_equal(transformed[:10], other[:10])
    np.testing.assert_array_equal(meta["normalization"]["scale"], raw[3:10].std(0))


@pytest.mark.parametrize("origin", [True, 9, 10.5, -1])
def test_origin_must_follow_the_fixed_training_boundary(origin):
    with pytest.raises(ValueError, match="origin"):
        Point(reservoir_size=8, train_len=9, warmup=2, horizon=2, origin=origin)


def test_scoring_preserves_nonfinite_prefix_and_right_censoring():
    truth = np.zeros((4, 3))
    pred = np.full_like(truth, 0.1)
    pred[2:] = np.nan
    result = budget.score(pred, truth, protocol())
    assert result["diverged"] and result["0.3"]["valid_steps"] == 2
    assert not result["0.3"]["censored"]
    result = budget.score(truth, truth, protocol())
    assert result["0.3"]["censored"]


@pytest.mark.parametrize("system", ["lorenz", "rossler"])
def test_canonical_protocol_has_disjoint_train_validation_test_and_ten_lt(system):
    p = budget.Protocol.canonical(system)
    assert p.validation_origins[0] == p.train_stop
    assert p.test_origins[0] == p.validation_stop
    assert p.horizon * p.dt * p.lambda_max >= 10
    assert len(p.validation_origins) == len(p.test_origins) == 3
    assert not set(budget.TRAIN_SEEDS) & set(budget.TEST_SEEDS)


@pytest.mark.parametrize("final_failure", [False, True])
def test_cli_freezes_all_winners_before_reading_any_final_test_values(
    monkeypatch, tmp_path, final_failure
):
    from contextlib import nullcontext

    # Fake adapters isolate the clock/leakage oracle from native resources.
    monkeypatch.setattr(budget, "single_thread_policy", lambda: nullcontext({"synthetic": True}))
    data, output = tmp_path / "data", tmp_path / "output"
    data.mkdir()
    (data / "data.json").write_text("{}")
    adapters = []
    real_load = np.load
    for system in budget.SYSTEMS:
        p = budget.Protocol.canonical(system, smoke=True)
        np.save(data / f"{system}.npy", values(p.test_stop))

    class Guarded:
        def __init__(self, array, system):
            self.array, self.system = array, system

        def __len__(self):
            return len(self.array)

        def __getitem__(self, item):
            if item.stop > budget.Protocol.canonical(self.system, smoke=True).validation_stop:
                frozen = json.loads((output / "winners.json").read_text())
                assert len(frozen["tracks"]) == 6
                assert not any("test" in t for t in frozen["tracks"].values())
            return self.array[item]

    def load(path, *args, **kwargs):
        return Guarded(real_load(path, *args, **kwargs), path.stem)

    def prepare(_, **kwargs):
        adapter = Adapter(cost=0, failures=int(final_failure and len(adapters) == 6))
        adapters.append(adapter)
        return adapter, {"startup_synthetic_warmup_seconds": 0.01}

    monkeypatch.setattr(budget.np, "load", load)
    monkeypatch.setattr(budget, "prepare_adapter", prepare)
    monkeypatch.setattr(budget, "julia_storage_pilot", lambda: {"selection": "dense"})
    monkeypatch.setattr(budget, "_source_manifest", lambda: {"files_sha256": {}})
    assert budget.main(
        [
            "--data-dir",
            str(data),
            "--output-dir",
            str(output),
            "--smoke",
            "--budget-seconds",
            "10",
            "--max-trials",
            "1",
        ]
    ) == int(final_failure)
    result = json.loads((output / "results.json").read_text())
    assert len(result["tracks"]) == 6
    assert all(len(t["test"]["work"]["runs"]) == 15 for t in result["tracks"].values())
    if final_failure:
        first = next(iter(result["tracks"].values()))["test"]
        assert first["summary"] is None and not first["complete"]
        assert sum(row["status"] == "failed" for row in first["work"]["runs"]) == 3
    else:
        assert all(t["test"]["summary"]["runs"] == 15 for t in result["tracks"].values())
    assert sum(len(a.training) for a in adapters) == 6 * (3 + 5)
    assert sum(len(a.histories) for a in adapters) == 6 * (9 + 15) - (3 if final_failure else 0)
    assert len(list(output.glob("*.sqlite3"))) == 6
    assert "≥" in (output / "report.md").read_text()


@pytest.mark.parametrize("library", ["resdag-cpu", "reservoirpy"])
def test_real_common_python_recipe_has_random_bias_and_no_readout_intercept(library):
    if library == "reservoirpy":
        pytest.importorskip("reservoirpy")
    import torch

    import resdag as rd

    params = CommonParams(20, 0.9, 0.3, 0.2, 0.1, 6, 1e-3)
    samples = np.random.default_rng(7).normal(size=(45, 3))
    candidate, timing = PythonBudgetAdapter(library).fit(samples, params, 7, 3)
    assert timing["fit_calls"] == 1
    if library == "resdag-cpu":
        cell = next(m for m in candidate.model.modules() if isinstance(m, rd.ESNCell))
        bias = cell.bias_h.detach().numpy()
        readout = next(m for m in candidate.model.modules() if isinstance(m, rd.RidgeReadoutLayer))
        assert readout.bias is None
        assert cell.weight_hh.dtype == torch.float64
    else:
        bias = candidate.reservoir.bias
        assert next(n for n in candidate.model.nodes if hasattr(n, "fit_bias")).fit_bias is False
        win = candidate.reservoir.Win
        win = win.toarray() if hasattr(win, "toarray") else win
        assert np.max(np.abs(win)) <= params.input_scaling
        assert np.count_nonzero(win) == win.size
    assert np.ptp(bias) > 0 and np.max(np.abs(bias)) <= params.bias_scaling
    if library == "resdag-cpu":
        normalized_input = (
            cell.weight_feedback.detach().numpy().ravel()[: params.reservoir_size]
            / params.input_scaling
        )
        assert not np.allclose(bias / params.bias_scaling, normalized_input, atol=1e-14, rtol=0)
        replay, _ = PythonBudgetAdapter(library).fit(samples, params, 7, 3)
        replay_cell = next(m for m in replay.model.modules() if isinstance(m, rd.ESNCell))
        np.testing.assert_array_equal(bias, replay_cell.bias_h.detach().numpy())
        changed, _ = PythonBudgetAdapter(library).fit(samples, params, 19, 3)
        changed_cell = next(m for m in changed.model.modules() if isinstance(m, rd.ESNCell))
        assert not np.array_equal(bias, changed_cell.bias_h.detach().numpy())
    first, _ = candidate.forecast(samples[-3:], 4)
    again, _ = candidate.forecast(samples[-3:], 4)
    assert first.dtype == np.float64 and np.isfinite(first).all()
    np.testing.assert_array_equal(first, again)
