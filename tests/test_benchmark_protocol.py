"""Independent clock, leakage and failure oracles for the comparison harness."""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from benchmarks.rc_bench import internal_cells, run
from benchmarks.rc_bench.adapters.base import Adapter
from benchmarks.rc_bench.adapters.julia_adapter import ReservoirComputingJLAdapter
from benchmarks.rc_bench.adapters.resdag_adapter import ResdagCPUAdapter
from benchmarks.rc_bench.adapters.reservoirpy_adapter import ReservoirPyAdapter
from benchmarks.rc_bench.config import Context, HParams, Point
from benchmarks.rc_bench.data import lorenz, prepare_point
from benchmarks.rc_bench.metrics import valid_prediction_time
from benchmarks.rc_bench.report import _vpt_cell, render
from benchmarks.rc_bench.timing import RunResult


def _point(**kwargs):
    return Point(reservoir_size=20, train_len=12, warmup=3, horizon=4, **kwargs)


def test_windows_have_no_training_targets_in_the_forecast_truth():
    point = _point()
    series = np.arange(point.series_len, dtype=float)[:, None]
    x, y = Adapter._train_window(series, point)
    history = Adapter._forecast_history(series, point)
    truth = Adapter._forecast_truth(series, point)
    np.testing.assert_array_equal(x[:, 0], np.arange(12))
    np.testing.assert_array_equal(y[:, 0], np.arange(1, 13))
    np.testing.assert_array_equal(history[:, 0], [10, 11, 12])
    np.testing.assert_array_equal(truth[:, 0], [13, 14, 15, 16])
    assert history[-1, 0] == y[-1, 0]
    assert truth[0, 0] == history[-1, 0] + 1


def test_future_observations_cannot_change_training_scaling():
    point = _point()
    series = np.arange(point.series_len * 2, dtype=float).reshape(-1, 2)
    changed = series.copy()
    changed[point.forecast_start :] = 1e9
    baseline, protocol = prepare_point(series, point)
    perturbed, other = prepare_point(changed, point)
    assert protocol == other
    observed = series[4:13]
    np.testing.assert_array_equal(protocol["normalization"]["mean"], observed.mean(0))
    np.testing.assert_array_equal(protocol["normalization"]["scale"], observed.std(0))
    np.testing.assert_array_equal(baseline[:13], perturbed[:13])
    np.testing.assert_allclose(baseline[4:13].mean(0), 0, atol=1e-15)
    np.testing.assert_allclose(baseline[4:13].std(0), 1, atol=1e-15)
    assert protocol["normalization"]["fit_rows"] == [4, 13]
    assert protocol["forecast_truth"] == [13, 17]


def test_constant_training_channel_is_finite_without_looking_ahead():
    point = _point()
    series = np.ones((point.series_len, 1))
    series[13:] = 9
    result, protocol = prepare_point(series, point)
    assert protocol["normalization"]["scale"] == [1]
    np.testing.assert_array_equal(result[:13], 0)
    np.testing.assert_array_equal(result[13:], 8)


def test_generator_preserves_original_units():
    import resdag as rd

    expected = rd.lorenz(15, seed=7, normalize=None, dtype=torch.float64)
    np.testing.assert_array_equal(lorenz(15, seed=7), expected.squeeze(0).numpy())


@pytest.mark.parametrize("field,value", [("warmup", 0), ("train_len", True), ("horizon", -1)])
def test_point_rejects_invalid_counts(field, value):
    with pytest.raises(ValueError, match=field):
        replace(_point(), **{field: value})


def test_point_requires_training_rows_after_washout():
    with pytest.raises(ValueError, match="warmup"):
        replace(_point(), warmup=12)


def test_window_rejects_missing_truth_instead_of_truncating_score():
    point = _point()
    with pytest.raises(ValueError, match="at least 17"):
        Adapter._forecast_truth(np.zeros((16, 2)), point)


def test_fixed_training_scale_stops_at_first_failure_and_uses_feature_rms():
    truth = np.zeros((4, 2))
    # Both channels have the same normalized error; the large raw units of
    # channel two must not dominate. Equality with the threshold is invalid.
    error = np.array([0.1, 0.29, 0.3, 0.1])[:, None]
    pred = error * np.array([1.0, 1000.0])
    assert valid_prediction_time(pred, truth, scale=np.array([1.0, 1000.0])) == 2
    assert valid_prediction_time(pred, truth, 0.5, scale=np.array([1.0, 1000.0])) == 4


def test_scoring_scale_is_independent_of_future_truth_variance():
    pred = np.full((5, 1), 0.4)
    truth = np.zeros_like(pred)
    assert valid_prediction_time(pred, truth, scale=np.ones(1)) == 0
    truth[-1] = 1e6
    assert valid_prediction_time(pred, truth, scale=np.ones(1)) == 0


@pytest.mark.parametrize("scale", [np.array([0.0]), np.array([np.nan]), np.ones(2)])
def test_metric_rejects_invalid_training_scales(scale):
    with pytest.raises(ValueError, match="scale"):
        valid_prediction_time(np.zeros((2, 1)), np.zeros((2, 1)), scale=scale)


def test_metric_requires_complete_forecasts_and_finite_truth():
    with pytest.raises(ValueError, match="matching"):
        valid_prediction_time(np.zeros((1, 1)), np.zeros((2, 1)), scale=np.ones(1))
    with pytest.raises(ValueError, match="finite"):
        valid_prediction_time(np.zeros((2, 1)), np.full((2, 1), np.nan), scale=np.ones(1))


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_first_nonfinite_prediction_ends_a_valid_prefix(invalid):
    pred = np.zeros((5, 1))
    pred[2] = invalid
    assert valid_prediction_time(pred, np.zeros_like(pred), scale=np.ones(1)) == 2


class _NextIndex:
    """Independent y(t)=x(t)+1 oracle, including the warmup output."""

    def __init__(self):
        self.histories = []
        self.calls = 0
        self.resets = 0

    def fit(self, x, y, *, warmup):
        np.testing.assert_array_equal(y, x + 1)
        self.training = x.copy(), warmup

    def run(self, history):
        self.histories.append(history.copy())
        return history + 1

    def __call__(self, value):
        self.calls += 1
        return value + 1

    def reset(self):
        self.resets += 1


@pytest.mark.parametrize("horizon", [1, 4])
@pytest.mark.parametrize("timed", [False, True])
def test_reservoirpy_first_prediction_is_the_warmup_output(monkeypatch, horizon, timed):
    point = replace(_point(), horizon=horizon)
    series = np.arange(point.series_len, dtype=float)[:, None]
    adapter = ReservoirPyAdapter()
    oracle = _NextIndex()
    monkeypatch.setattr(adapter, "_build", lambda *_: (oracle, oracle))
    if timed:
        result = adapter.time_forecast(series, HParams(dim=1), point, 2, 1)
        assert result.rmse == 0
        assert len(result.times) == 2
        repetitions = 3
    else:
        pred = adapter.predict_trajectory(series, HParams(dim=1), point)
        np.testing.assert_array_equal(pred[:, 0], np.arange(13, 13 + horizon))
        repetitions = 1
    assert oracle.calls == repetitions * (horizon - 1)
    assert oracle.resets == repetitions
    for history in oracle.histories:
        np.testing.assert_array_equal(history[:, 0], [10, 11, 12])


def test_resdag_honors_input_scaling_and_constructs_cpu_float64():
    adapter = ResdagCPUAdapter()
    hp = HParams(connectivity=0.5, input_scaling=0.2)
    first, dtype = adapter._build(hp, _point())
    second, _ = adapter._build(replace(hp, input_scaling=0.4), _point())
    from resdag.layers import ESNLayer
    from resdag.layers.readouts import RidgeReadoutLayer

    a = next(m.cell for m in first.modules() if isinstance(m, ESNLayer))
    b = next(m.cell for m in second.modules() if isinstance(m, ESNLayer))
    assert dtype == torch.float64
    assert a.weight_feedback.dtype == torch.float64
    assert a.bias_h is None
    assert next(m for m in first.modules() if isinstance(m, RidgeReadoutLayer)).bias is None
    torch.testing.assert_close(b.weight_feedback, a.weight_feedback * 2, rtol=0, atol=0)
    torch.testing.assert_close(a.weight_hh, b.weight_hh, rtol=0, atol=0)


def test_real_resdag_tiny_heldout_workload_is_finite_and_complete():
    point = replace(_point(), train_len=45)
    series, _ = prepare_point(lorenz(point.series_len), point)
    pred = ResdagCPUAdapter().predict_trajectory(series, HParams(), point)
    assert pred.shape == (4, 3)
    assert np.isfinite(pred).all()


def test_resdag_adapter_forecast_matches_an_independent_next_index_oracle(monkeypatch):
    import pytorch_symbolic as ps

    from resdag import ESNModel
    from resdag.layers.readouts import RidgeReadoutLayer

    def build(*_):
        x = ps.Input((1, 1), dtype=torch.float64)
        head = RidgeReadoutLayer(1, 1, alpha=0, name="output").double()
        return ESNModel(x, head(x)), torch.float64

    point = _point()
    adapter = ResdagCPUAdapter()
    monkeypatch.setattr(adapter, "_build", build)
    series = np.arange(point.series_len, dtype=float)[:, None]
    pred = adapter.predict_trajectory(series, HParams(dim=1), point)
    np.testing.assert_allclose(pred[:, 0], [13, 14, 15, 16], rtol=0, atol=1e-10)
    timing = adapter.time_forecast(series, HParams(dim=1), point, 2, 1)
    assert timing.rmse < 1e-10


def test_real_reservoirpy_first_output_matches_independent_history_boundary():
    pytest.importorskip("reservoirpy")
    point = replace(_point(), horizon=1, train_len=45)
    series, _ = prepare_point(lorenz(point.series_len), point)
    hp = HParams()
    adapter = ReservoirPyAdapter()
    esn, reservoir = adapter._build(hp, point)
    assert reservoir.input_connectivity == 1.0
    assert reservoir.bias == 0.0
    assert next(node for node in esn.nodes if hasattr(node, "fit_bias")).fit_bias is False
    # Independent explicit indices: observed inputs 0..44, targets 1..45,
    # synchronization 43..45; the last emitted value predicts sample 46.
    esn.fit(series[:45], series[1:46], warmup=3)
    reservoir.reset()
    expected = esn.run(np.ascontiguousarray(series[43:46]))[-1]
    actual = adapter.predict_trajectory(series, hp, point)
    np.testing.assert_allclose(actual[0], expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("operation", ["train", "forecast"])
def test_julia_keeps_requested_repeats_and_warmups(monkeypatch, operation):
    adapter = ReservoirComputingJLAdapter()
    seen = []
    monkeypatch.setattr(adapter, "_run", lambda *args: seen.append(args) or RunResult())
    method = adapter.time_train if operation == "train" else adapter.time_forecast
    method(np.zeros((_point().series_len, 1)), HParams(dim=1), _point(), 5, 2)
    assert seen[0][-3:] == (operation, 5, 2)


def test_julia_has_no_implicit_competitor_size_handicap(monkeypatch):
    adapter = ReservoirComputingJLAdapter()
    seen = []
    monkeypatch.setattr(adapter, "_run", lambda *args: seen.append(args) or RunResult())
    point = Point(reservoir_size=2500, train_len=20000, warmup=200, horizon=4)
    adapter.time_train(np.zeros((point.series_len, 1)), HParams(dim=1), point, 3, 1)
    assert seen[0][2] == point


def test_julia_rejects_an_unsupported_activation_before_running():
    with pytest.raises(ValueError, match="tanh"):
        ReservoirComputingJLAdapter().predict_trajectory(
            np.zeros((_point().series_len, 1)), HParams(activation="relu"), _point()
        )


def test_internal_profile_uses_heldout_history_and_truth(monkeypatch):
    import pytorch_symbolic as ps

    from resdag import ESNModel
    from resdag.layers.readouts import RidgeReadoutLayer

    def build(*_):
        x = ps.Input((1, 1), dtype=torch.float64)
        head = RidgeReadoutLayer(1, 1, alpha=0, name="output").double()
        return ESNModel(x, head(x))

    point = _point()
    monkeypatch.setattr(internal_cells, "_build_model", build)
    result = internal_cells._profile_cell(
        "classic_esn",
        20,
        "cpu",
        HParams(dim=1),
        internal_cells.CellPoint(
            reservoir_size=point.reservoir_size,
            train_len=point.train_len,
            warmup=point.warmup,
            horizon=point.horizon,
        ),
        np.arange(point.series_len, dtype=float)[:, None],
        2,
        1,
    )
    assert result["rmse"] < 1e-10


def test_report_preserves_censored_and_diverged_statuses():
    assert _vpt_cell({"p": {"vpt": 4, "vpt_lyap": 0.8, "vpt_censored": True}}, "p").startswith("≥4")
    assert "diverged" in _vpt_cell({"p": {"vpt": 2, "vpt_lyap": 0.4, "diverged": True}}, "p")


class _ProtocolProbe(Adapter):
    key = "probe"
    label = "Protocol probe"

    @classmethod
    def is_available(cls):
        return True, ""

    def time_train(self, series, hp, point, repeats, warmups):
        return RunResult(times=[0.001] * repeats)

    time_forecast = time_train

    def predict_trajectory(self, series, hp, point):
        return self._forecast_truth(series, point)


def _runner(monkeypatch, tmp_path, adapters, metric="precision"):
    context = Context("probe", "Protocol probe", "unit oracle", metric, (_point(),), "horizon")
    monkeypatch.setattr(run, "QUICK_CONTEXTS", (context,))
    monkeypatch.setattr(run, "all_adapters", lambda only: adapters)
    monkeypatch.setattr(
        run, "lorenz", lambda n, seed: np.repeat(np.arange(n, dtype=float)[:, None], 3, axis=1)
    )
    monkeypatch.setattr(run, "collect_env", lambda: {"timestamp": "unit-test"})
    monkeypatch.setattr(run, "_collect_versions", lambda: {})
    output = tmp_path / "result.json"
    return output, ["--quick", "--repeats", "2", "--warmups", "0", "--out", str(output)]


def test_runner_records_heldout_protocol_and_both_thresholds(monkeypatch, tmp_path):
    output, args = _runner(monkeypatch, tmp_path, [_ProtocolProbe()])
    assert run.main(args) == 0
    result = json.loads(output.read_text())
    cell = result["contexts"]["probe"]["points"][0]
    assert result["meta"]["protocol"] == "heldout-v1"
    assert result["meta"]["accuracy_metric"]["threshold"] == 0.3
    assert result["meta"]["accuracy_metric"]["secondary_threshold"] == 0.5
    assert cell["protocol"]["forecast_truth"] == [13, 17]
    assert cell["results"]["probe"]["vpt"] == 4
    assert cell["results"]["probe"]["secondary_vpt"] == 4
    assert cell["results"]["probe"]["vpt_censored"] is True
    assert cell["results"]["probe"]["secondary_vpt_censored"] is True
    assert cell["results"]["probe"]["diverged"] is False
    report = render(result)
    assert "Held-out protocol v1" in report
    assert "fixed per-channel" in report
    assert "0 warmups" in report


def test_available_adapter_failure_returns_nonzero_and_keeps_results(monkeypatch, tmp_path):
    class Broken(_ProtocolProbe):
        def predict_trajectory(self, *_):
            raise RuntimeError("broken available adapter")

    output, args = _runner(monkeypatch, tmp_path, [Broken()])
    assert run.main(args) == 1
    result = json.loads(output.read_text())
    assert (
        "broken available adapter"
        in result["contexts"]["probe"]["points"][0]["results"]["probe"]["error"]
    )


def test_truncated_predictions_fail_instead_of_receiving_a_shorter_score(monkeypatch, tmp_path):
    probe = _ProtocolProbe()
    monkeypatch.setattr(probe, "predict_trajectory", lambda *_: np.zeros((2, 3)))
    _, args = _runner(monkeypatch, tmp_path, [probe])
    assert run.main(args) == 1


def test_numerical_divergence_keeps_prefix_and_explicit_status(monkeypatch, tmp_path):
    probe = _ProtocolProbe()

    def diverging(series, hp, point):
        pred = probe._forecast_truth(series, point).copy()
        pred[2] = np.nan
        return pred

    monkeypatch.setattr(probe, "predict_trajectory", diverging)
    output, args = _runner(monkeypatch, tmp_path, [probe])
    assert run.main(args) == 0
    result = json.loads(output.read_text())["contexts"]["probe"]["points"][0]["results"]["probe"]
    assert result["vpt"] == 2
    assert result["secondary_vpt"] == 2
    assert result["diverged"] is True
    assert result["first_nonfinite_step"] == 2
    assert result["vpt_censored"] is False


def test_missing_optional_adapter_stays_an_explicit_skip(monkeypatch, tmp_path):
    class Missing(_ProtocolProbe):
        key = "missing"

        @classmethod
        def is_available(cls):
            return False, "optional package absent"

    output, args = _runner(monkeypatch, tmp_path, [Missing(), _ProtocolProbe()])
    assert run.main(args) == 0
    result = json.loads(output.read_text())
    assert result["meta"]["availability"]["missing"] == {
        "available": False,
        "reason": "optional package absent",
    }


def test_adapter_mutation_cannot_change_the_next_librarys_data(monkeypatch, tmp_path):
    class Mutating(_ProtocolProbe):
        key = "mutating"

        def predict_trajectory(self, series, hp, point):
            answer = self._forecast_truth(series, point).copy()
            series[:] = 999
            return answer

    probe = _ProtocolProbe()
    original = probe.predict_trajectory

    def check(series, hp, point):
        assert np.max(series) < 999
        return original(series, hp, point)

    monkeypatch.setattr(probe, "predict_trajectory", check)
    _, args = _runner(monkeypatch, tmp_path, [Mutating(), probe])
    assert run.main(args) == 0


def test_wrong_number_of_timings_fails_the_run(monkeypatch, tmp_path):
    probe = _ProtocolProbe()
    monkeypatch.setattr(probe, "time_train", lambda *_: RunResult(times=[0.001]))
    _, args = _runner(monkeypatch, tmp_path, [probe], metric="train")
    assert run.main(args) == 1


def test_historical_report_remains_explicitly_unverified():
    assert "Historical, unverified comparisons" in render({"meta": {}, "contexts": {}})
