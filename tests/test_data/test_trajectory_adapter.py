"""Physical-time and portable-metadata oracles for the lazy trajectory adapter."""

import builtins
import copy
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from resdag.data import _trajectory as adapter


@pytest.fixture
def trajectory():
    return pytest.importorskip("tsdynamics").Trajectory


def _context():
    return {
        "version": 1,
        "start_time": 2.0,
        "last_time": 2.3,
        "dt": 0.1,
        "n_samples": 4,
        "variables": ["position"],
        "source": {"metadata": {"system": "measured"}, "system": None},
        "sampling": {
            "requested_dt": None,
            "effective_dt": 0.1,
            "method": "linear",
            "resampled": False,
            "original_n_samples": 4,
            "original_start_time": 2.0,
            "original_last_time": 2.3,
        },
    }


def test_ordinary_values_and_context_validation_never_import_optional_dependency(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith("tsdynamics"):
            raise AssertionError("ordinary paths must not import TSDynamics")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert not adapter.is_trajectory(torch.zeros(2, 1))
    assert not adapter.is_trajectory(np.zeros((2, 1)))
    assert adapter.validate_resampling_options(None, "linear") == (None, "linear")
    assert adapter.validate_trajectory_context(_context()) == _context()


def test_missing_optional_dependency_has_actionable_type_preflight(monkeypatch):
    original = adapter.importlib.import_module

    def unavailable(name, *args, **kwargs):
        if name.startswith("tsdynamics"):
            raise ImportError("dependency absent")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(adapter.importlib, "import_module", unavailable)
    with pytest.raises(ImportError, match=r"resdag\[dynamics\].*3.12"):
        adapter.trajectory_type()


@pytest.mark.parametrize("dt", [True, np.bool_(True), 0, -1, float("inf"), float("nan"), "0.1", 1j])
def test_invalid_resampling_intervals_are_rejected(dt):
    with pytest.raises(ValueError, match="positive finite real"):
        adapter.validate_resampling_options(dt, "linear")


@pytest.mark.parametrize("method", [None, 3, "nearest", "LINEAR"])
def test_invalid_interpolation_methods_are_rejected(method):
    with pytest.raises(ValueError, match="resample_method"):
        adapter.validate_resampling_options(None, method)


def test_regular_samples_preserve_values_times_and_ignore_stale_metadata_dt(trajectory):
    times = np.array([1.0, 1.25, 1.5, 1.75])
    values = np.arange(4, dtype=np.float32)[:, None]
    source = trajectory(times, values, meta={"dt": 99, "params": {"a": 2}}, variables=("x",))
    prepared = adapter.prepare_trajectory(source)
    assert prepared.times is source.t
    assert prepared.values is source.y
    assert prepared.context["dt"] == 0.25
    assert prepared.context["sampling"]["resampled"] is False
    assert prepared.context["source"]["metadata"]["dt"] == 99
    prepared.context["source"]["metadata"]["params"]["a"] = 10
    assert source.meta["params"]["a"] == 2


@pytest.mark.parametrize("method", ["linear", "pchip", "cubic"])
def test_irregular_linear_signal_resamples_to_same_count_and_both_endpoints(trajectory, method):
    times = np.array([2.0, 2.07, 2.23, 2.3])
    source = trajectory(times, np.column_stack((2 * times + 1, -3 * times)), variables=("x", "y"))
    prepared = adapter.prepare_trajectory(source, method=method)
    expected = np.array([2.0, 2.1, 2.2, 2.3])
    np.testing.assert_allclose(prepared.times, expected, rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        prepared.values, np.column_stack((2 * expected + 1, -3 * expected)), rtol=1e-14, atol=1e-14
    )
    assert prepared.times[0] == times[0] and prepared.times[-1] == times[-1]
    assert prepared.context["sampling"]["resampled"] is True
    assert prepared.context["dt"] == pytest.approx(0.1)


def test_cubic_interpolation_matches_independent_cubic_polynomial(trajectory):
    times = np.array([0.0, 0.11, 0.38, 0.6])
    source = trajectory(times, times**3 - 2 * times + 1, variables=("x",))
    prepared = adapter.prepare_trajectory(source, method="cubic")
    expected = prepared.times**3 - 2 * prepared.times + 1
    np.testing.assert_allclose(prepared.values[:, 0], expected, rtol=1e-14, atol=1e-14)


def test_explicit_interval_anchors_last_observation_without_extrapolation(trajectory):
    times = np.array([0.03, 0.1, 0.24, 0.35])
    source = trajectory(times, 2 * times + 1, variables=("x",))
    prepared = adapter.prepare_trajectory(source, dt=0.1)
    np.testing.assert_allclose(prepared.times, [0.05, 0.15, 0.25, 0.35], atol=1e-15, rtol=0)
    assert prepared.times[0] >= times[0] and prepared.times[-1] == times[-1]
    np.testing.assert_allclose(prepared.values[:, 0], 2 * prepared.times + 1, rtol=1e-14)
    with pytest.raises(ValueError, match="at least two"):
        adapter.prepare_trajectory(source, dt=1.0)


def test_explicit_interval_retains_an_endpoint_with_roundoff(trajectory):
    source = trajectory([0.1, 0.2, 0.3, 0.4], [1.0, 2.0, 3.0, 4.0], variables=("x",))
    prepared = adapter.prepare_trajectory(source, dt=0.1)
    np.testing.assert_allclose(prepared.times, source.t, atol=1e-15, rtol=0)


def test_time_grid_precision_matches_public_sampling_contract(trajectory):
    source = trajectory(2.0 + 0.01 * np.arange(12), np.arange(12), variables=("x",))
    prepared = adapter.prepare_trajectory(source)
    adapter.validate_time_grid(prepared.times, prepared.context["dt"])
    result = adapter.forecast_trajectory(np.arange(20)[:, None], prepared.context)
    assert result.dt is not None
    assert result.dt == pytest.approx(0.01)


@pytest.mark.parametrize("irregular", [False, True])
def test_large_clock_offset_cannot_silently_produce_an_unrepresentable_grid(trajectory, irregular):
    times = 1e6 + 0.01 * np.arange(12)
    if irregular:
        times[4] += 0.001
    source = trajectory(times, np.arange(12), variables=("x",))
    with pytest.raises(ValueError, match="shift the time origin.*coarser"):
        adapter.prepare_trajectory(source)
    # Moving only the origin removes the floating-point resolution obstacle.
    shifted = trajectory(np.arange(12) * 0.01, np.arange(12), variables=("x",))
    prepared = adapter.prepare_trajectory(shifted)
    context = prepared.context
    context["start_time"] += 1e6
    context["last_time"] += 1e6
    context["sampling"]["original_start_time"] += 1e6
    context["sampling"]["original_last_time"] += 1e6
    with pytest.raises(ValueError, match="shift the time origin.*coarser"):
        adapter.forecast_trajectory(np.arange(20)[:, None], context)


def test_numpy_string_options_and_names_become_weights_only_safe_primitives(trajectory):
    dt, method = adapter.validate_resampling_options(None, np.str_("linear"))
    assert dt is None and type(method) is str
    source = trajectory([0.0, 0.1, 0.2], [1.0, 2.0, 3.0], variables=(np.str_("x"),))
    prepared = adapter.prepare_trajectory(source, method=np.str_("linear"))
    assert type(prepared.context["sampling"]["method"]) is str
    assert all(type(name) is str for name in prepared.context["variables"])
    names = adapter._signal_names([np.str_("x")], 1)
    assert type(names[0]) is str
    json.dumps(prepared.context, allow_nan=False)
    buffer = io.BytesIO()
    torch.save({"method": method, "names": names, "context": prepared.context}, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=True)
    assert adapter.validate_trajectory_context(restored["context"]) == prepared.context


@pytest.mark.parametrize(
    "times", [[0, 0.1, 0.1], [0.2, 0.1, 0], [0, np.nan, 0.2], [0, 0.1, np.inf], [0]]
)
def test_invalid_time_axes_are_rejected(trajectory, times):
    source = trajectory(times, np.zeros((len(times), 1)))
    with pytest.raises(ValueError, match="times|two"):
        adapter.prepare_trajectory(source)


@pytest.mark.parametrize(
    "values", [[0, np.nan, 1], [0, np.inf, 1], [0j, 1j, 2j], [True, False, True]]
)
def test_invalid_real_signal_values_are_rejected(trajectory, values):
    source = trajectory([0, 0.1, 0.2], values)
    with pytest.raises(ValueError, match="finite real"):
        adapter.prepare_trajectory(source)


@pytest.mark.parametrize("names", [["x", "x"], ["", "y"], ["  ", "y"]])
def test_invalid_mutated_public_signal_names_are_rejected(trajectory, names):
    source = trajectory([0, 0.1, 0.2], np.zeros((3, 2)))
    source.meta["variables"] = names
    with pytest.raises(ValueError, match="variables"):
        adapter.prepare_trajectory(source)


def test_context_preserves_safe_source_metadata_without_checkpointing_live_system(trajectory):
    system = SimpleNamespace(variables=("x",))
    source = trajectory(
        [0, 0.1, 0.2],
        [1.0, 2.0, 3.0],
        system=system,
        meta={
            "params": {"a": np.float64(2)},
            "ic": np.array([1.0]),
            "max_step": np.inf,
            "opaque": object(),
        },
    )
    prepared = adapter.prepare_trajectory(source)
    assert prepared.system is system
    assert prepared.context["source"]["system"]["name"] == "SimpleNamespace"
    json.dumps(prepared.context, allow_nan=False)
    buffer = io.BytesIO()
    torch.save({"trajectory": prepared.context}, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=True)["trajectory"]
    assert adapter.validate_trajectory_context(restored) == prepared.context
    assert restored["source"]["metadata"]["ic"] == [1.0]
    assert restored["source"]["metadata"]["params"]["a"] == 2.0


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("dt", -0.1),
        ("dt", float("nan")),
        ("n_samples", 1),
        ("n_samples", True),
        ("last_time", 3.0),
        ("variables", ["x", "x"]),
        ("source", {"system": object()}),
        ("sampling", {}),
    ],
)
def test_malformed_checkpoint_context_is_rejected(field, value):
    context = _context()
    context[field] = value
    with pytest.raises(ValueError, match="context|metadata|variables"):
        adapter.validate_trajectory_context(context)


def test_forecast_times_names_source_and_transient_system_are_honest(trajectory):
    context = _context()
    context["source"]["metadata"].update({"solver": "rk45", "params": {"a": 2}})
    before = copy.deepcopy(context)
    system = SimpleNamespace(variables=("position",))
    result = adapter.forecast_trajectory(np.array([[4.0], [5.0], [6.0]]), context, system=system)
    assert isinstance(result, trajectory) and result.system is system
    np.testing.assert_allclose(result.t, [2.4, 2.5, 2.6], rtol=0, atol=1e-15)
    assert result.variables == ("position",)
    assert "solver" not in result.meta
    assert result.meta["source"]["metadata"]["solver"] == "rk45"
    result.meta["source"]["metadata"]["params"]["a"] = 3
    assert context == before
    checkpoint_result = adapter.forecast_trajectory(np.ones((1, 1)), context)
    assert checkpoint_result.system is None


def test_explicit_warmup_prediction_times_have_no_duplicate_seam(trajectory):
    context = _context()
    times = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
    result = adapter.forecast_trajectory(np.arange(5)[:, None], context, times=times)
    np.testing.assert_array_equal(result.t, times)
    assert len(result.t) == context["n_samples"] - 1 + 2


@pytest.mark.parametrize("with_warmup", [False, True])
def test_forecast_initial_condition_describes_first_returned_prediction(trajectory, with_warmup):
    context = _context()
    context["source"]["metadata"]["ic"] = [-99.0]
    before = copy.deepcopy(context)
    predictions = np.array([[4.0], [5.0], [6.0]])
    times = np.array([2.1, 2.2, 2.3]) if with_warmup else None
    result = adapter.forecast_trajectory(predictions, context, times=times)
    np.testing.assert_array_equal(result.meta["ic"], [4.0])
    assert result.meta["t0"] == result.t[0]
    assert result.meta["source"]["metadata"]["ic"] == [-99.0]
    assert not np.shares_memory(result.meta["ic"], predictions)
    predictions[0, 0] = 100.0
    np.testing.assert_array_equal(result.meta["ic"], [4.0])
    assert context == before


@pytest.mark.parametrize("times", [[2.4, 2.4], [2.5, 2.4], [2.4], [2.4, float("nan")]])
def test_invalid_explicit_output_times_are_rejected(times):
    with pytest.raises(ValueError, match="Forecast times"):
        adapter.forecast_trajectory(np.ones((2, 1)), _context(), times=times)
