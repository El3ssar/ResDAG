"""Physical-clock and checkpoint invariants that need no optional trajectory type."""

import builtins
import copy
import json

import numpy as np
import pytest

from resdag.data import _trajectory_ports as adapter
from resdag.data._trajectory import validate_time_grid


def _port(times, values=None):
    times = np.asarray(times, dtype=np.float64)
    if values is None:
        values = (2 * times + 3)[:, None]
    values = np.asarray(values)
    return adapter._Port(
        times,
        values,
        [f"signal_{i}" for i in range(values.shape[1])],
        {"metadata": {"units": "m"}, "system": None},
        None,
        0,
    )


def _record(start, last, count, name):
    return {
        "variables": [name],
        "source": {"metadata": {"units": "m"}, "system": None},
        "observation": {"start_time": start, "last_time": last, "n_samples": count},
        "resampled": True,
        "held_out_samples": 0,
    }


@pytest.fixture
def context():
    # Origins 7..18 predict far-head labels 9..20 and near-head labels 8..19.
    # Inputs retain the complete shared 2..20 history, including late observations.
    return json.loads(
        json.dumps(
            {
                "version": 2,
                "clock": {
                    "start_time": 2.0,
                    "last_time": 20.0,
                    "dt": 1.0,
                    "n_samples": 19,
                    "reference_input": "state",
                    "requested_dt": None,
                    "method": "linear",
                },
                "inputs": {
                    "state": _record(0.0, 20.0, 21, "position"),
                    "forcing": _record(1.25, 20.25, 39, "force"),
                },
                "outputs": {
                    "far": _record(7.0, 20.0, 53, "far_value"),
                    "near": _record(8.0, 20.0, 25, "near_value"),
                },
                "supervision": {"start": 5, "stop": 17, "prediction_leads": {"far": 2, "near": 1}},
            }
        )
    )


def _validate(context, **kwargs):
    return adapter.validate_trajectory_ports_context(
        context,
        ("state", "forcing"),
        (1, 1),
        ("far", "near"),
        (1, 1),
        **kwargs,
    )


def test_shared_grid_uses_declared_reference_interval_and_latest_common_endpoint():
    state = _port(np.arange(21))
    forcing = _port(np.arange(1.25, 20.26, 0.5))
    grid, interval = adapter._shared_grid((state, forcing), None)
    np.testing.assert_array_equal(grid, np.arange(2, 21))
    assert interval == 1
    reversed_grid, interval = adapter._shared_grid((forcing, state), None)
    np.testing.assert_array_equal(reversed_grid, np.arange(1.5, 20.01, 0.5))
    assert interval == 0.5
    # Neither endpoint is invented from a target's time range.
    for port in (state, forcing):
        sampled = adapter._sample(port, grid, 1, "linear", "input")
        np.testing.assert_array_equal(sampled[:, 0], 2 * grid + 3)


def test_explicit_interval_right_anchors_and_never_extends_observed_support():
    grid, interval = adapter._shared_grid((_port([2, 2.2, 2.9, 4]),), 0.75)
    np.testing.assert_array_equal(grid, [2.5, 3.25, 4])
    assert interval == 0.75
    singleton, interval = adapter._shared_grid((_port([1e9]),), 1)
    np.testing.assert_array_equal(singleton, [1e9])
    assert interval == 1
    wider_than_support, _ = adapter._shared_grid((_port([2, 3]),), 2)
    np.testing.assert_array_equal(wider_than_support, [3])


@pytest.mark.parametrize(
    "ports,dt,match",
    [
        ((), None, "at least one"),
        ((_port([]),), 1, "at least one"),
        ((_port([0, 1]), _port([2, 3])), 1, "no shared"),
        ((_port([1]),), None, "singleton"),
        ((_port([0, 1]),), 0, "positive finite"),
        ((_port([0, 1]),), np.inf, "positive finite"),
        ((_port([0, 1]),), True, "positive finite"),
        ((_port([0, np.nan]),), 1, "not representable"),
        ((_port([0, np.inf]),), 1, "not representable"),
        ((_port([-1e308, 1e308], [[0], [0]]),), 1, "not representable"),
        ((_port([0, 1]),), 1e-20, "too many"),
    ],
)
def test_shared_grid_rejects_unsupported_or_unrepresentable_clocks(ports, dt, match):
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match=match):
            adapter._shared_grid(ports, dt)


@pytest.mark.parametrize(
    "times,dt",
    [
        ([], 1),
        ([0, np.nan], 1),
        ([0, np.inf], 1),
        ([0, 0], 1),
        ([1, 0], 1),
        ([0, 1, 2.1], 1),
        ([0, 1], 2),
        ([[0, 1]], 1),
        ([0j, 1j], 1),
        (1e6 + 0.01 * np.arange(12), 0.01),
    ],
)
def test_stored_grid_must_represent_the_actual_uniform_interval(times, dt):
    with pytest.raises(ValueError, match="Time grid|time grid"):
        validate_time_grid(times, dt)


def test_sample_linear_interpolation_matches_piecewise_equations_for_each_channel():
    port = _port([0, 0.5, 2, 4], [[3, 0], [4, 1], [7, -1], [11, 3]])
    queries = np.array([0.25, 1, 3])
    actual = adapter._sample(port, queries, 0.25, "linear", "forcing")
    expected = np.array([[3.5, 0.5], [5, 1 / 3], [9, 1]])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-16)


def test_sample_pchip_matches_monotone_plateau_hermite_oracle():
    # Plateau slopes are zero; the endpoint derivatives are 3/2.
    # Hermite evaluation at half intervals gives these exact rational values.
    actual = adapter._sample(
        _port([0, 1, 2, 3], [[0], [1], [1], [2]]),
        np.array([0.5, 1.5, 2.5]),
        0.5,
        "pchip",
        "target",
    )
    np.testing.assert_array_equal(actual[:, 0], [11 / 16, 1, 21 / 16])


def test_sample_cubic_reproduces_two_cubic_polynomials_between_irregular_knots():
    times = np.array([0, 0.4, 1.5, 3, 5])
    values = np.stack((times**3 - 2 * times + 5, 2 * times**3 + times**2), axis=1)
    queries = np.array([0.2, 1, 2, 4])
    actual = adapter._sample(_port(times, values), queries, 0.2, "cubic", "target")
    expected = np.stack((queries**3 - 2 * queries + 5, 2 * queries**3 + queries**2), axis=1)
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-14)


def test_sample_exact_rows_are_independent_and_empty_queries_keep_width_dtype():
    port = _port([0, 1], np.array([[1, 2], [3, 4]], dtype=np.float32))
    actual = adapter._sample(port, port.times, 1, "linear", "input")
    actual[0, 0] = 999
    assert port.values[0, 0] == 1
    assert actual.dtype == np.float32
    empty = adapter._sample(port, np.array([]), 1, "linear", "input")
    assert empty.shape == (0, 2)
    assert empty.dtype == np.float32


@pytest.mark.parametrize("origin", [0, 1e9])
def test_singleton_sample_tolerance_is_relative_to_interval_not_epoch(origin):
    port = _port([origin + 1], [[42]])
    repeated = adapter._sample(port, np.array([origin + 1, origin + 1]), 1, "linear", "driver")
    np.testing.assert_array_equal(repeated, [[42], [42]])
    with pytest.raises(ValueError, match="extrapolation"):
        adapter._sample(port, np.array([origin + 2]), 1, "linear", "driver")


def test_sample_clamps_only_tiny_endpoint_roundoff_and_rejects_missing_support():
    port = _port([0, 1])
    actual = adapter._sample(port, np.array([-1e-10, 1 + 1e-10]), 1, "linear", "driver")
    np.testing.assert_array_equal(actual[:, 0], [3, 5])
    for queries in ([-1e-8], [1 + 1e-8]):
        with pytest.raises(ValueError, match="extrapolation"):
            adapter._sample(port, np.array(queries), 1, "linear", "driver")
    with pytest.raises(ValueError, match="does not cover"):
        adapter._sample(_port([]), np.array([0]), 1, "linear", "driver")
    with pytest.raises(ValueError, match="nonfinite"):
        adapter._sample(_port([0, 1], [[0], [np.inf]]), np.array([0.5]), 1, "linear", "driver")


def test_plain_json_context_preserves_physical_leads_and_returns_independent_copy(context):
    validated = _validate(context, prediction_leads=(2, 1))
    assert validated == context
    assert validated is not context
    validated["inputs"]["state"]["source"]["metadata"]["units"] = "changed"
    assert context["inputs"]["state"]["source"]["metadata"]["units"] == "m"
    # Lead lookup follows model names rather than insertion order in JSON.
    context["supervision"]["prediction_leads"] = {"near": 1, "far": 2}
    _validate(context, prediction_leads=(2, 1))
    with pytest.raises(ValueError, match="bounds/leads"):
        _validate(context, prediction_leads=(1, 2))


@pytest.mark.parametrize(
    "path,value,match",
    [
        (("version",), 1, "version 2"),
        (("version",), True, "version 2"),
        (("clock",), [], "clock fields"),
        (("clock", "start_time"), True, "clock times"),
        (("clock", "last_time"), np.inf, "JSON-safe"),
        (("clock", "dt"), 0, "positive finite"),
        (("clock", "dt"), None, "clock times"),
        (("clock", "requested_dt"), -1, "positive finite"),
        (("clock", "method"), "nearest", "resample_method"),
        (("clock", "n_samples"), True, "count/reference"),
        (("clock", "n_samples"), 0, "count/reference"),
        (("clock", "n_samples"), 20, "times/count"),
        (("clock", "n_samples"), 10**400, "not representable"),
        (("clock", "start_time"), 21, "times/count"),
        (("clock", "reference_input"), "forcing", "count/reference"),
        (("inputs",), {}, "declared model ports"),
        (("outputs",), {"far": {}}, "declared model ports"),
        (("inputs", "state"), {}, "record"),
        (("inputs", "state", "variables"), [], "every signal"),
        (("inputs", "state", "variables"), [" "], "non-empty"),
        (("inputs", "state", "source"), None, "observation metadata"),
        (("inputs", "state", "source"), {"bad": object()}, "JSON-safe"),
        (("inputs", "state", "observation"), {}, "observation support"),
        (("inputs", "state", "observation", "n_samples"), False, "observation metadata"),
        (("inputs", "state", "observation", "n_samples"), 0, "observation metadata"),
        (("inputs", "state", "observation", "n_samples"), 1, "observation metadata"),
        (("inputs", "state", "observation", "start_time"), True, "observation metadata"),
        (("inputs", "state", "observation", "start_time"), 21, "observation metadata"),
        (("inputs", "state", "observation", "start_time"), 3, "extrapolates an input"),
        (("inputs", "state", "observation", "last_time"), 19, "extrapolates an input"),
        (("inputs", "state", "observation", "last_time"), 21, "latest common"),
        (("inputs", "state", "resampled"), 1, "observation metadata"),
        (("inputs", "state", "held_out_samples"), -1, "observation metadata"),
        (("inputs", "state", "held_out_samples"), True, "observation metadata"),
        (("supervision",), {}, "supervision metadata"),
        (("supervision", "start"), -1, "bounds/leads"),
        (("supervision", "start"), True, "bounds/leads"),
        (("supervision", "stop"), 5, "bounds/leads"),
        (("supervision", "stop"), 20, "bounds/leads"),
        (("supervision", "prediction_leads"), {"far": 2}, "bounds/leads"),
        (("supervision", "prediction_leads", "far"), 0, "bounds/leads"),
        (("supervision", "prediction_leads", "far"), True, "bounds/leads"),
        (("supervision", "prediction_leads", "far"), 3, "beyond observed"),
        (("outputs", "far", "observation", "start_time"), 10, "extrapolates or includes"),
        (("outputs", "far", "observation", "last_time"), 19, "extrapolates or includes"),
        (("outputs", "far", "observation", "last_time"), 21, "extrapolates or includes"),
    ],
)
def test_restored_context_rejects_corrupt_clock_support_and_target_labels(
    context, path, value, match
):
    destination = context
    for key in path[:-1]:
        destination = destination[key]
    destination[path[-1]] = value
    with pytest.raises(ValueError, match=match):
        _validate(context)


def test_context_rejects_extra_fields_cycles_and_unknown_optional_output(context):
    for path in ((), ("clock",), ("inputs", "state"), ("supervision",)):
        modified = copy.deepcopy(context)
        destination = modified
        for key in path:
            destination = destination[key]
        destination["unexpected"] = None
        with pytest.raises(ValueError, match="fields|record|metadata"):
            _validate(modified)
    context["inputs"]["state"]["source"]["recursive"] = context
    with pytest.raises(ValueError, match="JSON-safe"):
        _validate(context)


def test_unfitted_history_context_can_omit_output_metadata_but_not_supervised_targets(context):
    context["outputs"] = {}
    context["supervision"] = None
    _validate(context, require_outputs=False)
    with pytest.raises(ValueError, match="declared model ports"):
        _validate(context)
    context["outputs"] = {"unknown": _record(0, 20, 21, "unknown")}
    with pytest.raises(ValueError, match="declared model ports"):
        _validate(context, require_outputs=False)
    context["outputs"] = {}
    context["supervision"] = {"start": 5, "stop": 17, "prediction_leads": {"far": 2, "near": 1}}
    with pytest.raises(ValueError, match="beyond observed"):
        _validate(context, require_outputs=False)


def test_singleton_history_context_has_no_invented_origin_or_supervision(context):
    context["clock"].update(start_time=1e9, last_time=1e9, n_samples=1)
    for name in context["inputs"]:
        context["inputs"][name]["observation"] = {
            "start_time": 1e9,
            "last_time": 1e9,
            "n_samples": 1,
        }
    context["supervision"] = None
    _validate(context)
    times = adapter.trajectory_port_forecast_times(context, 2, history_length=1)
    np.testing.assert_array_equal(times, [1e9 + 1, 1e9 + 2])


def test_clock_metadata_snapshot_and_forecast_times_need_no_optional_import(monkeypatch, context):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith("tsdynamics"):
            raise AssertionError("Clock algebra must not import the optional package")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    state, forcing = _port(np.arange(21)), _port(np.arange(1.25, 20.26, 0.5))
    grid, dt = adapter._shared_grid((state, forcing), None)
    snapshot = adapter._context(
        grid,
        dt,
        None,
        "linear",
        ("state", "forcing"),
        (state, forcing),
        context["outputs"],
        context["supervision"],
    )
    _validate(snapshot, prediction_leads=(2, 1))
    state.source["metadata"]["units"] = "mutated"
    assert snapshot["inputs"]["state"]["source"]["metadata"]["units"] == "m"
    future = adapter.trajectory_port_forecast_times(snapshot, 4, history_length=19)
    np.testing.assert_array_equal(future, [21, 22, 23, 24])
    with_warmup = adapter.trajectory_port_forecast_times(
        snapshot,
        4,
        history_length=19,
        warmup_lead=2,
        return_warmup=True,
    )
    np.testing.assert_array_equal(with_warmup, np.arange(4, 25))
