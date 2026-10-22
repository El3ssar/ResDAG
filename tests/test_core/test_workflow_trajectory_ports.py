"""Physical-time, train-window and checkpoint oracles for named trajectories."""

import copy
import io

import numpy as np
import pytest
import torch
from torch import nn

import resdag as rd


@pytest.fixture
def ts():
    return pytest.importorskip("tsdynamics")


def _state(times):
    return times * (times + 1) / 2


def _trajectory(ts, times, function, name):
    times = np.asarray(times, dtype=np.float64)
    return ts.Trajectory(times, function(times)[:, None], variables=(name,))


def _model(**options):
    state, forcing = (rd.reservoir_input(1, dtype=torch.float64) for _ in range(2))
    features = rd.Concatenate()(state, forcing)
    one = rd.RidgeReadoutLayer(2, 1, alpha=0, name="private_near").double()(features)
    two = rd.RidgeReadoutLayer(2, 1, alpha=0, name="private_far").double()(features)
    return rd.Model(
        inputs={"state": state, "forcing": forcing},
        outputs={"near": one, "far": two},
        prediction_lead={"near": 1, "far": 2},
        feedback=[rd.Feedback("far", "state", delay=2)],
        washout=2,
        **options,
    )


def _observations(ts):
    inputs = {
        "forcing": _trajectory(ts, np.arange(0.25, 12.26, 0.5), lambda t: t + 1, "force"),
        "state": _trajectory(ts, np.arange(13), _state, "position"),
    }
    targets = {
        "far": _trajectory(ts, np.arange(5, 17), _state, "far_position"),
        "near": _trajectory(ts, np.arange(6, 17), _state, "near_position"),
    }
    return inputs, targets


def _same(left, right):
    assert left.keys() == right.keys()
    for name, expected in right.items():
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(left[name], expected, rtol=0, atol=0)
        else:
            assert left[name] == expected


def test_named_physical_targets_fit_exact_window_without_double_washout(ts):
    inputs, targets = _observations(ts)
    model = _model().fit(inputs, targets)
    assert model.washout == 2
    context = model.get_extra_state()["trajectory"]
    assert context["version"] == 2
    assert context["clock"]["dt"] == 1
    assert context["supervision"] == {
        "start": 4,
        "stop": 10,
        "prediction_leads": {"near": 1, "far": 2},
    }
    # Origins are 5..10. The earlier observed prefix only warms the graph;
    # later observed 11,12 remain available for the forecast boundary.
    expected_inputs = (_state(np.arange(5, 11)), np.arange(6, 12))
    expected_targets = (_state(np.arange(6, 12)), _state(np.arange(7, 13)))
    for scaler, values in zip(model.input_scalers, expected_inputs):
        torch.testing.assert_close(scaler._stat0, torch.tensor(values.mean()).reshape(1, 1, 1))
    for scaler, values in zip(model.output_scalers, expected_targets):
        torch.testing.assert_close(scaler._stat0, torch.tensor(values.mean()).reshape(1, 1, 1))
    np.testing.assert_array_equal(model._history_0[:, 0].numpy(), _state(np.arange(9, 13)))
    np.testing.assert_array_equal(model._history_1[:, 0].numpy(), np.arange(10, 14))
    driver = _trajectory(ts, [13, 13.25, 14.5, 15], lambda t: t + 1, "force")
    result = model.forecast(horizon=4, future_inputs={"forcing": driver})
    assert list(result) == ["near", "far"]
    for name, value in result.items():
        assert value.variables == (f"{name}_position",)
        np.testing.assert_array_equal(value.t, [13, 14, 15, 16])
        np.testing.assert_allclose(value.y[:, 0], _state(value.t), atol=1e-10, rtol=0)
    tensors = model.forecast(horizon=4, future_inputs={"forcing": driver}, return_type="tensor")
    for name in model.output_names:
        torch.testing.assert_close(tensors[name], torch.from_numpy(result[name].y), rtol=0, atol=0)


def test_named_warmup_heads_share_physical_grid_and_singleton_driver(ts):
    inputs, targets = _observations(ts)
    model = _model().fit(inputs, targets)
    driver = _trajectory(ts, [13], lambda t: t + 1, "force")
    result = model.forecast(
        inputs, horizon=2, future_inputs={"forcing": driver}, return_warmup=True
    )
    for value in result.values():
        np.testing.assert_array_equal(value.t, np.arange(3, 15))
        np.testing.assert_allclose(value.y[:, 0], _state(value.t), atol=1e-10, rtol=0)
    first = model.forecast(horizon=1)
    for value in first.values():
        np.testing.assert_array_equal(value.t, [13])
        np.testing.assert_allclose(value.y[:, 0], [_state(13)], atol=1e-10, rtol=0)


def test_named_matches_independent_aligned_tensor_fit_and_manual_forecast(ts):
    inputs, targets = _observations(ts)
    named = _model().fit(inputs, targets)
    # Fit the independently derived regression origin window 5..10. This
    # graph is affine, so warmup has no effect on its algebraic oracle.
    time = torch.arange(5, 11, dtype=torch.float64)[:, None]
    tensor = _model()
    tensor.washout = 0
    tensor.fit(
        {"state": time * (time + 1) / 2, "forcing": time + 1},
        {"near": (time + 1) * (time + 2) / 2, "far": (time + 2) * (time + 3) / 2},
    )
    for left, right in zip(named.graph.parameters(), tensor.graph.parameters()):
        torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-12)
    history = torch.arange(1, 13, dtype=torch.float64)[:, None]
    driver = torch.arange(14, 17, dtype=torch.float64)[:, None]
    expected = tensor.forecast(
        {"state": history * (history + 1) / 2, "forcing": history + 1},
        horizon=4,
        future_inputs={"forcing": driver},
    )
    actual = named.forecast(horizon=4, future_inputs={"forcing": driver}, return_type="tensor")
    for name in named.output_names:
        torch.testing.assert_close(actual[name], expected[name], atol=1e-10, rtol=1e-12)


@pytest.mark.parametrize("method", ["pchip", "cubic"])
def test_future_target_suffix_cannot_change_fitted_weights_or_transforms(ts, method):
    inputs, targets = _observations(ts)
    baseline = _model(resample_method=method).fit(inputs, targets)
    changed = {}
    for name, target in targets.items():
        values = target.y.copy()
        values[target.t > 12] = np.nan
        changed[name] = ts.Trajectory(target.t, values, variables=target.variables)
    perturbed = _model(resample_method=method).fit(inputs, changed)
    _same(perturbed.state_dict(), baseline.state_dict())


def test_named_context_checkpoint_roundtrip_and_float_cast(ts):
    inputs, targets = _observations(ts)
    model = _model().fit(inputs, targets)
    driver = _trajectory(ts, [13], lambda t: t + 1, "force")
    expected = model.forecast(horizon=2, future_inputs={"forcing": driver})
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    state = torch.load(stream, weights_only=True)
    restored = _model()
    restored.load_state_dict(state)
    actual = restored.forecast(horizon=2, future_inputs={"forcing": driver})
    for name in model.output_names:
        np.testing.assert_array_equal(actual[name].t, expected[name].t)
        np.testing.assert_array_equal(actual[name].y, expected[name].y)
    restored.float()
    actual = restored.forecast(horizon=2, future_inputs={"forcing": driver})
    for name in model.output_names:
        np.testing.assert_array_equal(actual[name].t, expected[name].t)
        np.testing.assert_allclose(actual[name].y, expected[name].y, rtol=1e-5)


def test_late_target_custom_stats_use_actual_labels_and_absolute_washout(ts):
    inputs, targets = _observations(ts)
    incoming, outgoing = rd.Standardize(1).double(), rd.Standardize(1).double()
    model = _model(input_transforms={"state": incoming}, output_transforms={"near": outgoing}).fit(
        inputs, targets
    )
    torch.testing.assert_close(incoming.mean, torch.tensor([_state(np.arange(5, 11)).mean()]))
    torch.testing.assert_close(outgoing.mean, torch.tensor([_state(np.arange(6, 12)).mean()]))
    assert model.input_scalers[0] is incoming and model.output_scalers[0] is outgoing


def test_tensor_history_cannot_supply_clock_for_trajectory_driver(ts):
    inputs, targets = _observations(ts)
    model = _model().fit(inputs, targets)
    state = model._history_0.clone()
    forcing = model._history_1.clone()
    driver = _trajectory(ts, [13], lambda t: t + 1, "force")
    calls = []
    model.graph.register_forward_pre_hook(lambda *args: calls.append(True))
    with pytest.raises(ValueError, match="new Tensor history has no absolute clock"):
        model.forecast(
            {"state": state, "forcing": forcing},
            horizon=2,
            future_inputs={"forcing": driver},
            return_type="tensor",
        )
    assert not calls


@pytest.mark.parametrize("failure", ["mixed", "no_support", "wrong_names", "bad_driver"])
def test_named_failed_fit_or_forecast_preserves_live_state_and_metadata(ts, failure):
    inputs, targets = _observations(ts)
    model = _model().fit(inputs, targets)
    before = copy.deepcopy(model.state_dict())
    calls = []
    model.graph.register_forward_pre_hook(lambda *args: calls.append(True))
    if failure == "mixed":
        bad = dict(inputs, forcing=torch.ones(12, 1).double())
        with pytest.raises(TypeError, match="mixed Tensor/Trajectory"):
            model.fit(bad, targets)
    elif failure == "no_support":
        bad = dict(targets, near=_trajectory(ts, np.arange(20, 25), _state, "near_position"))
        with pytest.raises(ValueError, match="no jointly supported"):
            model.fit(inputs, bad)
    elif failure == "wrong_names":
        with pytest.raises(ValueError, match="components/order"):
            model.forecast(
                {
                    "state": _trajectory(ts, np.arange(13), _state, "wrong"),
                    "forcing": inputs["forcing"],
                },
                horizon=1,
            )
    else:
        with pytest.raises(ValueError, match="extrapolation"):
            model.forecast(
                horizon=2,
                future_inputs={"forcing": _trajectory(ts, [14], lambda t: t + 1, "force")},
            )
    assert not calls
    _same(model.state_dict(), before)


def test_named_recurrent_warmup_uses_full_observed_prefix_before_supervision(ts):
    x = rd.reservoir_input(1, dtype=torch.float64)
    reservoir = rd.ESNLayer(6, 1, seed=2).double()
    output = rd.RidgeReadoutLayer(6, 1, alpha=1e-4).double()(reservoir(x))
    model = rd.Model(inputs={"state": x}, outputs={"next": output}, washout=2)
    clock = np.arange(20, dtype=np.float64)
    inputs = _trajectory(ts, clock, np.sin, "position")
    target = _trajectory(ts, clock[8:], np.cos, "target")
    model.fit({"state": inputs}, {"next": target})
    # Independently run normalized observed origin prefix0..18, then solve
    # using origins7..18 against labels at clocks8..19.
    scaler = model.input_scalers[0]
    normalized = scaler(torch.from_numpy(inputs.y).unsqueeze(0))
    reservoir.reset_state()
    with torch.no_grad():
        features = reservoir(normalized[:, :19])[:, 7:]
    expected = rd.RidgeReadoutLayer(6, 1, alpha=1e-4).double()
    labels = model.output_scalers[0](torch.from_numpy(target.y).unsqueeze(0))
    expected.fit(features, labels)
    actual = model.graph.outputs[0].layer
    torch.testing.assert_close(actual.weight, expected.weight, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(actual.bias, expected.bias, rtol=1e-10, atol=1e-12)
    assert model._history_0.shape == (3, 1)


class Cumulative(nn.Module):
    supports_framewise = False

    def forward(self, values):
        return values.cumsum(dim=1)


def test_temporal_graph_keeps_complete_observations_after_supervision(ts):
    x = rd.reservoir_input(1, dtype=torch.float64)
    y = rd.RidgeReadoutLayer(1, 1, alpha=0).double()(Cumulative()(x))
    model = rd.Model(
        inputs={"driver": x},
        outputs={"total": y},
        washout=2,
        preprocessing="off",
        feedback=(),
    )
    observed = _trajectory(ts, np.arange(21), lambda t: t + 1, "sample")
    target = _trajectory(ts, np.arange(8, 20), _state, "total")
    model.fit({"driver": observed}, {"total": target})
    assert model._history_0.shape == (21, 1)
    future = _trajectory(ts, [21, 22], lambda t: t + 1, "sample")
    result = model.forecast(
        horizon=3, future_inputs={"driver": future}, execution="history", return_warmup=True
    )
    np.testing.assert_array_equal(result.t, np.arange(1, 24))
    np.testing.assert_allclose(result.y[:, 0], _state(result.t), atol=1e-10)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_named_mps_tensor_forecast_stays_native_and_trajectory_returns_shared_clock(
    ts, monkeypatch
):
    inputs, targets = _observations(ts)
    model = _model().fit(inputs, targets).float().to("mps")
    future = _trajectory(ts, [13], lambda t: t + 1, "force")
    with monkeypatch.context() as patched:

        def forbidden(*args, **kwargs):
            raise AssertionError("tensor result moved to CPU/NumPy")

        patched.setattr(torch.Tensor, "cpu", forbidden)
        patched.setattr(torch.Tensor, "numpy", forbidden)
        tensors = model.forecast(horizon=2, future_inputs={"forcing": future}, return_type="tensor")
    for value in tensors.values():
        assert value.device.type == "mps" and value.dtype == torch.float32
        torch.testing.assert_close(
            value.cpu(), torch.tensor([[91.0], [105.0]]), rtol=1e-5, atol=1e-5
        )
    result = model.forecast(horizon=2, future_inputs={"forcing": future})
    for value in result.values():
        np.testing.assert_array_equal(value.t, [13, 14])
        np.testing.assert_allclose(value.y[:, 0], [91, 105], atol=1e-4)
