"""Independent physical-time, named-route and normalization feedback oracles."""

import copy
import io

import numpy as np
import pytest
import torch
from torch import nn

from resdag import ESNLayer, Model, RidgeReadoutLayer, reservoir_input
from resdag.core._feedback import Feedback
from resdag.ensemble import CoupledEnsembleESNModel
from resdag.layers import Concatenate
from resdag.models import classic_esn, parallel_esn


def _ramp(*, lead=1, delay=None, preprocessing="standard", device="cpu"):
    dtype = torch.float32 if device == "mps" else torch.float64
    x = reservoir_input(1, dtype=dtype, device=device)
    y = RidgeReadoutLayer(
        1,
        1,
        alpha=0.0,
        name="private_head",
        solver="solve" if device == "mps" else "cholesky",
        use_float64=device != "mps",
        gram_dtype=torch.float32 if device == "mps" else None,
    ).to(device=device, dtype=dtype)(x)
    return Model(
        inputs={"state": x},
        outputs={"next": y},
        washout=0,
        preprocessing=preprocessing,
        prediction_lead=lead,
        feedback=None if delay is None else [Feedback("next", "state", delay)],
    )


@pytest.mark.parametrize("lead,delay", [(1, 1), (2, 2), (3, 3)])
@pytest.mark.parametrize("horizon", [1, 2, 5])
def test_matching_prediction_lead_and_route_delay_start_at_immediate_unseen_sample(
    lead, delay, horizon
):
    observed = torch.arange(8, dtype=torch.float64)[:, None]
    model = _ramp(lead=lead, delay=delay).fit(observed)
    expected = torch.arange(8, 8 + horizon, dtype=torch.float64)[:, None]
    torch.testing.assert_close(model.forecast(horizon=horizon), expected)
    torch.testing.assert_close(model.forecast(observed, horizon=horizon), expected)
    with_warmup = model.forecast(observed, horizon=horizon, return_warmup=True)
    torch.testing.assert_close(
        with_warmup, torch.arange(lead, 8 + horizon, dtype=torch.float64)[:, None]
    )


def test_two_step_head_seeds_both_pending_warmup_predictions_and_auto_delay_matches_lead():
    observed = torch.arange(5, dtype=torch.float64)[:, None]
    model = _ramp(lead=2).fit(observed)
    assert model.prediction_leads == {"next": 2} and model.feedback is None
    torch.testing.assert_close(
        model.forecast(observed, horizon=4), torch.tensor([[5.0], [6.0], [7.0], [8.0]]).double()
    )
    # The input fit window is 0,1,2; the final two observations are targets,
    # not leaked into the fitted input scaling.
    torch.testing.assert_close(model.input_scalers[0]._stat0, torch.tensor([[[1.0]]]).double())


def test_one_step_prediction_with_two_step_connection_is_deliberately_lagged():
    observed = torch.arange(5, dtype=torch.float64)[:, None]
    model = _ramp(lead=1, delay=2).fit(observed)
    torch.testing.assert_close(
        model.forecast(observed, horizon=4), torch.tensor([[5.0], [5.0], [6.0], [6.0]]).double()
    )


@pytest.mark.parametrize("method", ["standard", "minmax", "noncentered", "meanpreserving", "off"])
def test_independently_fitted_target_scaling_is_bridged_in_physical_units(method):
    observed = torch.arange(8, dtype=torch.float64)[:, None]
    model = _ramp(preprocessing=method).fit(observed, targets=observed + 1)
    torch.testing.assert_close(
        model.forecast(observed, horizon=5), torch.arange(8, 13, dtype=torch.float64)[:, None]
    )


def _driven_two_leads():
    state, driver = (reservoir_input(1, dtype=torch.float64) for _ in range(2))
    features = Concatenate()(state, driver)
    one = RidgeReadoutLayer(2, 1, alpha=0.0, name="private_one").double()(features)
    two = RidgeReadoutLayer(2, 1, alpha=0.0, name="private_two").double()(features)
    return Model(
        inputs={"forcing": driver, "state": state},
        outputs={"two_steps": two, "one_step": one},
        prediction_lead={"two_steps": 2},
        feedback=[Feedback("two_steps", "state", 2)],
        washout=0,
    )


def test_named_nonfirst_feedback_multihead_leads_and_drivers_share_one_physical_clock():
    model = _driven_two_leads()
    clock = torch.arange(12, dtype=torch.float64)[:, None]
    state, forcing = clock * (clock + 1) / 2, clock + 1
    # x[t+1]=x[t]+d[t], d[t]=t+1; x[t+2]=x[t]+2*d[t]+1.
    model.fit(
        {"state": state, "forcing": forcing},
        {"one_step": state + forcing, "two_steps": state + 2 * forcing + 1},
    )
    history = {"state": state[:4], "forcing": forcing[:4]}
    expected = torch.tensor([[10.0], [15.0], [21.0], [28.0]]).double()
    result = model.forecast(history, horizon=4, future_inputs={"forcing": forcing[4:7]})
    assert list(result) == ["two_steps", "one_step"]
    for value in result.values():
        torch.testing.assert_close(value, expected)
    longer = model.forecast(
        history, horizon=4, future_inputs={"forcing": forcing[4:7]}, return_warmup=True
    )
    for value in longer.values():
        torch.testing.assert_close(value, torch.cat((state[2:4], expected)))
    first = model.forecast(history, horizon=1)
    for value in first.values():
        torch.testing.assert_close(value, expected[:1])


def test_empty_feedback_routes_leave_all_inputs_named_external_drivers():
    observed = torch.arange(8, dtype=torch.float64)[:, None]
    x = reservoir_input(1, dtype=torch.float64)
    y = RidgeReadoutLayer(1, 1, alpha=0.0).double()(x)
    model = Model(inputs={"state": x}, outputs={"next": y}, feedback=(), washout=0).fit(observed)
    result = model.forecast(
        observed[:5],
        horizon=4,
        future_inputs={"state": torch.tensor([[20.0], [30.0], [40.0]]).double()},
    )
    torch.testing.assert_close(result, torch.tensor([[5.0], [21.0], [31.0], [41.0]]).double())


def test_one_output_can_feed_multiple_independently_scaled_inputs():
    a, b = (reservoir_input(1, dtype=torch.float64) for _ in range(2))
    y = RidgeReadoutLayer(2, 1, alpha=0.0).double()(Concatenate()(a, b))
    model = Model(
        inputs={"a": a, "b": b},
        outputs={"next": y},
        feedback=[Feedback("next", "a"), Feedback("next", "b")],
        washout=0,
    )
    train_a = torch.tensor([[-10.0], [-5.0], [0.0], [5.0], [10.0]]).double()
    train_b = torch.tensor([[3.0], [-7.0], [2.0], [8.0], [-3.0]]).double()
    model.fit({"b": train_b, "a": train_a}, targets=(train_a + train_b) / 2 + 1)
    history = torch.arange(5, dtype=torch.float64)[:, None]
    torch.testing.assert_close(
        model.forecast({"b": history, "a": history}, horizon=4),
        torch.arange(5, 9, dtype=torch.float64)[:, None],
    )


def test_cross_connected_heads_consume_delayed_emissions_from_the_same_graph_clock():
    fast, slow = (reservoir_input(1, dtype=torch.float64) for _ in range(2))
    fast_y = RidgeReadoutLayer(1, 1, alpha=0.0, name="fast_head").double()(fast)
    slow_y = RidgeReadoutLayer(1, 1, alpha=0.0, name="slow_head").double()(slow)
    model = Model(
        inputs={"slow": slow, "fast": fast},
        outputs={"slow_next": slow_y, "fast_next": fast_y},
        feedback=[Feedback("slow_next", "fast", 1), Feedback("fast_next", "slow", 2)],
        washout=0,
    )
    training = torch.arange(8, dtype=torch.float64)[:, None]
    model.fit(
        {"fast": training, "slow": training * 10},
        {"fast_next": training + 1, "slow_next": training * 10 + 10},
    )
    actual = model.forecast(
        {
            "fast": torch.tensor([[0.0], [1.0], [2.0]]).double(),
            "slow": torch.tensor([[10.0], [20.0], [30.0]]).double(),
        },
        horizon=4,
    )
    torch.testing.assert_close(
        actual["fast_next"], torch.tensor([[3.0], [41.0], [13.0], [14.0]]).double()
    )
    torch.testing.assert_close(
        actual["slow_next"], torch.tensor([[40.0], [12.0], [13.0], [51.0]]).double()
    )


@pytest.mark.parametrize("family", ["symbolic", "parallel", "coupled"])
def test_pointwise_history_fallback_matches_fast_values_states_and_allows_new_forward_batch(family):
    if family == "parallel":
        graph = parallel_esn(
            4, core_width=2, halo=1, reservoir_size=5, seed=31, dtype=torch.float64
        )
        width = 4
    else:
        members = [classic_esn(5, 1, 1, seed=seed, dtype=torch.float64) for seed in (31, 32)]
        graph = members[0] if family == "symbolic" else CoupledEnsembleESNModel(members)
        width = 1
    training = torch.randn(
        1, 24, width, generator=torch.Generator().manual_seed(71), dtype=torch.float64
    )
    model = Model(graph=graph, washout=2).fit(training).eval()
    reference = copy.deepcopy(model)
    history = training[:, -6:]
    expected = reference.forecast(history, horizon=5, return_warmup=True)
    with torch.inference_mode():
        actual = model.forecast(history, horizon=5, return_warmup=True, execution="history")
    torch.testing.assert_close(actual, expected)
    for name, value in model.get_state().items():
        torch.testing.assert_close(value, reference.get_state()[name])
        assert not value.is_inference()
    # A forecast is an evolved state, not a user-pinned set_state restore.
    source = torch.ones(2, 2, width, dtype=torch.float64, requires_grad=True)
    model(source).sum().backward()
    assert source.grad is not None and bool(torch.isfinite(source.grad).all())


def test_failed_temporal_feature_capture_preserves_a_previously_fitted_model():
    class Temporal(nn.Module):
        drop_prefix = False

        def forward(self, value):
            return value[:, 1:] if self.drop_prefix else value.cumsum(1)

    x = reservoir_input(1, dtype=torch.float64)
    temporal = Temporal()
    head = RidgeReadoutLayer(1, 1, alpha=0.0).double()
    model = Model(inputs={"x": x}, outputs={"y": head(temporal(x))}, washout=2)
    training = torch.arange(1, 9, dtype=torch.float64)[:, None]
    model.fit(training, targets=training.cumsum(0))
    temporal.drop_prefix = True
    before = copy.deepcopy(model.state_dict())
    scalers = model.input_scalers, model.output_scalers
    with pytest.raises(ValueError, match="complete input batch/time"):
        model.fit(training + 2, targets=training.cumsum(0) + 5)
    assert model.input_scalers is scalers[0] and model.output_scalers is scalers[1]
    for name, value in before.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize(
    "options,error",
    [
        ({"prediction_lead": 0}, "positive integer"),
        ({"prediction_lead": True}, "positive integer"),
        ({"prediction_lead": {"missing": 2}}, "unknown output"),
        ({"feedback": [Feedback("missing", "state")]}, "Unknown feedback output"),
        ({"feedback": [Feedback("next", "missing")]}, "Unknown feedback input"),
        ({"feedback": [Feedback("next", "state"), Feedback("next", "state", 2)]}, "more than one"),
    ],
)
def test_invalid_leads_and_named_routes_are_rejected_at_construction(options, error):
    x = reservoir_input(1, dtype=torch.float64)
    y = RidgeReadoutLayer(1, 1).double()(x)
    with pytest.raises(ValueError, match=error):
        Model(inputs={"state": x}, outputs={"next": y}, **options)


@pytest.mark.parametrize("delay", [0, -1, True, 1.5])
def test_feedback_delay_is_a_positive_integer_graph_distance(delay):
    with pytest.raises(ValueError, match="feedback delay must be a positive integer"):
        Feedback("out", "in", delay)


def test_explicit_route_widths_are_checked_without_executing_graph_hooks():
    x = reservoir_input(1, dtype=torch.float64)
    head = RidgeReadoutLayer(1, 2).double()
    y = head(x)
    calls = []
    handle = head.register_forward_hook(lambda *_: calls.append(None))
    try:
        with pytest.raises(ValueError, match="width 2.*width 1"):
            Model(inputs={"state": x}, outputs={"next": y}, feedback=[Feedback("next", "state")])
    finally:
        handle.remove()
    assert calls == []


@pytest.mark.parametrize(
    "failure", ["short_history", "missing_driver", "short_driver", "driver_keys"]
)
def test_forecast_preflight_preserves_states_modes_and_hooks(failure):
    state, driver = (reservoir_input(1, dtype=torch.float64) for _ in range(2))
    reservoir = ESNLayer(3, feedback_size=1, input_size=1, dtype=torch.float64, seed=71)
    head = RidgeReadoutLayer(3, 1, alpha=1e-3).double()
    model = Model(
        inputs={"forcing": driver, "state": state},
        outputs={"next": head(reservoir(state, driver))},
        feedback=[Feedback("next", "state", 2)],
        prediction_lead=2,
        washout=0,
    )
    values = torch.arange(8, dtype=torch.float64)[:, None]
    model.fit({"forcing": values, "state": values}, targets=values + 2)
    model({"forcing": values, "state": values})
    before = model.get_state()
    model.train()
    head.eval()
    modes = [module.training for module in model.modules()]
    calls = []
    handle = head.register_forward_hook(lambda *_: calls.append(None))
    history = {"forcing": values[:3], "state": values[:3]}
    options = {"horizon": 5, "future_inputs": {"forcing": values[3:6]}}
    if failure == "short_history":
        history = {name: value[:1] for name, value in history.items()}
    elif failure == "missing_driver":
        options.pop("future_inputs")
    elif failure == "short_driver":
        options["future_inputs"] = {"forcing": values[:2]}
    else:
        options["future_inputs"] = {"state": values[:3]}
    try:
        with pytest.raises(ValueError):
            model.forecast(history, **options)
    finally:
        handle.remove()
    assert calls == [] and [module.training for module in model.modules()] == modes
    for name, value in before.items():
        torch.testing.assert_close(model.get_state()[name], value, rtol=0, atol=0)


def test_leads_and_connections_restore_as_safe_checkpoint_metadata():
    observed = torch.arange(8, dtype=torch.float64)[:, None]
    original = _ramp(lead=2, delay=2).fit(observed)
    stream = io.BytesIO()
    torch.save(original.state_dict(), stream)
    stream.seek(0)
    restored = _ramp().eval()
    restored.load_state_dict(torch.load(stream, weights_only=True))
    assert restored.prediction_leads == {"next": 2}
    assert restored.feedback == (Feedback("next", "state", 2),)
    torch.testing.assert_close(restored.forecast(horizon=5), original.forecast(horizon=5))
    invalid = copy.deepcopy(original.state_dict())
    invalid["_extra_state"]["feedback"][0]["input"] = "missing"
    before = copy.deepcopy(restored.state_dict())
    with pytest.raises(RuntimeError, match="Invalid Model checkpoint feedback"):
        restored.load_state_dict(invalid)
    for name, value in before.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)


def test_history_execution_replays_a_sequence_module_without_discarding_older_inputs():
    class PrefixSum(nn.Module):
        def forward(self, value):
            return value.cumsum(1)

    x = reservoir_input(1, dtype=torch.float64)
    y = RidgeReadoutLayer(1, 1, alpha=0.0).double()(PrefixSum()(x))
    model = Model(inputs={"state": x}, outputs={"sum": y}, washout=0, preprocessing="off")
    training = torch.ones(6, 1, dtype=torch.float64)
    model.fit(training, targets=training.cumsum(0))
    history = torch.tensor([[1.0], [2.0], [3.0]]).double()
    torch.testing.assert_close(
        model.forecast(history, horizon=3, execution="history"),
        torch.tensor([[6.0], [12.0], [24.0]]).double(),
    )


def test_two_step_trajectory_forecasts_keep_immediate_future_and_warmup_physical_dates():
    ts = pytest.importorskip("tsdynamics")
    observed = ts.Trajectory(
        10 + 0.25 * np.arange(5), np.arange(5.0)[:, None], variables=("position",)
    )
    model = _ramp(lead=2).fit(observed)
    result = model.forecast(observed, horizon=4, return_warmup=True)
    np.testing.assert_allclose(result.y, np.arange(2.0, 9.0)[:, None], atol=1e-11)
    np.testing.assert_allclose(result.t, 10 + 0.25 * np.arange(2, 9), atol=1e-12)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_delayed_feedback_and_scaling_bridge_keep_tensor_math_on_native_mps(monkeypatch):
    observed = torch.arange(8, dtype=torch.float32, device="mps")[:, None]
    model = _ramp(lead=2, delay=2, device="mps").fit(observed, targets=observed + 2)

    def unexpected(*args, **kwargs):
        raise AssertionError("Tensor forecast must not copy to CPU or NumPy")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", unexpected)
        patch.setattr(torch.Tensor, "numpy", unexpected)
        result = model.forecast(observed, horizon=5, return_type="tensor")
    assert result.device.type == "mps" and result.dtype == torch.float32
    torch.testing.assert_close(
        result, torch.arange(8, 13, dtype=torch.float32, device="mps")[:, None]
    )
