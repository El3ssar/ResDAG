"""Independent interpolation, regression and physical-clock integration oracles."""

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


def _position(t):
    t = np.asarray(t)
    return t * t + 3 * t + 2


def _forcing(t):
    t = np.asarray(t)
    return 2 * t * t - t + 4


def _trajectory(ts, times, values, name):
    return ts.Trajectory(
        np.asarray(times, dtype=np.float64), np.asarray(values)[:, None], variables=(name,)
    )


class _Affine(nn.Module):
    supports_framewise = True

    def __init__(self):
        super().__init__()
        self.center = nn.Parameter(torch.zeros(1, dtype=torch.float64))
        self.register_buffer("scale", torch.ones(1, dtype=torch.float64))
        self.fail_fit = False

    @torch.no_grad()
    def fit(self, values):
        flat = values.reshape(-1, 1)
        self.center.copy_(flat.mean(0))
        self.scale.copy_(flat.std(0, unbiased=False) + 2)
        if self.fail_fit:
            raise RuntimeError("intentional staged custom fit failure")
        return self

    def forward(self, values):
        return (2 if self.training else 1) * (values - self.center) / self.scale

    def inverse(self, values):
        return values / (2 if self.training else 1) * self.scale + self.center


def _case(ts, *, custom=True):
    state, forcing = (rd.reservoir_input(1, dtype=torch.float64) for _ in range(2))
    features = rd.Concatenate()(state, forcing)
    one = rd.SVDReadoutLayer(2, 1, alpha=0.0, name="private_one").double()
    two = rd.SVDReadoutLayer(2, 1, alpha=0.0, name="private_two").double()
    incoming, outgoing = _Affine(), _Affine()
    options = (
        {
            "input_transforms": {"state": incoming},
            "output_transforms": {"future_two": outgoing},
        }
        if custom
        else {}
    )
    model = rd.Model(
        inputs={"state": state, "forcing": forcing},
        outputs={"future_two": two(features), "future_one": one(features)},
        prediction_lead={"future_two": 2, "future_one": 1},
        feedback=[rd.Feedback("future_one", "state")],
        washout=2,
        **options,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    parameter_ids = [id(parameter) for parameter in model.parameters()]
    t_state, t_driver = np.arange(21.0), np.arange(1.25, 20.26, 0.5)
    t_one, t_two = np.arange(8.0, 22.01, 0.5), np.arange(7.0, 22.01, 0.25)
    # Reverse both supplied mappings relative to the model's declared order.
    data = {
        "forcing": _trajectory(ts, t_driver, _forcing(t_driver), "forcing"),
        "state": _trajectory(ts, t_state, _position(t_state), "position"),
    }
    targets = {
        "future_one": _trajectory(ts, t_one, _position(t_one), "next_position"),
        "future_two": _trajectory(ts, t_two, _position(t_two), "later_position"),
    }
    model.fit(data, targets=targets)
    assert parameter_ids == [id(parameter) for parameter in model.parameters()]
    assert parameter_ids == [
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    ]
    if custom:
        assert model.input_scalers[0] is incoming and model.output_scalers[0] is outgoing
    return model, data, targets, (two, one), (incoming, outgoing)


@pytest.mark.parametrize("custom", [False, True])
def test_late_targets_two_leads_and_port_transforms_match_independent_lstsq_and_dates(ts, custom):
    model, data, _, heads, transforms = _case(ts, custom=custom)
    # Shared input clock is 2..20. Target supports/lead distances permit origins
    # 7..18, with physical labels through20. Washout2 is not applied again at7.
    origin = np.arange(7.0, 19.0)
    driver = data["forcing"]
    inputs = torch.from_numpy(
        np.stack((_position(origin), np.interp(origin, driver.t, driver.y[:, 0])), 1)
    )
    targets = torch.from_numpy(np.stack((_position(origin + 2), _position(origin + 1)), 1))
    normalized_inputs = (inputs - inputs.mean(0)) / inputs.std(0)
    normalized_targets = (targets - targets.mean(0)) / targets.std(0)
    if custom:
        normalized_inputs[:, 0] = (inputs[:, 0] - inputs[:, 0].mean()) / (
            inputs[:, 0].std(unbiased=False) + 2
        )
        normalized_targets[:, 0] = (targets[:, 0] - targets[:, 0].mean()) / (
            targets[:, 0].std(unbiased=False) + 2
        )
        torch.testing.assert_close(transforms[0].center, inputs.mean(0)[:1], rtol=0, atol=0)
        torch.testing.assert_close(transforms[1].center, targets.mean(0)[:1], rtol=0, atol=0)
    design = torch.cat((normalized_inputs, torch.ones(len(inputs), 1, dtype=torch.float64)), 1)
    coefficients = torch.linalg.lstsq(design, normalized_targets).solution
    for index, head in enumerate(heads):
        torch.testing.assert_close(
            head.weight, coefficients[:-1, index][None], rtol=2e-9, atol=2e-10
        )
        torch.testing.assert_close(head.bias, coefficients[-1:, index], rtol=2e-9, atol=2e-10)
    future_times = np.arange(20.25, 23.26, 0.5)
    future = _trajectory(ts, future_times, _forcing(future_times), "forcing")
    model.train()
    model.graph.eval()
    model.output_scalers[0].eval()
    modes = [module.training for module in model.modules()]
    result = model.forecast(horizon=4, future_inputs={"forcing": future})
    for name, prediction in result.items():
        np.testing.assert_array_equal(prediction.t, np.arange(21.0, 25.0))
        np.testing.assert_allclose(
            prediction.y[:, 0], _position(prediction.t), rtol=2e-10, atol=2e-7
        )
        assert prediction.variables == (
            "later_position" if name == "future_two" else "next_position",
        )
    assert [module.training for module in model.modules()] == modes
    warm = model.forecast(data, horizon=4, future_inputs={"forcing": future}, return_warmup=True)
    for prediction in warm.values():
        np.testing.assert_array_equal(prediction.t, np.arange(4.0, 25.0))
        np.testing.assert_allclose(
            prediction.y[:, 0], _position(prediction.t), rtol=2e-10, atol=2e-7
        )


def test_late_target_temporal_fit_and_cached_forecast_preserve_full_observed_prefix(ts):
    class PrefixSum(nn.Module):
        def forward(self, value):
            return value.cumsum(1)

    x = rd.reservoir_input(1, dtype=torch.float64)
    head = rd.SVDReadoutLayer(1, 1, alpha=0.0).double()
    model = rd.Model(
        inputs={"state": x},
        outputs={"future": head(PrefixSum()(x))},
        prediction_lead=2,
        washout=2,
        preprocessing="off",
    )
    times, target_times = np.arange(21.0), np.arange(8.0, 22.01)
    model.fit(
        {"state": _trajectory(ts, times, _position(times), "position")},
        targets={
            "future": _trajectory(ts, target_times, _position(target_times), "future_position")
        },
    )
    cumulative = np.cumsum(_position(times))
    origins = np.arange(6, 19)
    design = torch.from_numpy(np.stack((cumulative[origins], np.ones(len(origins))), 1))
    coefficients = torch.linalg.lstsq(
        design, torch.from_numpy(_position(origins + 2)).double()[:, None]
    ).solution[:, 0]
    torch.testing.assert_close(head.weight, coefficients[:1][None], rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(head.bias, coefficients[1:], rtol=1e-10, atol=1e-10)
    emissions = {
        int(t): float(coefficients[0]) * cumulative[int(t)] + float(coefficients[1]) for t in times
    }
    total = cumulative[-1]
    for origin in (21, 22):
        total += emissions[origin - 2]
        emissions[origin] = float(coefficients[0]) * total + float(coefficients[1])
    expected = np.array([emissions[t - 2] for t in (21, 22, 23, 24)])
    prediction = model.forecast(horizon=4, execution="history")
    np.testing.assert_array_equal(prediction.t, [21.0, 22.0, 23.0, 24.0])
    np.testing.assert_allclose(prediction.y[:, 0], expected, rtol=1e-10, atol=1e-9)


def _same(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _same(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _same(a, b)
    else:
        assert actual == expected


def test_clocked_custom_checkpoint_optimizer_identity_and_failure_rollback(ts):
    model, data, targets, _, _ = _case(ts)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    checkpoint = torch.load(stream, weights_only=True)
    restored = copy.deepcopy(model)
    optimizer = torch.optim.SGD(restored.parameters(), lr=0.01)
    identities = [id(parameter) for parameter in restored.parameters()]
    with torch.no_grad():
        for parameter in restored.graph.parameters():
            parameter.zero_()
    restored.load_state_dict(checkpoint)
    future = torch.from_numpy((_forcing(np.arange(21.0, 24.0)) + 0.125)[:, None])
    _same(
        restored.forecast(horizon=4, future_inputs={"forcing": future}, return_type="tensor"),
        model.forecast(horizon=4, future_inputs={"forcing": future}, return_type="tensor"),
    )
    restored.train()
    restored.graph.eval()
    restored.output_scalers[0].eval()
    for failure in ("target_support", "custom_fit", "future_tick", "checkpoint_clock"):
        before = copy.deepcopy(restored.state_dict())
        modes = [module.training for module in restored.modules()]
        calls = []
        handles = [
            layer.register_forward_hook(lambda *_: calls.append(None))
            for layer in restored.graph.modules()
            if isinstance(layer, rd.ReadoutLayer)
        ]
        try:
            with pytest.raises((ValueError, RuntimeError)):
                if failure == "target_support":
                    bad = {
                        name: _trajectory(ts, [30.0, 31.0], [1.0, 2.0], "bad") for name in targets
                    }
                    restored.fit(data, targets=bad)
                elif failure == "custom_fit":
                    restored.input_scalers[0].fail_fit = True
                    restored.fit(data, targets=targets)
                elif failure == "future_tick":
                    restored.forecast(
                        horizon=2,
                        future_inputs={"forcing": _trajectory(ts, [22.0], [1.0], "forcing")},
                    )
                else:
                    bad = copy.deepcopy(checkpoint)
                    context = bad["_extra_state"]["trajectory"]
                    context["supervision"]["stop"] = context["clock"]["n_samples"]
                    restored.load_state_dict(bad)
        finally:
            for handle in handles:
                handle.remove()
            restored.input_scalers[0].fail_fit = False
        _same(restored.state_dict(), before)
        assert [module.training for module in restored.modules()] == modes and calls == []
        assert identities == [id(parameter) for parameter in restored.parameters()]
        assert identities == [
            id(parameter) for group in optimizer.param_groups for parameter in group["params"]
        ]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_cached_named_custom_tensor_forecast_stays_on_native_mps_without_numpy(ts, monkeypatch):
    model, _, _, _, _ = _case(ts)
    model.to(device="mps", dtype=torch.float32)
    future = torch.from_numpy((_forcing(np.arange(21.0, 24.0)) + 0.125)[:, None]).to(
        device="mps", dtype=torch.float32
    )
    expected = torch.from_numpy(_position(np.arange(21.0, 25.0))[:, None]).float()
    original_to = torch.Tensor.to

    def forbidden(*args, **kwargs):
        raise AssertionError("Cached tensor forecasting crossed the CPU/NumPy boundary")

    def device_to(value, *args, **kwargs):
        device = kwargs.get("device")
        if args and isinstance(args[0], (str, torch.device, torch.Tensor)):
            device = args[0].device if isinstance(args[0], torch.Tensor) else args[0]
        if value.device.type == "mps" and device is not None and torch.device(device).type == "cpu":
            forbidden()
        return original_to(value, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", forbidden)
        patch.setattr(torch.Tensor, "numpy", forbidden)
        patch.setattr(torch.Tensor, "to", device_to)
        for name in ("asarray", "interp", "arange", "concatenate"):
            patch.setattr(np, name, forbidden)
        result = model.forecast(horizon=4, future_inputs={"forcing": future}, return_type="tensor")
    for value in result.values():
        assert value.device.type == "mps" and value.dtype == torch.float32
        torch.testing.assert_close(value.cpu(), expected, rtol=1e-4, atol=1e-3)
