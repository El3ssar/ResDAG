"""Paper-equation oracles and integration tests for unsupervised ESN calibration."""

import copy
import io

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from resdag import ESNLayer, ESNModel, Input, RidgeReadoutLayer
from resdag.training import ESNTrainer, calibrate_flow_control, calibrate_intrinsic_plasticity


def _layer(**kwargs):
    options = dict(
        reservoir_size=3,
        feedback_size=2,
        spectral_radius=0.4,
        seed=42,
        bias_scaling=0.0,
        dtype=torch.float64,
    )
    return ESNLayer(**(options | kwargs))


def _data(batch=2, length=30, features=2):
    return torch.randn(
        batch, length, features, generator=torch.Generator().manual_seed(6), dtype=torch.float64
    )


def _run(layer, *inputs, state=None):
    if state is None:
        state = layer.cell.init_state(inputs[0].shape[0], inputs[0].device, inputs[0].dtype)
    return layer.forward_stateless(list(inputs), state)


def _manual_ip(layer, x, *, rate, mean, std, warmup=0):
    """Direct scalar-indexed transcription of paper eqs 10 and 11."""
    n = layer.reservoir_size
    a = torch.ones(n, dtype=x.dtype)
    b = torch.zeros(n, dtype=x.dtype)
    h = torch.zeros(x.shape[0], n, dtype=x.dtype)
    for time in range(x.shape[1]):
        raw = F.linear(x[:, time], layer.weight_feedback) + F.linear(h, layer.weight_hh)
        if layer.bias_h is not None:
            raw += layer.bias_h
        y = torch.tanh(a * raw + b)
        h = layer.cell._apply_leak(h, y)
        if time < warmup:
            continue
        delta_b = torch.empty_like(y)
        for neuron in range(n):
            for item in range(x.shape[0]):
                output = y[item, neuron]
                delta_b[item, neuron] = -rate * (
                    -mean / std**2 + output / std**2 * (2 * std**2 + 1 - output**2 + mean * output)
                )
        next_a = a + rate / a + (delta_b * raw).mean(0)
        b = b + delta_b.mean(0)
        a = next_a.clamp(1e-3, 100)
    return a, b


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("warmup", [0, 2])
def test_ip_matches_independent_equation_oracle(batch, warmup):
    layer = _layer(leak_rate=[0.3, 0.5, 1.0])
    x = _data(batch=batch, length=5)
    expected_a, expected_b = _manual_ip(layer, x, rate=2e-3, mean=0.1, std=0.3, warmup=warmup)
    report = calibrate_intrinsic_plasticity(
        layer, x, learning_rate=2e-3, target_mean=0.1, target_std=0.3, warmup=warmup
    )
    torch.testing.assert_close(layer.cell._intrinsic_gain, expected_a, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(layer.cell._intrinsic_bias, expected_b, atol=1e-14, rtol=1e-14)
    assert report.steps == 5 - warmup
    assert report.samples == batch * (5 - warmup)


@pytest.mark.parametrize("mode", ["local", "global"])
def test_flow_matches_independent_equation_oracle(mode):
    layer = _layer()
    x = _data(length=6)
    gain = torch.ones(3, dtype=x.dtype)
    state = torch.zeros(2, 3, dtype=x.dtype)
    for time in range(x.shape[1]):
        recurrent = F.linear(state, layer.weight_hh) * gain
        output = torch.tanh(F.linear(x[:, time], layer.weight_feedback) + layer.bias_h + recurrent)
        delta = (0.8**2 * state**2 - recurrent**2).mean(0)
        if mode == "global":
            delta = delta.mean()
        gain = (gain * (1 + 0.03 * delta)).clamp(1e-3, 100)
        state = output
    report = calibrate_flow_control(layer, x, target_radius=0.8, learning_rate=0.03, mode=mode)
    torch.testing.assert_close(layer.cell._recurrent_gain, gain, atol=1e-14, rtol=1e-14)
    measured = torch.linalg.eigvals(gain[:, None] * layer.weight_hh).abs().max().item()
    assert report.spectral_radius_after == pytest.approx(measured, rel=1e-12)


@pytest.mark.parametrize("calibrator", [calibrate_intrinsic_plasticity, calibrate_flow_control])
def test_chunk_projection_state_dtype_and_gradient_paths(calibrator):
    layer = _layer(input_size=1)
    fb, driver = _data(), _data(features=1)
    original = {k: v.clone() for k, v in layer.named_parameters()}
    layer.train()
    layer(fb[:, :3], driver[:, :3])
    report = calibrator(layer, fb, driver=driver, warmup=3, epochs=2)
    assert layer.training and layer.state is None
    assert report.steps == 54
    assert report.activity_std_after > 0
    for key, value in layer.named_parameters():
        assert torch.equal(value, original[key])
        assert not value.requires_grad
    inputs = [fb[:, 0], driver[:, 0]]
    initial = layer.cell.init_state(2, fb.device, fb.dtype)
    slow, _ = layer.cell(inputs, initial)
    fast, _ = layer.cell.step(layer.cell.project_inputs(inputs), initial)
    torch.testing.assert_close(slow, fast)
    whole, end = _run(layer, fb, driver)
    first, middle = _run(layer, fb[:, :11], driver[:, :11])
    second, end_chunked = _run(layer, fb[:, 11:], driver[:, 11:], state=middle)
    torch.testing.assert_close(whole, torch.cat([first, second], 1))
    torch.testing.assert_close(end, end_chunked)
    train_fb = fb.clone().requires_grad_()
    result, _ = _run(layer, train_fb, driver)
    result.square().sum().backward()
    assert train_fb.grad is not None and torch.isfinite(train_fb.grad).all()
    assert train_fb.grad.abs().sum() > 0
    for value in layer.cell.buffers():
        assert value.dtype == torch.float64


def test_ip_reduces_excess_activity_and_recenters_biased_training_data():
    layer = _layer(reservoir_size=10, bias_scaling=1.0)
    x = _data(length=500)
    report = calibrate_intrinsic_plasticity(layer, x, target_std=0.2, epochs=3, warmup=20)
    assert abs(report.activity_std_after - 0.2) < abs(report.activity_std_before - 0.2) * 0.25
    assert abs(report.activity_mean_after) < abs(report.activity_mean_before)


def test_flow_moves_small_random_reservoir_towards_target_under_independent_inputs():
    layer = _layer(
        reservoir_size=12,
        feedback_size=12,
        spectral_radius=0.3,
        bias=False,
        feedback_initializer=(torch.nn.init.eye_, {}),
    )
    x = _data(batch=4, length=1200, features=12) * 0.4
    report = calibrate_flow_control(layer, x, target_radius=0.8, learning_rate=0.03, epochs=2)
    assert abs(report.spectral_radius_after - 0.8) < abs(report.spectral_radius_before - 0.8) * 0.4


@pytest.mark.parametrize("calibrator", [calibrate_intrinsic_plasticity, calibrate_flow_control])
def test_unbatched_tuple_driver_and_reproducibility(calibrator):
    a, b = _layer(input_size=1, noise=0.5), _layer(input_size=1, noise=0.5)
    a.train()
    b.eval()
    x, u = _data(batch=1).squeeze(0), _data(batch=1, features=1).squeeze(0)
    state = torch.random.get_rng_state().clone()
    ra = calibrator(a, (x, u), epochs=2)
    rb = calibrator(b, x.unsqueeze(0), driver=u.unsqueeze(0), epochs=2)
    assert ra == rb
    assert torch.equal(torch.random.get_rng_state(), state)
    assert a.training and not b.training
    for key, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[key])


@pytest.mark.parametrize("calibrator", [calibrate_intrinsic_plasticity, calibrate_flow_control])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"warmup": -1},
        {"warmup": True},
        {"warmup": 30},
        {"epochs": 0},
        {"epochs": 1.5},
        {"learning_rate": 0},
        {"learning_rate": float("nan")},
        {"min_gain": 0},
        {"max_gain": float("inf")},
        {"min_gain": 5, "max_gain": 1},
    ],
)
def test_invalid_options_leave_state_weights_and_mode_unchanged(calibrator, kwargs):
    layer = _layer()
    x = _data()
    layer.eval()
    layer(x[:, :3])
    state = layer.state.clone()
    saved = copy.deepcopy(layer.state_dict())
    with pytest.raises((ValueError, TypeError)):
        calibrator(layer, x, **kwargs)
    assert not layer.training
    assert torch.equal(layer.state, state)
    assert saved.keys() == layer.state_dict().keys()
    assert all(torch.equal(value, saved[key]) for key, value in layer.state_dict().items())


@pytest.mark.parametrize("calibrator", [calibrate_intrinsic_plasticity, calibrate_flow_control])
@pytest.mark.parametrize(
    "kind", ["nan", "inf", "dtype", "empty", "features", "rank", "streams", "missing_driver"]
)
def test_invalid_data_is_transactional(calibrator, kind):
    layer = _layer(input_size=1 if kind == "missing_driver" else None)
    x = _data()
    if kind in ("nan", "inf"):
        x[0, 2, 0] = float(kind)
    elif kind == "dtype":
        x = x.float()
    elif kind == "empty":
        x = x[:, :0]
    elif kind == "features":
        x = x[..., :1]
    elif kind == "rank":
        x = x[0, 0]
    elif kind == "streams":
        x = (x, x)
    saved = copy.deepcopy(layer.state_dict())
    with pytest.raises((ValueError, TypeError)):
        calibrator(layer, x)
    assert saved.keys() == layer.state_dict().keys()
    assert all(torch.equal(value, saved[key]) for key, value in layer.state_dict().items())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_mean": 1.0},
        {"target_mean": float("inf")},
        {"target_std": 0.0},
        {"target_std": 1.1},
        {"target_mean": 0.9, "target_std": 0.5},
    ],
)
def test_rejects_impossible_ip_targets(kwargs):
    with pytest.raises(ValueError):
        calibrate_intrinsic_plasticity(_layer(), _data(), **kwargs)


@pytest.mark.parametrize(
    "kwargs", [{"mode": "other"}, {"target_radius": 0}, {"target_radius": float("nan")}]
)
def test_rejects_invalid_flow_options(kwargs):
    with pytest.raises(ValueError):
        calibrate_flow_control(_layer(), _data(), **kwargs)


def test_unsupported_regimes_are_explicit():
    with pytest.raises(ValueError, match="activation"):
        calibrate_intrinsic_plasticity(_layer(activation="relu"), _data())
    with pytest.raises(TypeError, match="ESNLayer"):
        calibrate_flow_control(torch.nn.Linear(2, 3), _data())
    with pytest.raises(ValueError, match="leak"):
        calibrate_flow_control(_layer(leak_rate=0.5), _data())
    layer = _layer()
    calibrate_intrinsic_plasticity(layer, _data())
    with pytest.raises(ValueError, match="before intrinsic"):
        calibrate_flow_control(layer, _data())
    empty = _layer(recurrent_initializer=(torch.nn.init.zeros_, {}))
    with pytest.raises(ValueError, match="all-zero"):
        calibrate_flow_control(empty, _data())


@pytest.mark.parametrize("sparse", [False, True])
def test_combined_calibration_checkpoint_old_checkpoint_and_pickle(sparse):
    source = _layer(sparse=sparse)
    x = _data()
    legacy = copy.deepcopy(source.state_dict())
    calibrate_flow_control(source, x)
    calibrate_intrinsic_plasticity(source, x)
    reference, _ = _run(source, x)
    destination = _layer(sparse=sparse)
    destination.load_state_dict(source.state_dict(), strict=True)
    output, _ = _run(destination, x)
    torch.testing.assert_close(output, reference)
    buffer = io.BytesIO()
    torch.save(destination, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=False)
    torch.testing.assert_close(_run(restored, x)[0], reference)
    destination.load_state_dict(legacy, strict=True)
    assert destination.cell._intrinsic_gain is None
    assert destination.cell._intrinsic_bias is None
    assert destination.cell._recurrent_gain is None
    torch.testing.assert_close(_run(destination, x)[0], _run(_layer(), x)[0])


def test_calibrated_layer_readout_fit_and_forecast_pipeline():
    layer = _layer(feedback_size=1, reservoir_size=8)
    x = torch.sin(torch.arange(100, dtype=torch.float64) * 0.1).view(1, -1, 1)
    calibrate_intrinsic_plasticity(layer, x[:, :70], target_std=0.3)
    feedback = Input(shape=(10, 1), dtype=torch.float64)
    output = RidgeReadoutLayer(8, 1, alpha=1e-7, name="output").double()(layer(feedback))
    model = ESNModel(feedback, output).double()
    ESNTrainer(model).fit(warmup_inputs=x[:, :10], train_inputs=x[:, 10:69], targets=x[:, 11:70])
    prediction = model.forecast(x[:, :70], horizon=5)
    assert prediction.shape == (1, 5, 1)
    assert torch.isfinite(prediction).all()
    module = model.as_module()
    u = x[:, :10].clone().requires_grad_()
    module(u).sum().backward()
    assert u.grad is not None and torch.isfinite(u.grad).all()


def test_extreme_finite_input_failure_does_not_partially_adapt():
    layer = _layer()
    x = _data()
    calibrate_intrinsic_plasticity(layer, x)
    saved = copy.deepcopy(layer.state_dict())
    layer(x[:, :3])
    state = layer.state.clone()
    with pytest.raises(RuntimeError, match="diverged"):
        calibrate_intrinsic_plasticity(layer, x, learning_rate=1e308)
    assert all(torch.equal(value, saved[key]) for key, value in layer.state_dict().items())
    torch.testing.assert_close(layer.state, state)
    with pytest.raises(ValueError, match="too small"):
        calibrate_intrinsic_plasticity(layer, x, target_std=1e-200)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_actual_mps_calibration_and_input_gradients():
    # This exercises the native local accelerator; hosted CPU jobs skip it.
    layer = ESNLayer(4, 2, seed=13, spectral_radius=0.4, device="mps")
    x = _data(batch=1, length=12).float().to("mps")
    report = calibrate_flow_control(layer, x, warmup=2)
    assert report.spectral_radius_after > 0
    calibrate_intrinsic_plasticity(layer, x, target_std=0.3)
    inputs = x.detach().requires_grad_()
    output, _ = _run(layer, inputs)
    output.sum().backward()
    assert inputs.grad is not None and bool(torch.isfinite(inputs.grad).all())
    assert layer.weight_hh.device.type == "mps"
    assert layer.cell._intrinsic_gain.device.type == "mps"


@pytest.mark.parametrize("sparse", [False, True])
def test_spectral_radius_property_tracks_both_calibration_gains_and_checkpoints(sparse):
    layer = _layer(sparse=sparse)
    x = _data(length=100)
    original = layer.weight_hh.clone()
    assert layer.spectral_radius_achieved == pytest.approx(0.4, rel=1e-12)
    report = calibrate_flow_control(layer, x, learning_rate=0.1, epochs=3)
    assert layer.spectral_radius_achieved == pytest.approx(report.spectral_radius_after, rel=1e-12)
    calibrate_intrinsic_plasticity(layer, x, target_std=0.3)
    effective = layer.cell._intrinsic_gain[:, None] * layer.cell._recurrent_gain[:, None] * original
    expected = torch.linalg.eigvals(effective).abs().max().item()
    assert layer.spectral_radius_achieved == pytest.approx(expected, rel=1e-12)
    torch.testing.assert_close(layer.weight_hh, original, atol=0, rtol=0)
    destination = _layer(sparse=sparse)
    destination.load_state_dict(layer.state_dict())
    assert destination.spectral_radius_achieved == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("length", [63, 64, 65, 10_000])
def test_report_preserves_low_float32_activity_variance_near_saturation(length):
    layer = ESNLayer(
        1,
        1,
        bias=False,
        feedback_initializer=torch.nn.init.ones_,
        recurrent_initializer=torch.nn.init.zeros_,
        seed=5,
    )
    inputs = (3 + 0.001 * torch.sin(torch.arange(length, dtype=torch.float32))).reshape(1, -1, 1)
    observed = torch.tanh(inputs).double()
    expected_mean = observed.mean().item()
    expected_std = observed.std(correction=0).item()
    assert 0 < expected_std < 1e-5
    report = calibrate_intrinsic_plasticity(layer, inputs, learning_rate=1e-12)
    assert report.activity_mean_before == pytest.approx(expected_mean, abs=1e-15)
    assert report.activity_std_before == pytest.approx(expected_std, rel=1e-11)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_report_uses_stable_float32_centered_variance():
    layer = ESNLayer(
        1,
        1,
        bias=False,
        feedback_initializer=torch.nn.init.ones_,
        recurrent_initializer=torch.nn.init.zeros_,
        seed=5,
        device="mps",
    )
    inputs = (
        (3 + 0.001 * torch.sin(torch.arange(257, dtype=torch.float32))).reshape(1, -1, 1).to("mps")
    )
    # Compare the actual backend's activations, whose last bits can differ from CPU tanh.
    observed = torch.tanh(inputs).cpu().double()
    report = calibrate_intrinsic_plasticity(layer, inputs, learning_rate=1e-12)
    assert report.activity_mean_before == pytest.approx(observed.mean().item(), abs=1e-10)
    assert report.activity_std_before == pytest.approx(observed.std(correction=0).item(), rel=1e-5)
