"""Workflow persistence, failed-fit isolation and previously reproduced boundaries."""

import copy
import io

import pytest
import torch
from torch import nn

from resdag.core import reservoir_input
from resdag.core.workflow import Model
from resdag.layers import ESNLayer
from resdag.layers.readouts import CGReadoutLayer, RidgeReadoutLayer


def _series(dtype=torch.float32):
    return torch.arange(24, dtype=dtype).reshape(-1, 1)


def _model(*, preprocessing="standard", washout=2, recurrent=False):
    x = reservoir_input(1)
    hidden = ESNLayer(6, feedback_size=1)(x) if recurrent else x
    y = RidgeReadoutLayer(6 if recurrent else 1, 1, alpha=1e-6)(hidden)
    return Model(
        inputs={"state": x},
        outputs={"prediction": y},
        preprocessing=preprocessing,
        washout=washout,
    )


def _driver_model(*, reverse_ports=False):
    state, driver = reservoir_input(1), reservoir_input(1)
    hidden = nn.Bilinear(1, 1, 1, bias=False)(state, driver)
    output = RidgeReadoutLayer(1, 1)(hidden)
    inputs = (
        {"driver": driver, "state": state} if reverse_ports else {"state": state, "driver": driver}
    )
    return Model(inputs=inputs, outputs={"prediction": output}, washout=2)


def _assert_state_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
        else:
            assert actual[key] == value


def test_fit_inside_inference_mode_preserves_upstream_gradients():
    model = _model()
    with torch.inference_mode():
        model.fit(_series())
    x = _series().requires_grad_()
    prediction = model(x)
    prediction.sum().backward()
    torch.testing.assert_close(prediction, x + 1, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=1e-5, atol=1e-5)
    assert all(not value.is_inference() for value in model.buffers())
    assert all(not value.requires_grad for value in model.input_scalers.buffers())
    assert all(not value.requires_grad for value in model.output_scalers.buffers())


@pytest.mark.parametrize("assign", [False, True])
@pytest.mark.parametrize("load_inference", [False, True])
def test_fresh_checkpoint_load_restores_configuration_history_predictions_and_gradients(
    assign, load_inference
):
    source = _model(preprocessing="minmax", washout=3).fit(_series())
    stream = io.BytesIO()
    torch.save(source.state_dict(), stream)
    stream.seek(0)
    state = torch.load(stream, weights_only=True)
    restored = _model(preprocessing=None, washout=0)
    with torch.inference_mode(load_inference):
        restored.load_state_dict(state, assign=assign)
    assert restored.is_fitted
    assert restored.preprocessing == "minmax"
    assert restored.washout == 3
    torch.testing.assert_close(
        restored.forecast(horizon=5), source.forecast(horizon=5), rtol=0, atol=0
    )
    x = _series().requires_grad_()
    restored(x).sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=1e-5, atol=1e-5)
    assert all(not value.is_inference() for value in restored.buffers())
    assert "model" not in restored._modules
    assert list(restored._modules).count("graph") == 1


def test_prefixed_state_dict_roundtrip_preserves_unknown_scaler_width_and_metadata():
    source = nn.Sequential(_model(preprocessing="noncentered", washout=4).fit(_series()))
    restored = nn.Sequential(_model(preprocessing=None, washout=0))
    restored.load_state_dict(source.state_dict())
    assert restored[0].preprocessing == "noncentered"
    assert restored[0].washout == 4
    torch.testing.assert_close(restored(_series()), source(_series()), rtol=0, atol=0)
    torch.testing.assert_close(
        restored[0].forecast(horizon=3), source[0].forecast(horizon=3), rtol=0, atol=0
    )


def test_assign_loading_preserves_checkpoint_dtype_instead_of_destination_dtype():
    source = _model().double().fit(_series(torch.float64))
    restored = _model()
    restored.load_state_dict(source.state_dict(), assign=True)
    assert all(value.dtype == torch.float64 for value in restored.parameters())
    assert all(
        value.dtype == torch.float64 for value in restored.buffers() if value.is_floating_point()
    )
    torch.testing.assert_close(
        restored.forecast(horizon=3), source.forecast(horizon=3), rtol=0, atol=0
    )


def test_inference_created_history_is_loaded_as_a_detached_registered_buffer():
    source = _model().fit(_series())
    state = copy.deepcopy(source.state_dict())
    with torch.inference_mode():
        state["_history_0"] = state["_history_0"].clone()
    restored = _model()
    with torch.inference_mode():
        restored.load_state_dict(state, assign=True)
    assert not restored._history_0.is_inference()
    assert not restored._history_0.requires_grad
    assert "_history_0" in dict(restored.named_buffers())
    torch.testing.assert_close(
        restored.forecast(horizon=2), source.forecast(horizon=2), rtol=0, atol=0
    )


def test_unfitted_checkpoint_clears_previous_fit_and_cached_history():
    source = _model(preprocessing="minmax", washout=0)
    restored = _model().fit(_series())
    restored.load_state_dict(source.state_dict())
    assert not restored.is_fitted
    assert restored.preprocessing == "minmax"
    assert restored.washout == 0
    assert not restored.input_scalers[0].is_fitted
    assert not restored.output_scalers[0].is_fitted
    assert restored._history_0.numel() == 0
    with pytest.raises(RuntimeError, match="fit"):
        restored(_series())


def test_equal_width_input_port_reordering_is_rejected_before_mutation():
    source = _driver_model()
    target = _driver_model(reverse_ports=True)
    before = copy.deepcopy(target.state_dict())
    with pytest.raises(RuntimeError, match="input_names"):
        target.load_state_dict(source.state_dict())
    _assert_state_equal(target.state_dict(), before)


@pytest.mark.parametrize("field", ["input_names", "output_names"])
def test_checkpoint_port_names_cannot_silently_remap_equal_width_streams(field):
    source = _model().fit(_series())
    state = copy.deepcopy(source.state_dict())
    state["_extra_state"][field] = ("renamed",)
    restored = _model().fit(_series() + 7)
    before = copy.deepcopy(restored.state_dict())
    with pytest.raises(RuntimeError, match=field):
        restored.load_state_dict(state)
    _assert_state_equal(restored.state_dict(), before)


@pytest.mark.parametrize("change", ["method", "width", "missing_fit", "empty_history"])
def test_fitted_checkpoint_enforces_its_declared_preprocessing_and_history_schema(change):
    source = _model().fit(_series())
    state = copy.deepcopy(source.state_dict())
    if change == "method":
        state["input_scalers.0._method_code"] = torch.tensor(0)
    elif change == "width":
        state["input_scalers.0._stat0"] = torch.zeros(1, 1, 2)
        state["input_scalers.0._stat1"] = torch.ones(1, 1, 2)
    elif change == "missing_fit":
        state["output_scalers.0._stat0"] = torch.empty(1, 1, 0)
        state["output_scalers.0._stat1"] = torch.empty(1, 1, 0)
    else:
        state["_history_0"] = torch.empty(0, 1)
    restored = _model(preprocessing="minmax").fit(_series() + 11)
    before = copy.deepcopy(restored.state_dict())
    with pytest.raises(RuntimeError, match="checkpoint"):
        restored.load_state_dict(state)
    _assert_state_equal(restored.state_dict(), before)


@pytest.mark.parametrize("failure", ["mapping", "solver"])
def test_failed_refit_preserves_graph_weights_scalers_history_and_carried_state(
    failure, monkeypatch
):
    model = _model(recurrent=True).fit(_series())
    model(_series()[:5])
    before = copy.deepcopy(model.state_dict())
    recurrent_state = model.get_state()
    assert recurrent_state
    parameter_ids = tuple(id(value) for value in model.parameters())
    if failure == "mapping":
        with pytest.raises(ValueError, match="keys"):
            model.fit({"wrong": _series() + 10})
    else:

        def failing_solver(*args, **kwargs):
            raise RuntimeError("deliberate solve failure")

        monkeypatch.setattr(RidgeReadoutLayer, "_fit_impl", failing_solver)
        with pytest.raises(RuntimeError, match="deliberate solve failure"):
            model.fit(_series() + 10)
    assert model.is_fitted
    assert tuple(id(value) for value in model.parameters()) == parameter_ids
    _assert_state_equal(model.state_dict(), before)
    _assert_state_equal(model.get_state(), recurrent_state)


def test_horizon_one_allows_explicitly_empty_future_driver_streams():
    model = _driver_model()
    series = _series()
    drivers = series.flip(0)
    model.fit({"state": series, "driver": drivers}, targets=series)
    history = {"state": series[-3:], "driver": drivers[-3:]}
    expected = model.forecast(history, horizon=1)
    actual = model.forecast(history, horizon=1, future_inputs={"driver": torch.empty(0, 1)})
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_native_mps_fit_cpu_checkpoint_reload_and_upstream_gradients():
    def build():
        x = reservoir_input(1)
        hidden = ESNLayer(8, feedback_size=1, spectral_radius=0.5, seed=73)(x)
        # MPS has no float64 and lacks the direct Cholesky-solve operator.
        y = CGReadoutLayer(8, 1, alpha=1e-2, max_iter=200, tol=1e-6, use_float64=False)(hidden)
        return Model(inputs={"state": x}, outputs={"prediction": y}, washout=5)

    time = torch.linspace(0, 9, 80)
    series = (7 + torch.sin(time) + 0.05 * time).unsqueeze(-1)
    cpu = build()
    source = copy.deepcopy(cpu).to("mps")
    cpu.fit(series)
    source.fit(series.to("mps"))
    assert all(value.device.type == "mps" for value in source.buffers())
    expected_cpu = cpu.forecast(horizon=6)
    expected_mps = source.forecast(horizon=6).cpu()
    torch.testing.assert_close(expected_mps, expected_cpu, rtol=5e-4, atol=5e-4)
    source.cpu()
    assert all(value.device.type == "cpu" for value in source.buffers())
    restored = build().to("mps")
    restored.load_state_dict(source.state_dict())
    assert all(value.device.type == "mps" for value in restored.buffers())
    actual = restored.forecast(horizon=6)
    assert actual.device.type == "mps"
    torch.testing.assert_close(actual.cpu(), expected_mps, rtol=1e-5, atol=1e-5)
    cpu.reset_state()
    restored.reset_state()
    x_cpu = series[:12].clone().requires_grad_()
    x_mps = x_cpu.detach().to("mps").requires_grad_()
    output_cpu, output_mps = cpu(x_cpu), restored(x_mps)
    torch.testing.assert_close(output_mps.cpu(), output_cpu, rtol=5e-4, atol=5e-4)
    output_cpu.square().mean().backward()
    output_mps.square().mean().backward()
    assert bool(torch.isfinite(x_mps.grad).all()) and bool((x_mps.grad != 0).any())
    torch.testing.assert_close(x_mps.grad.cpu(), x_cpu.grad, rtol=1e-3, atol=1e-4)
