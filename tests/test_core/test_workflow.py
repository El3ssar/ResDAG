"""Original-unit workflow contracts, independent timing and numerical oracles."""

import copy

import pytest
import torch
from torch import nn

from resdag import ESNLayer, ESNModel, RidgeReadoutLayer, reservoir_input
from resdag.core.workflow import Model
from resdag.layers import Concatenate
from resdag.models import classic_esn
from resdag.training import ESNTrainer


def _affine(*, preprocessing="standard", washout=1, cls=RidgeReadoutLayer):
    x = reservoir_input(1, dtype=torch.float64)
    head = cls(1, 1, name="private_head", alpha=0).double()
    return Model(
        inputs={"state": x},
        outputs={"prediction": head(x)},
        washout=washout,
        preprocessing=preprocessing,
    )


def _assert_state_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        if isinstance(actual[key], torch.Tensor):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        else:
            assert actual[key] == expected[key]


def test_manual_scaled_training_and_forecast_match_without_leakage():
    generator = torch.Generator().manual_seed(30)
    series = torch.randn(2, 40, 2, generator=generator, dtype=torch.float64) + 100
    graph = classic_esn(8, 2, 2, seed=13, dtype=torch.float64, readout="ridge")
    manual = copy.deepcopy(graph)
    model = Model(graph=graph, washout=4)
    parameter_ids = [id(value) for value in graph.parameters()]
    model.fit(series)
    # Independent arithmetic: statistics exclude washout and final observed target.
    fitting_inputs = series[:, 4:-1]
    mean = fitting_inputs.mean((0, 1), keepdim=True)
    std = fitting_inputs.std((0, 1), keepdim=True)
    scaled = (series - mean) / std
    ESNTrainer(manual).fit((scaled[:, :4],), (scaled[:, 4:-1],), {"output": scaled[:, 5:]})
    _assert_state_equal(graph.state_dict(), manual.state_dict())
    assert parameter_ids == [id(value) for value in model.graph.parameters()]
    assert set(model._modules) == {"graph", "input_scalers", "output_scalers"}
    assert model.model is graph
    assert not any(key.startswith("model.") for key in model.state_dict())
    torch.testing.assert_close(model.input_scalers[0]._stat0, mean, rtol=0, atol=0)
    torch.testing.assert_close(model.output_scalers[0]._stat0, mean, rtol=0, atol=0)
    torch.testing.assert_close(model._history_0, series[:, -5:], rtol=0, atol=0)
    manual.eval()
    expected = manual.forecast(scaled[:, -5:], horizon=7) * std + mean
    torch.testing.assert_close(model.forecast(horizon=7), expected, rtol=0, atol=0)
    before = copy.deepcopy(model.state_dict())
    model.forecast(series[:, -5:] + 1e3, horizon=2)
    _assert_state_equal(model.state_dict(), before)


def test_named_multihead_targets_do_not_need_internal_readout_names():
    x = reservoir_input(1, dtype=torch.float64)
    a = RidgeReadoutLayer(1, 1, name="internal_a", alpha=0).double()(x)
    b = RidgeReadoutLayer(1, 1, name="internal_b", alpha=0).double()(x)
    model = Model(inputs={"measurement": x}, outputs={"position": a, "velocity": b}, washout=2)
    series = torch.linspace(-4, 7, 30, dtype=torch.float64)[:, None]
    targets = {"velocity": -2 * series + 5, "position": 3 * series - 8}
    with pytest.raises(ValueError, match="explicit named targets"):
        model.fit(series)
    model.fit(series, targets)
    output = model(series)
    assert list(output) == ["position", "velocity"]
    for name, expected in targets.items():
        torch.testing.assert_close(output[name], expected, rtol=1e-12, atol=1e-12)
    assert not torch.equal(model.input_scalers[0]._stat0, model.output_scalers[0]._stat0)
    before = model.get_state()
    with pytest.raises(ValueError, match="feedback bridge"):
        model.forecast(series[-2:], horizon=3)
    _assert_state_equal(model.get_state(), before)


def test_stateless_outputs_keep_original_units_states_and_upstream_gradients():
    graph = classic_esn(6, 1, 1, seed=18, dtype=torch.float64)
    model = Model(graph=graph, washout=2).fit(torch.randn(1, 25, 1, dtype=torch.float64))
    model(torch.ones(1, 2, 1, dtype=torch.float64))
    stored = model.get_state()
    data = torch.randn(1, 7, 1, dtype=torch.float64, requires_grad=True)
    all_output, all_states = model.forward_stateless(data)
    a, state = model.forward_stateless(data[:, :3])
    b, final = model.forward_stateless(data[:, 3:], states=state)
    torch.testing.assert_close(torch.cat((a, b), 1), all_output, rtol=0, atol=0)
    _assert_state_equal(final, all_states)
    _assert_state_equal(model.get_state(), stored)
    all_output.square().sum().backward()
    assert data.grad is not None and bool((data.grad != 0).any())


def test_internal_readouts_are_rejected_before_mutation():
    x = reservoir_input(1, dtype=torch.float64)
    a = RidgeReadoutLayer(1, 1, name="internal").double()(x)
    b = RidgeReadoutLayer(1, 1, name="terminal").double()(a)
    model = Model(inputs={"state": x}, outputs={"result": b}, washout=1)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="no internal fitted readouts"):
        model.fit(torch.arange(10, dtype=torch.float64)[:, None])
    _assert_state_equal(model.state_dict(), before)


def test_hidden_sequential_reservoir_supports_stateful_fit_and_checkpoint_restore():
    def build():
        x = reservoir_input(1, dtype=torch.float64)
        before = nn.Identity()
        hidden = nn.Sequential(
            nn.Identity(),
            ESNLayer(4, feedback_size=1, seed=7, dtype=torch.float64),
        )(before(x))
        y = RidgeReadoutLayer(4, 1, alpha=1e-5).double()(hidden)
        return Model(inputs={"state": x}, outputs={"prediction": y}, washout=2), before

    series = torch.sin(torch.arange(24, dtype=torch.float64) / 4).reshape(-1, 1)
    model, before = build()
    assert model.fit(series) is model
    assert model.is_fitted
    model.eval()
    output = model(series[:5])
    assert output.shape == (5, 1) and bool(torch.isfinite(output).all())
    carried = model.get_state()
    assert carried
    expected = model(series[5:9])
    model.reset_state()
    assert model.get_state() == {}
    model.set_state(carried)
    torch.testing.assert_close(model(series[5:9]), expected, rtol=0, atol=0)

    restored, _ = build()
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    restored.eval()
    assert restored.is_fitted
    restored.set_state(model.get_state())
    torch.testing.assert_close(restored(series[9:12]), model(series[9:12]), rtol=0, atol=0)

    stored = model.get_state()
    calls = []
    handle = before.register_forward_hook(lambda *_: calls.append(None))
    try:
        for execute in (
            lambda: model.forward_stateless(series[:5]),
            lambda: model.forecast(horizon=2),
        ):
            with pytest.raises(ValueError, match="hidden inside.*functional graph node"):
                execute()
            assert calls == []
            _assert_state_equal(model.get_state(), stored)
    finally:
        handle.remove()


def test_dtype_conversion_moves_statistics_cached_history_and_forecasts():
    model = _affine().fit(torch.arange(20, dtype=torch.float64)[:, None]).float()
    assert all(
        value.dtype == torch.float32 for value in model.buffers() if value.is_floating_point()
    )
    result = model.forecast(horizon=2)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, torch.tensor([[20.0], [21.0]]), rtol=1e-5, atol=1e-5)


def test_unsupported_backend_and_constructor_conflicts_have_clear_errors():
    graph = _affine().graph
    with pytest.raises(ValueError, match="not both"):
        Model(graph=graph, inputs={"x": graph.inputs[0]})
    with pytest.raises(TypeError, match="currently supports ESNModel"):
        Model(graph=nn.Linear(1, 1))
    x, y = reservoir_input(1), reservoir_input(1)
    two_inputs = ESNModel((x, y), Concatenate()(x, y))
    with pytest.raises(ValueError, match="distinct names"):
        Model(graph=two_inputs, input_names=("x", "x"))
    with pytest.raises(ValueError, match="non-empty named"):
        Model()


@pytest.mark.parametrize("entry", ["forward", "fit", "forecast", "stateless"])
def test_raw_graph_port_mutation_is_rejected_before_state_or_hooks_change(entry):
    model = Model(graph=classic_esn(4, 1, 1, seed=21, dtype=torch.float64), washout=1)
    training = torch.arange(20, dtype=torch.float64).reshape(-1, 1)
    model.fit(training)
    model(training[-2:])
    stored = model.get_state()
    model.graph.add_output(model.graph.outputs[0].parents[0])
    calls = []
    handle = model.graph.register_forward_hook(lambda *_: calls.append(1))
    try:
        with pytest.raises(ValueError, match="ports changed; construct a new Model"):
            if entry == "forward":
                model(training)
            elif entry == "fit":
                model.fit(training)
            elif entry == "forecast":
                model.forecast(horizon=2)
            else:
                model.forward_stateless(training)
    finally:
        handle.remove()
    assert calls == []
    _assert_state_equal(model.get_state(), stored)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_native_mps_recurrent_workflow_matches_cpu_forecast_and_upstream_gradients():
    graph = classic_esn(5, 1, 1, seed=22, dtype=torch.float64)
    training = torch.sin(torch.linspace(0, 7, 40, dtype=torch.float64))[:, None]
    cpu = Model(graph=graph, washout=3).fit(training).float()
    accelerated = copy.deepcopy(cpu).to("mps")
    expected = cpu.forecast(horizon=4)
    actual = accelerated.forecast(horizon=4)
    assert actual.device.type == "mps"
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-4)
    cpu.reset_state()
    accelerated.reset_state()
    x = torch.ones(2, 3, 1, requires_grad=True)
    on_mps = x.detach().clone().to("mps").requires_grad_()
    cpu(x).square().sum().backward()
    accelerated(on_mps).square().sum().backward()
    torch.testing.assert_close(on_mps.grad.cpu(), x.grad, rtol=1e-4, atol=1e-4)


class _RefitFailingReadout(RidgeReadoutLayer):
    fail = False

    def fit(self, states, targets):
        if self.fail:
            raise RuntimeError("deliberate refit failure")
        return super().fit(states, targets)


@pytest.mark.parametrize("fail", [False, True])
def test_refit_excludes_only_transient_attached_states_from_staging_copy(fail):
    from resdag.layers.reservoirs import BaseReservoirLayer

    x = reservoir_input(1, dtype=torch.float64)
    from resdag import ESNLayer

    reservoir = ESNLayer(4, feedback_size=1, seed=42, dtype=torch.float64)
    reservoir.detach_state_between_calls = False
    readout = _RefitFailingReadout(4, 1, name="internal").double()
    model = Model(inputs={"state": x}, outputs={"next": readout(reservoir(x))}, washout=2)
    series = torch.randn(1, 20, 1, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    model.fit(series)
    parameter_ids = [id(value) for value in model.parameters()]
    source = torch.randn(1, 3, 1, dtype=torch.float64, requires_grad=True)
    output = model(source)
    layer = next(layer for layer in model.graph.modules() if isinstance(layer, BaseReservoirLayer))
    live_state = layer.state
    assert live_state is not None and live_state.grad_fn is not None
    before = copy.deepcopy(model.state_dict())
    readout.fail = fail
    if fail:
        with pytest.raises(RuntimeError, match="deliberate refit failure"):
            model.fit(series + 2)
        assert layer.state is live_state
        _assert_state_equal(model.state_dict(), before)
        output.square().sum().backward()
        assert source.grad is not None and bool((source.grad != 0).any())
    else:
        model.fit(series + 2)
        assert layer.state is None
        assert model.is_fitted
    assert parameter_ids == [id(value) for value in model.parameters()]
