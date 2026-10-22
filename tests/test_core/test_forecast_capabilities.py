"""Fast execution is declared; history replay preserves whole-prefix semantics."""

import copy

import pytest
import torch
from pytorch_symbolic import useful_layers
from torch import nn

import resdag as rd
from resdag.core._flat_inference import build_flat_step
from resdag.core._forecast_capabilities import (
    ForecastSequenceExecutor,
    validate_forecast_execution,
)


class _Cumulative(nn.Module):
    def forward(self, value):
        return value.cumsum(1)


class _FramewiseShift(nn.Module):
    supports_framewise = True

    def forward(self, value):
        return value + 1


class _InheritedShift(_FramewiseShift):
    pass


class _OverriddenTemporal(_FramewiseShift):
    def forward(self, value):
        return value.cumsum(1)


def _graph(layer, *, reservoir=False):
    symbolic = rd.reservoir_input(1, dtype=torch.float64)
    value = layer(symbolic)
    if reservoir:
        recurrent = rd.ESNLayer(
            1, feedback_size=1, activation="identity", seed=7, dtype=torch.float64
        )
        with torch.no_grad():
            recurrent.cell.weight_hh.fill_(0.5)
            recurrent.cell.weight_feedback.fill_(1)
            recurrent.cell.bias_h.zero_()
        value = recurrent(value)
    return rd.ESNModel(symbolic, value)


def _history():
    return torch.tensor([[[1.0], [2.0], [3.0]]], dtype=torch.float64)


@pytest.mark.parametrize("entry", ["raw_one", "raw_many", "flat", "build", "executor"])
def test_undeclared_sequence_operation_rejects_before_hooks_modes_or_state(entry):
    graph = _graph(_Cumulative(), reservoir=True)
    graph(_history())
    state = graph.get_state()
    modes = [module.training for module in graph.modules()]
    calls = []
    handles = [
        module.register_forward_hook(lambda *_: calls.append(1)) for module in graph.modules()
    ]
    try:
        with pytest.raises(ValueError, match="supports_framewise=True.*execution='history'"):
            if entry.startswith("raw"):
                graph.forecast(_history(), horizon=1 if entry == "raw_one" else 3)
            elif entry == "flat":
                graph._get_flat_step()
            elif entry == "build":
                build_flat_step(graph)
            else:
                ForecastSequenceExecutor(graph)
    finally:
        for handle in handles:
            handle.remove()
    assert calls == [] and modes == [module.training for module in graph.modules()]
    for name, value in graph.get_state().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    # Whole-sequence customization remains available through ordinary APIs.
    full, _ = graph.forward_stateless(_history())
    assert full.shape == _history().shape


def test_cached_flat_preflight_detects_new_temporal_child_without_running_it():
    container = nn.Sequential(nn.Identity())
    graph = _graph(container)
    graph._get_flat_step()
    container.append(_Cumulative())
    calls = []
    handle = container.register_forward_hook(lambda *_: calls.append(1))
    try:
        with pytest.raises(ValueError, match="one frame at a time"):
            graph._get_flat_step()
    finally:
        handle.remove()
    assert calls == []


@pytest.mark.parametrize("layer", [_FramewiseShift(), _InheritedShift()])
def test_explicit_framewise_declaration_keeps_native_fast_behavior(layer):
    graph = _graph(layer)
    expected = torch.tensor([[[4.0], [5.0], [6.0]]], dtype=torch.float64)
    torch.testing.assert_close(graph.forecast(_history(), horizon=3), expected, rtol=0, atol=0)


def test_override_must_renew_inherited_framewise_declaration():
    graph = _graph(_OverriddenTemporal())
    with pytest.raises(ValueError, match="_OverriddenTemporal"):
        validate_forecast_execution(graph)
    validate_forecast_execution(graph, "history")


def test_instance_declaration_allows_existing_custom_graph_operation():
    layer = _FramewiseShift()
    layer.forward = lambda value: value * 2
    with pytest.raises(ValueError, match="one frame at a time"):
        validate_forecast_execution(_graph(layer))
    layer.supports_framewise = True
    graph = _graph(layer)
    validate_forecast_execution(graph)
    torch.testing.assert_close(
        graph.forecast(_history(), horizon=2), torch.tensor([[[6.0], [12.0]]]).double()
    )


@pytest.mark.parametrize("execution", ["fast", "history"])
def test_pointwise_and_reservoir_execution_matches_full_prefix_oracle(execution):
    graph = _graph(nn.Sequential(nn.Linear(1, 1, dtype=torch.float64), nn.Tanh()), reservoir=True)
    graph(_history())
    stored = {
        name: layer.state for name, layer in graph.named_modules() if isinstance(layer, rd.ESNLayer)
    }
    executor = ForecastSequenceExecutor(graph, execution)
    outputs, states = executor.start((_history(),))
    prefix = _history()
    for _ in range(4):
        frame = outputs[0][:, -1:]
        prefix = torch.cat((prefix, frame), 1)
        expected, expected_states = graph.forward_stateless(prefix)
        outputs, states = executor.advance((frame,))
        torch.testing.assert_close(outputs[0], expected[:, -1:], rtol=1e-13, atol=1e-13)
        for name, state in states.items():
            torch.testing.assert_close(state, expected_states[name], rtol=0, atol=0)
    assert all(
        layer.state is stored[name]
        for name, layer in graph.named_modules()
        if isinstance(layer, rd.ESNLayer)
    )


def test_history_replay_retains_observed_and_generated_prefix_for_cumulative_operation():
    graph = _graph(_Cumulative())
    executor = ForecastSequenceExecutor(graph, "history")
    outputs, states = executor.start((_history(),))
    predictions = [outputs[0][:, -1:]]
    for _ in range(3):
        outputs, states = executor.advance((predictions[-1],))
        predictions.append(outputs[0])
    torch.testing.assert_close(
        torch.cat(predictions, 1),
        torch.tensor([[[6.0], [12.0], [24.0], [48.0]]]).double(),
        rtol=0,
        atol=0,
    )
    assert states == {} and graph.get_state() == {}


def test_history_replays_from_original_explicit_state_with_matching_gradients():
    graph = _graph(_Cumulative(), reservoir=True)
    source = _history().requires_grad_()
    name = graph._get_execution_plan().reservoir_names[0]
    initial = {name: torch.tensor([[2.0]], dtype=torch.float64, requires_grad=True)}
    executor = ForecastSequenceExecutor(graph, "history")
    outputs, _ = executor.start((source,), states=initial)
    predictions = [outputs[0][:, -1:]]
    for _ in range(2):
        outputs, final = executor.advance((predictions[-1],))
        predictions.append(outputs[0])
    actual = torch.cat(predictions, 1)
    oracle_source = source.detach().clone().requires_grad_()
    oracle_initial = {name: initial[name].detach().clone().requires_grad_()}
    prefix = oracle_source
    expected = []
    for _ in range(3):
        output, oracle_final = graph.forward_stateless(prefix, states=oracle_initial)
        frame = output[:, -1:]
        expected.append(frame)
        prefix = torch.cat((prefix, frame), 1)
    expected = torch.cat(expected, 1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(final[name], oracle_final[name], rtol=0, atol=0)
    actual_grad = torch.autograd.grad(actual.square().sum(), (source, initial[name]))
    expected_grad = torch.autograd.grad(
        expected.square().sum(), (oracle_source, oracle_initial[name])
    )
    for observed, reference in zip(actual_grad, expected_grad):
        torch.testing.assert_close(observed, reference, rtol=1e-13, atol=1e-13)
    assert graph.get_state() == {}


@pytest.mark.parametrize("execution,compile", [("unknown", False), ("history", True)])
def test_invalid_execution_configuration_rejects_before_hooks(execution, compile):
    graph = _graph(nn.Identity())
    calls = []
    handle = graph.register_forward_hook(lambda *_: calls.append(1))
    try:
        with pytest.raises(ValueError):
            ForecastSequenceExecutor(graph, execution, compile=compile)
    finally:
        handle.remove()
    assert calls == []


@pytest.mark.parametrize(
    "operation",
    [
        lambda value: value + 2,
        lambda value: 2 + value,
        lambda value: value - 2,
        lambda value: 2 - value,
        lambda value: value * 2,
        lambda value: value / 2,
        lambda value: 2 / value,
        lambda value: value**2,
        lambda value: 2**value,
        lambda value: value % 2,
        lambda value: 2 % value,
        lambda value: -value,
        lambda value: abs(value),
        lambda value: value + value,
        lambda value: value**value,
        lambda value: value / value,
        lambda value: value[:, :, :1],
        lambda value: value[..., [0]],
    ],
)
def test_verified_symbolic_scalar_and_feature_operations_remain_fast(operation):
    symbolic = rd.reservoir_input(1, dtype=torch.float64)
    graph = rd.ESNModel(symbolic, operation(symbolic))
    validate_forecast_execution(graph)
    flat = graph._get_flat_step()
    frame = torch.tensor([[[0.5]]], dtype=torch.float64)
    output, _ = flat.step((frame,), [])
    torch.testing.assert_close(output[0], graph(frame), rtol=0, atol=0)


@pytest.mark.parametrize(
    "constant", [torch.tensor([2.0]), torch.tensor([[2.0]]), torch.tensor([[[2.0]]])]
)
def test_symbolic_feature_only_broadcast_constants_are_safe(constant):
    symbolic = rd.reservoir_input(1)
    graph = rd.ESNModel(symbolic, symbolic + constant)
    validate_forecast_execution(graph)


def test_time_shaped_broadcast_constant_is_not_implicitly_framewise():
    symbolic = rd.reservoir_input(1)
    graph = rd.ESNModel(symbolic, symbolic + torch.ones(3, 1))
    with pytest.raises(ValueError, match="LambdaOpLayer"):
        validate_forecast_execution(graph)


class _TemporalTensor(torch.Tensor):
    @classmethod
    def __torch_function__(cls, function, types, args=(), kwargs=None):
        if function.__name__ == "add":
            return args[0].cumsum(1)
        return super().__torch_function__(function, types, args, kwargs)


def test_tensor_subclass_cannot_claim_scalar_arithmetic_safety_by_shape_alone():
    symbolic = rd.reservoir_input(1)
    constant = torch.ones(1).as_subclass(_TemporalTensor)
    graph = rd.ESNModel(symbolic, symbolic + constant)
    with pytest.raises(ValueError, match="LambdaOpLayer"):
        validate_forecast_execution(graph)


@pytest.mark.parametrize(
    "layer",
    [
        useful_layers.LambdaOpLayer(lambda value: value.cumsum(1)),
        useful_layers.LambdaOpLayer(lambda value: value + 1),
        useful_layers.NamedLambdaOpLayer(lambda value: value + 1, "arbitrary"),
        useful_layers.MethodCall("cumsum", 1),
        useful_layers.SliceLayer((slice(None), slice(0, 1), slice(None))),
        nn.Softmax(dim=1),
        nn.LayerNorm((1, 1), dtype=torch.float64),
    ],
)
def test_generic_lambdas_and_temporal_operations_are_never_blanket_approved(layer):
    graph = _graph(layer)
    with pytest.raises(ValueError, match="one frame at a time"):
        validate_forecast_execution(graph)


class _TemporalAverage(nn.Module):
    def forward(self, stacked):
        return stacked.mean(0).cumsum(1)


@pytest.mark.parametrize("temporal_member", [False, True])
def test_coupled_members_and_custom_aggregator_are_checked_and_replayed(temporal_member):
    member = _graph(_Cumulative() if temporal_member else nn.Identity())
    graph = rd.CoupledEnsembleESNModel(
        [member, copy.deepcopy(member)],
        aggregator="mean" if temporal_member else _TemporalAverage(),
    )
    with pytest.raises(ValueError, match="one frame at a time"):
        validate_forecast_execution(graph)
    executor = ForecastSequenceExecutor(graph, "history")
    output, _ = executor.start((_history(),))
    output, _ = executor.advance((output[0][:, -1:],))
    torch.testing.assert_close(output[0], torch.tensor([[[12.0]]]).double(), rtol=0, atol=0)


def test_parallel_native_reservoir_and_readout_remain_fast():
    graph = rd.parallel_esn(4, 2, 1, 3, seed=7, dtype=torch.float64)
    validate_forecast_execution(graph)
    executor = ForecastSequenceExecutor(graph)
    history = torch.ones(1, 4, 4, dtype=torch.float64)
    executor.start((history,))
    output, states = executor.advance((history[:, :1],))
    assert output[0].shape == (1, 1, 4) and states["reservoir"].shape == (2, 3)


@pytest.mark.parametrize("execution", ["fast", "history"])
def test_executor_requires_start_and_one_matching_frame_per_input(execution):
    executor = ForecastSequenceExecutor(_graph(nn.Identity()), execution)
    with pytest.raises(RuntimeError, match="Call start"):
        executor.advance((_history()[:, :1],))
    executor.start((_history(),))
    with pytest.raises(ValueError, match="one frame"):
        executor.advance((_history(),))
    output, _ = executor.advance((_history()[:, :1],))
    torch.testing.assert_close(output[0], _history()[:, :1], rtol=0, atol=0)


class _FailingCumulative(nn.Module):
    fail = False

    def forward(self, value):
        if self.fail:
            raise RuntimeError("deliberate sequence failure")
        return value.cumsum(1)


def test_failed_history_advance_preserves_original_prefix():
    layer = _FailingCumulative()
    executor = ForecastSequenceExecutor(_graph(layer), "history")
    output, _ = executor.start((_history(),))
    frame = output[0][:, -1:]
    layer.fail = True
    with pytest.raises(RuntimeError, match="deliberate sequence failure"):
        executor.advance((frame,))
    layer.fail = False
    output, _ = executor.advance((frame,))
    torch.testing.assert_close(output[0], torch.tensor([[[12.0]]]).double(), rtol=0, atol=0)


def test_cached_scalar_arithmetic_rejects_mutated_time_shaped_constant():
    symbolic = rd.reservoir_input(1)
    constant = torch.ones(1)
    graph = rd.ESNModel(symbolic, symbolic + constant)
    graph._get_flat_step()
    constant.resize_(3, 1)
    with pytest.raises(ValueError, match="LambdaOpLayer"):
        graph._get_flat_step()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Native MPS unavailable")
def test_history_replay_and_gradients_remain_on_native_device_without_numpy_or_cpu(monkeypatch):
    graph = _graph(_Cumulative(), reservoir=True).float().to("mps")
    source = _history().float().to("mps").requires_grad_()

    def forbidden(*args, **kwargs):
        raise AssertionError("Tensor history execution must not transfer through CPU or NumPy")

    with monkeypatch.context() as context:
        context.setattr(torch.Tensor, "cpu", forbidden)
        context.setattr(torch.Tensor, "numpy", forbidden)
        executor = ForecastSequenceExecutor(graph, "history")
        output, _ = executor.start((source,))
        output, states = executor.advance((output[0][:, -1:],))
        output[0].sum().backward()
        assert output[0].device.type == "mps"
        assert all(value.device.type == "mps" for value in states.values())
        assert source.grad is not None and source.grad.device.type == "mps"
        assert bool(torch.isfinite(source.grad).all())


def _temporal_workflow(*, bias=True):
    symbolic = rd.reservoir_input(1, dtype=torch.float64)
    readout = rd.SVDReadoutLayer(1, 1, alpha=0, bias=bias).double()
    model = rd.Model(
        inputs={"series": symbolic},
        outputs={"prediction": readout(_Cumulative()(symbolic))},
        washout=2,
        preprocessing="off",
    )
    return model, readout


def test_temporal_readout_fit_uses_full_prefix_features_before_washout_slice():
    model, readout = _temporal_workflow(bias=False)
    training = torch.tensor([[1.0], [2.0], [4.0], [3.0], [5.0], [2.0]], dtype=torch.float64)
    target = 2 * training.cumsum(0)
    model.fit(training, targets={"prediction": target})
    torch.testing.assert_close(
        readout.weight, torch.tensor([[2.0]]).double(), rtol=1e-13, atol=1e-13
    )
    torch.testing.assert_close(model(training), target, rtol=1e-13, atol=1e-13)


def test_temporal_implicit_fit_and_cached_history_match_independent_prefix_oracle():
    model, readout = _temporal_workflow()
    observed = torch.tensor([[1.0], [2.0], [4.0], [3.0], [5.0], [2.0], [6.0]], dtype=torch.float64)
    features = observed[:-1].cumsum(0)[model.washout :]
    design = torch.cat((features, torch.ones_like(features)), 1)
    coefficients = torch.linalg.lstsq(design, observed[1:][model.washout :]).solution
    model.fit(observed)
    torch.testing.assert_close(readout.weight, coefficients[:1].T, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(readout.bias, coefficients[1], rtol=1e-12, atol=1e-12)
    prefix = observed.clone()
    expected = []
    for _ in range(3):
        frame = prefix.sum(0, keepdim=True) @ coefficients[:1] + coefficients[1]
        expected.append(frame)
        prefix = torch.cat((prefix, frame), 0)
    expected = torch.cat(expected)
    cached = model.forecast(horizon=3, execution="history")
    explicit = model.forecast(history=observed, horizon=3, execution="history")
    torch.testing.assert_close(cached, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(explicit, expected, rtol=1e-12, atol=1e-12)
