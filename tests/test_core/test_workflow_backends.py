"""Private adapters agree with independent raw-family training and rollouts."""

import copy
import io

import pytest
import torch
from torch import nn

from resdag import ESNLayer, ESNModel, Model, RidgeReadoutLayer, reservoir_input
from resdag.ensemble.coupled import CoupledEnsembleESNModel
from resdag.layers import Concatenate
from resdag.models import classic_esn, parallel_esn
from resdag.training import ESNTrainer


def _parallel():
    return parallel_esn(
        input_dim=6,
        core_width=2,
        halo=1,
        reservoir_size=7,
        seed=23,
        dtype=torch.float64,
        readout_alpha=1e-5,
    )


def _member(seed, name, *, driven=False, failing=False):
    feedback = reservoir_input(1, dtype=torch.float64)
    reservoir = ESNLayer(
        5,
        feedback_size=1,
        input_size=1 if driven else 0,
        seed=seed,
        dtype=torch.float64,
    )
    inputs = (feedback, reservoir_input(1, dtype=torch.float64)) if driven else (feedback,)
    # A genuine branch exposes the raw feedback beside its recurrent features.
    features = Concatenate()(reservoir(*inputs), feedback)
    cls = _FailingReadout if failing else RidgeReadoutLayer
    head = cls(6, 1, name=name, alpha=1e-5).double()
    return ESNModel(inputs, head(features))


def _coupled(*, driven=False, aggregator="mean", failing=False):
    return CoupledEnsembleESNModel(
        [
            _member(31, "first_private_head", driven=driven),
            _member(32, "different_head", driven=driven, failing=failing),
        ],
        aggregator=aggregator,
    )


def _series(width, *, batch=2):
    values = torch.randn(
        batch, 32, width, generator=torch.Generator().manual_seed(43), dtype=torch.float64
    )
    return values * torch.arange(1, width + 1, dtype=torch.float64) + 17


def _equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _equal(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _equal(left, right)
    else:
        assert actual == expected


@pytest.mark.parametrize("family", ["parallel", "coupled"])
@pytest.mark.parametrize("return_warmup", [False, True])
def test_scaled_fit_and_forecast_match_raw_backend_oracle(family, return_warmup):
    graph = _parallel() if family == "parallel" else _coupled()
    raw = copy.deepcopy(graph)
    width = 6 if family == "parallel" else 1
    values = _series(width)
    model = Model(graph=graph, input_names=("observed",), output_names=("prediction",), washout=3)
    identities = [id(parameter) for parameter in graph.parameters()]
    assert model.fit(values) is model
    mean = values[:, 3:-1].mean((0, 1), keepdim=True)
    std = values[:, 3:-1].std((0, 1), keepdim=True)
    scaled = (values - mean) / std
    if family == "parallel":
        raw.fit(scaled[:, :3], scaled[:, 3:-1], scaled[:, 4:])
    else:
        # The raw Tensor convenience path assumes identical internal head names.
        # Fit each member explicitly to verify the wrapper's named target mapping.
        for member, key in zip(raw.models, ("first_private_head", "different_head")):
            ESNTrainer(member).fit((scaled[:, :3],), (scaled[:, 3:-1],), {key: scaled[:, 4:]})
    _equal(model.graph.state_dict(), raw.state_dict())
    assert identities == [id(parameter) for parameter in model.graph.parameters()]
    assert model.graph is graph and not hasattr(model, "model")
    assert set(model._modules) == {"graph", "input_scalers", "output_scalers"}
    assert not any(key.startswith("model.") for key in model.state_dict())
    raw.eval()
    expected = raw.forecast(scaled[:, -4:], horizon=5, return_warmup=return_warmup) * std + mean
    actual = model.forecast(horizon=5, return_warmup=return_warmup)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (2, 8 if return_warmup else 5, width)
    assert model.input_scalers[0].n_features == width
    if family == "parallel":
        assert model.get_state()["reservoir"].shape == (6, 7)


def test_parallel_full_field_next_sample_seam_matches_physical_increment():
    graph = parallel_esn(
        6,
        2,
        0,
        2,
        activation="identity",
        seed=12,
        dtype=torch.float64,
        readout="svd",
        readout_alpha=0,
    )
    with torch.no_grad():
        graph.reservoir.cell.weight_hh.zero_()
        graph.reservoir.cell.weight_feedback.copy_(torch.eye(2, dtype=torch.float64))
        graph.reservoir.cell.bias_h.zero_()
    series = torch.arange(20, dtype=torch.float64)[:, None] + torch.arange(6)[None, :]
    model = Model(graph=graph, washout=0).fit(series)
    expected = series[-1:] + torch.arange(1, 5, dtype=torch.float64)[:, None]
    torch.testing.assert_close(model.forecast(horizon=4), expected, rtol=1e-12, atol=1e-12)
    history = series[-3:]
    result = model.forecast(history=history, horizon=4, return_warmup=True)
    torch.testing.assert_close(result, torch.cat((history[1:], expected)), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("horizon", [1, 4])
@pytest.mark.parametrize("aggregate", ["mean", "median"])
def test_coupled_driven_branched_forecast_matches_explicit_step_oracle(horizon, aggregate):
    graph = _coupled(driven=True, aggregator=aggregate)
    model = Model(
        graph=graph,
        input_names=("state", "driver"),
        output_names=("next",),
        washout=2,
        preprocessing="off",
    )
    state, driver = _series(1), _series(1).flip(1)
    target = 0.9 * state + 0.3 * driver
    model.fit({"driver": driver, "state": state}, {"next": target})
    history = (state[:, -3:], driver[:, -3:])
    future = driver[:, : horizon - 1]
    raw = copy.deepcopy(graph).eval()
    raw.reset_reservoirs()
    stacked = torch.stack([member(*history) for member in raw.models])

    def aggregate_outputs(stacked):
        return stacked.mean(0) if aggregate == "mean" else stacked.quantile(0.5, dim=0)

    prediction = aggregate_outputs(stacked)[:, -1:]
    predictions = [prediction]
    for step in range(horizon - 1):
        prediction = aggregate_outputs(
            torch.stack([member(prediction, future[:, step : step + 1]) for member in raw.models])
        )
        predictions.append(prediction)
    before = copy.deepcopy(model.state_dict())
    result = model.forecast(
        {"driver": history[1], "state": history[0]},
        horizon=horizon,
        future_inputs={"driver": future},
    )
    torch.testing.assert_close(result, torch.cat(predictions, 1), rtol=1e-12, atol=1e-12)
    _equal(model.state_dict(), before)


@pytest.mark.parametrize("family", ["parallel", "coupled"])
def test_state_hooks_chunk_gradients_and_deepcopy_remain_native(family):
    graph = _parallel() if family == "parallel" else _coupled()
    width = 6 if family == "parallel" else 1
    model = Model(graph=graph, washout=2).fit(_series(width))
    model.eval()
    data = _series(width)[:, :7].requires_grad_()
    calls = []
    handle = model.graph.register_forward_hook(lambda *_: calls.append("graph"))
    whole = model(data)
    handle.remove()
    assert calls == ["graph"]
    model.reset_state()
    for reservoir in model.graph.modules():
        if isinstance(reservoir, ESNLayer):
            reservoir.detach_state_between_calls = False
    prefix = model(data[:, :3])
    stored = model.get_state()
    suffix = model(data[:, 3:])
    expected_state = model.get_state()
    torch.testing.assert_close(torch.cat((prefix, suffix), 1), whole, rtol=1e-12, atol=1e-12)
    whole_gradient = torch.autograd.grad(whole.square().sum(), data)[0]
    carried_gradient = torch.autograd.grad(torch.cat((prefix, suffix), 1).square().sum(), data)[0]
    torch.testing.assert_close(carried_gradient, whole_gradient, rtol=1e-11, atol=1e-11)
    model.set_state(stored)
    torch.testing.assert_close(model(data[:, 3:]), suffix, rtol=0, atol=0)
    _equal(model.get_state(), expected_state)
    model.reset_state()
    explicit, states = model.forward_stateless(data)
    a, middle = model.forward_stateless(data[:, :3])
    b, final = model.forward_stateless(data[:, 3:], states=middle)
    _equal(final, states)
    _equal(model.get_state(), {})
    torch.testing.assert_close(torch.cat((a, b), 1), explicit, rtol=1e-12, atol=1e-12)
    full_grad = torch.autograd.grad(explicit.square().sum(), data)[0]
    chunk_grad = torch.autograd.grad(torch.cat((a, b), 1).square().sum(), data)[0]
    torch.testing.assert_close(chunk_grad, full_grad, rtol=1e-11, atol=1e-11)
    assert bool((full_grad != 0).any())
    copied = copy.deepcopy(model)
    assert copied.graph is not model.graph
    torch.testing.assert_close(copied.forecast(horizon=3), model.forecast(horizon=3))


@pytest.mark.parametrize("family", ["parallel", "coupled"])
@pytest.mark.parametrize("assign", [False, True])
def test_checkpoint_roundtrip_and_dtype_move(family, assign):
    graph = _parallel() if family == "parallel" else _coupled()
    width = 6 if family == "parallel" else 1
    model = Model(graph=graph, washout=3).fit(_series(width))
    expected = model.forecast(horizon=4)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    checkpoint = torch.load(stream, weights_only=True)
    restored = Model(graph=_parallel() if family == "parallel" else _coupled(), washout=0)
    restored.load_state_dict(checkpoint, assign=assign)
    assert restored.washout == 3 and restored.is_fitted
    torch.testing.assert_close(restored.forecast(horizon=4), expected, rtol=0, atol=0)
    moved = restored.float()
    result = moved.forecast(horizon=4)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result.double(), expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("family", ["parallel", "coupled"])
def test_fit_in_inference_mode_with_zero_washout_keeps_upstream_gradients(family):
    graph = _parallel() if family == "parallel" else _coupled()
    width = 6 if family == "parallel" else 1
    model = Model(graph=graph, washout=0)
    with torch.inference_mode():
        model.fit(_series(width))
    assert all(not value.is_inference() for value in model.buffers())
    data = _series(width)[:, :3].requires_grad_()
    model(data).sum().backward()
    assert data.grad is not None and bool((data.grad != 0).any())


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
@pytest.mark.parametrize("family", ["parallel", "coupled"])
def test_native_mps_backend_forecast_and_upstream_gradients_match_cpu(family):
    graph = _parallel() if family == "parallel" else _coupled()
    width = 6 if family == "parallel" else 1
    cpu = Model(graph=graph, washout=2).fit(_series(width)).float()
    mps = copy.deepcopy(cpu).to("mps")
    expected, actual = cpu.forecast(horizon=3), mps.forecast(horizon=3)
    assert actual.device.type == "mps"
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-4, atol=2e-4)
    data = _series(width)[:, :3].float().requires_grad_()
    mps_data = data.detach().clone().to("mps").requires_grad_()
    cpu_prediction, _ = cpu.forward_stateless(data)
    mps_prediction, _ = mps.forward_stateless(mps_data)
    cpu_prediction.square().sum().backward()
    mps_prediction.square().sum().backward()
    torch.testing.assert_close(mps_data.grad.cpu(), data.grad, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("family", ["parallel", "coupled"])
def test_unsupported_compiled_forecast_rejects_before_state_hooks_or_modes(family):
    graph = _parallel() if family == "parallel" else _coupled()
    width = 6 if family == "parallel" else 1
    model = Model(graph=graph, washout=2).fit(_series(width))
    model(_series(width)[:, :2])
    state = model.get_state()
    flags = [module.training for module in model.modules()]
    calls = []
    handles = [
        module.register_forward_pre_hook(lambda *_: calls.append(1)) for module in graph.modules()
    ]
    try:
        with pytest.raises(ValueError, match="Compiled forecasting is not supported"):
            model.forecast(horizon=3, compile=True)
    finally:
        for handle in handles:
            handle.remove()
    assert calls == [] and flags == [module.training for module in model.modules()]
    _equal(model.get_state(), state)


@pytest.mark.parametrize("mutation", ["geometry", "member", "aggregator", "port"])
def test_backend_structural_mutation_rejects_before_state_changes(mutation):
    graph = _parallel() if mutation == "geometry" else _coupled()
    width = 6 if mutation == "geometry" else 1
    model = Model(graph=graph, washout=2).fit(_series(width))
    model(_series(width)[:, :2])
    if mutation == "geometry":
        graph.halo = 2
    elif mutation == "member":
        graph.models[0] = copy.deepcopy(graph.models[0])
    elif mutation == "aggregator":
        graph._aggregator_str = "median"
    else:
        graph.models[0].add_output(graph.models[0].outputs[0])
    stored = model.get_state()
    calls = []
    handle = graph.register_forward_hook(lambda *_: calls.append(1))
    try:
        with pytest.raises(ValueError, match="ports changed; construct a new Model"):
            model(_series(width)[:, :2])
    finally:
        handle.remove()
    assert calls == []
    _equal(model.get_state(), stored)


@pytest.mark.parametrize(
    "mismatch", ["kind", "geometry", "member_count", "aggregator", "parallel_size", "member_size"]
)
def test_checkpoint_backend_schema_mismatch_is_preflighted(mismatch):
    source = Model(graph=_parallel(), washout=2).fit(_series(6))
    if mismatch == "kind":
        destination = Model(graph=classic_esn(7, 6, 6, seed=11, dtype=torch.float64))
    elif mismatch == "geometry":
        destination = Model(graph=parallel_esn(6, 3, 1, 7, seed=23, dtype=torch.float64))
    elif mismatch == "parallel_size":
        destination = Model(graph=parallel_esn(6, 2, 1, 9, seed=23, dtype=torch.float64))
    else:
        source = Model(graph=_coupled(), washout=2).fit(_series(1))
        graph = _coupled(aggregator="median" if mismatch == "aggregator" else "mean")
        if mismatch == "member_count":
            graph.models.append(_member(33, "third_head"))
        elif mismatch == "member_size":
            graph.models[0] = classic_esn(8, 1, 1, seed=31, dtype=torch.float64)
        destination = Model(graph=graph)
    before = copy.deepcopy(destination.state_dict())
    with pytest.raises(RuntimeError, match="backend schema differs"):
        destination.load_state_dict(source.state_dict())
    _equal(destination.state_dict(), before)


@pytest.mark.parametrize("missing", ["backend", "prediction_lead", "feedback", "transforms"])
def test_missing_current_checkpoint_schema_rejects_before_copy(missing):
    model = Model(graph=classic_esn(5, 1, 1, seed=19, dtype=torch.float64), washout=2)
    model.fit(_series(1))
    checkpoint = copy.deepcopy(model.state_dict())
    del checkpoint["_extra_state"][missing]
    restored = Model(graph=classic_esn(5, 1, 1, seed=19, dtype=torch.float64))
    before = copy.deepcopy(restored.state_dict())
    with pytest.raises(RuntimeError, match="metadata fields"):
        restored.load_state_dict(checkpoint)
    _equal(restored.state_dict(), before)


class _WeightedAggregate(nn.Module):
    supports_framewise = True

    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([0.3, 0.7], dtype=torch.float64))

    def forward(self, stacked):
        weights = self.logits.softmax(0).reshape(-1, 1, 1, 1)
        return (stacked * weights).sum(0)


def test_custom_feedforward_aggregator_keeps_parameter_identity_and_gradients():
    graph = _coupled(aggregator=_WeightedAggregate())
    weight = graph.aggregator_module.logits
    model = Model(graph=graph, washout=2).fit(_series(1))
    assert model.graph.aggregator_module.logits is weight
    model.reset_state()
    data = _series(1)[:, :4].requires_grad_()
    prediction, _ = model.forward_stateless(data)
    prediction.square().sum().backward()
    assert data.grad is not None and weight.grad is not None
    assert bool((data.grad != 0).any()) and bool((weight.grad != 0).any())
    restored = Model(graph=_coupled(aggregator=_WeightedAggregate()))
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored.forecast(horizon=3), model.forecast(horizon=3))


class _ReadoutAggregate(nn.Module):
    supports_framewise = True

    def __init__(self):
        super().__init__()
        self.average = RidgeReadoutLayer(2, 1, bias=False).double()
        with torch.no_grad():
            self.average.weight.fill_(0.5)

    def forward(self, stacked):
        return self.average(stacked.movedim(0, -1).squeeze(-2))


def test_user_supplied_readout_based_aggregator_is_not_automatically_fitted():
    graph = _coupled(aggregator=_ReadoutAggregate())
    assert not graph.aggregator_module.average.is_fitted
    initial = graph.aggregator_module.average.weight.detach().clone()
    model = Model(graph=graph, washout=2).fit(_series(1))
    _equal(graph.aggregator_module.average.weight, initial)
    assert not graph.aggregator_module.average.is_fitted
    model.reset_state()
    data = _series(1)[:, :3]
    torch.testing.assert_close(model(data), model.forward_stateless(data)[0])
    forecast = model.forecast(horizon=3)
    restored = Model(graph=_coupled(aggregator=_ReadoutAggregate()))
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored.forecast(horizon=3), forecast)


class _RecurrentAggregate(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = ESNLayer(1, feedback_size=1, seed=2, dtype=torch.float64)

    def forward(self, stacked):
        return self.hidden(stacked.mean(0))


@pytest.mark.parametrize("entry", ["stateless", "forecast"])
def test_hidden_recurrent_aggregator_is_preflighted_before_hooks_or_state(entry):
    graph = _coupled(aggregator=_RecurrentAggregate())
    model = Model(graph=graph, preprocessing="off", washout=2).fit(_series(1))
    model(_series(1)[:, :2])
    assert graph.aggregator_module.hidden.state is not None
    before = graph.aggregator_module.hidden.state.clone()
    member_states = model.get_state()
    calls = []
    handles = [
        module.register_forward_hook(lambda *_: calls.append(1)) for module in graph.modules()
    ]
    try:
        with pytest.raises(ValueError, match="hidden inside a coupled aggregator"):
            if entry == "stateless":
                model.forward_stateless(_series(1)[:, :2])
            else:
                model.forecast(horizon=3)
    finally:
        for handle in handles:
            handle.remove()
    assert calls == []
    _equal(graph.aggregator_module.hidden.state, before)
    _equal(model.get_state(), member_states)


class _FailingReadout(RidgeReadoutLayer):
    fail = False

    def fit(self, states, targets):
        if self.fail:
            raise RuntimeError("deliberate adapter fit failure")
        return super().fit(states, targets)


@pytest.mark.parametrize("family", ["parallel", "coupled"])
def test_failed_refit_keeps_all_weights_scalers_tail_and_attached_states(family):
    if family == "parallel":
        graph = _parallel()
        graph.readout = _FailingReadout(7, 2, name="output", alpha=1e-5).double()
    else:
        graph = _coupled(failing=True)
    width = 6 if family == "parallel" else 1
    model = Model(graph=graph, washout=2).fit(_series(width))
    for reservoir in graph.modules():
        if isinstance(reservoir, ESNLayer):
            reservoir.detach_state_between_calls = False
    source = _series(width)[:, :3].requires_grad_()
    prediction = model(source)
    live_states = [layer.state for layer in graph.modules() if isinstance(layer, ESNLayer)]
    fitted = copy.deepcopy(model.state_dict())
    for head in graph.modules():
        if isinstance(head, _FailingReadout):
            head.fail = True
    with pytest.raises(RuntimeError, match="deliberate adapter fit failure"):
        model.fit(_series(width) + 100)
    _equal(model.state_dict(), fitted)
    assert all(
        before is layer.state
        for before, layer in zip(
            live_states, (m for m in graph.modules() if isinstance(m, ESNLayer))
        )
    )
    prediction.sum().backward()
    assert source.grad is not None and bool((source.grad != 0).any())
