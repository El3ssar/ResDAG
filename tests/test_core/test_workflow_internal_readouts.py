"""Hidden supervised stages preserve units, clocks, state and public outputs."""

import copy

import pytest
import torch
from torch import nn

import resdag as rd
from resdag.ensemble import CoupledEnsembleESNModel


class _ConditionalRidge(rd.RidgeReadoutLayer):
    fail = False

    def fit(self, states, targets):
        if self.fail:
            raise RuntimeError("independent hidden-stage failure")
        return super().fit(states, targets)


def _build(*, preprocessing="standard", nested=False, name="stage", lead=1):
    source = rd.reservoir_input(1, dtype=torch.float64)
    internal = _ConditionalRidge(1, 1, alpha=0.05, name=name).double()
    memory = rd.ESNLayer(1, feedback_size=1, activation="identity", bias=False, dtype=torch.float64)
    with torch.no_grad():
        memory.weight_hh.fill_(0.4)
        memory.weight_feedback.fill_(1)
    memory.detach_state_between_calls = False
    features = nn.Sequential(internal, memory)(source) if nested else memory(internal(source))
    final = _ConditionalRidge(2, 1, alpha=0.07, name="terminal").double()
    graph = rd.ESNModel(source, final(rd.Concatenate()(features, source)))
    model = rd.Model(
        graph=graph,
        input_names=("state",),
        output_names=("prediction",),
        preprocessing=preprocessing,
        prediction_lead=lead,
        washout=3,
    )
    return model, internal, memory, final


def _series(two_dimensional):
    values = torch.sin(torch.linspace(-1, 5, 38, dtype=torch.float64)).reshape(2, 19, 1)
    values = values + torch.tensor([3.0, 5.0], dtype=torch.float64).reshape(2, 1, 1)
    return values[0] if two_dimensional else values


def _canonical(values):
    return values.unsqueeze(0) if values.ndim == 2 else values


def _ridge(features, targets, alpha):
    x = features.reshape(-1, features.shape[-1])
    y = targets.reshape(-1, targets.shape[-1])
    mean_x, mean_y = x.mean(0), y.mean(0)
    centered = x - mean_x
    coefficients = torch.linalg.solve(
        centered.T @ centered + alpha * torch.eye(x.shape[1], dtype=x.dtype),
        centered.T @ (y - mean_y),
    )
    return coefficients.T, mean_y - mean_x @ coefficients


def _statistics(values, enabled):
    if enabled:
        return values.mean((0, 1), keepdim=True), values.std((0, 1), keepdim=True)
    return values.new_zeros(1, 1, 1), values.new_ones(1, 1, 1)


def _states(inputs):
    state = torch.zeros_like(inputs[:, 0])
    outputs = []
    for sample in inputs.unbind(1):
        state = 0.4 * state + sample
        outputs.append(state)
    return torch.stack(outputs, dim=1)


def _reference(values, hidden, public, *, normalize, stop=None, implicit=False):
    values, hidden, public = map(_canonical, (values, hidden, public))
    stop = values.shape[1] if stop is None else stop
    mean_x, scale_x = _statistics(values[:, 3:stop], normalize)
    output_fit = values[:, 3:stop] if implicit else public[:, 3:]
    mean_y, scale_y = _statistics(output_fit, normalize)
    scaled = (values[:, :stop] - mean_x) / scale_x
    first = _ridge(scaled[:, 3:], hidden[:, 3:stop], 0.05)
    internal_output = scaled @ first[0].T + first[1]
    states = _states(internal_output)
    features = torch.cat((states, scaled), dim=-1)
    last = _ridge(features[:, 3:], (public[:, 3:] - mean_y) / scale_y, 0.07)
    return {
        "first": first,
        "last": last,
        "states": states,
        "prediction": (features @ last[0].T + last[1]) * scale_y + mean_y,
        "mean_x": mean_x,
        "scale_x": scale_x,
        "mean_y": mean_y,
        "scale_y": scale_y,
    }


def _forecast(reference, history, horizon):
    normalized = (_canonical(history) - reference["mean_x"]) / reference["scale_x"]
    hidden = normalized @ reference["first"][0].T + reference["first"][1]
    state = _states(hidden)[:, -1:]
    feedback = normalized[:, -1:]
    result = []
    for tick in range(horizon):
        if tick:
            feedback = (result[-1] - reference["mean_x"]) / reference["scale_x"]
            hidden_frame = feedback @ reference["first"][0].T + reference["first"][1]
            state = 0.4 * state + hidden_frame
        features = torch.cat((state, feedback), dim=-1)
        result.append(
            (features @ reference["last"][0].T + reference["last"][1]) * reference["scale_y"]
            + reference["mean_y"]
        )
    return torch.cat(result, dim=1), state


def _coupled_forecast(references, history, horizon):
    states, predictions = [], []
    for reference in references:
        prediction, state = _forecast(reference, history, 1)
        states.append(state)
        predictions.append(prediction)
    result = [torch.stack(predictions).mean(0)]
    for _ in range(horizon - 1):
        predictions = []
        for index, reference in enumerate(references):
            feedback = (result[-1] - reference["mean_x"]) / reference["scale_x"]
            hidden = feedback @ reference["first"][0].T + reference["first"][1]
            states[index] = 0.4 * states[index] + hidden
            features = torch.cat((states[index], feedback), dim=-1)
            predictions.append(
                (features @ reference["last"][0].T + reference["last"][1]) * reference["scale_y"]
                + reference["mean_y"]
            )
        result.append(torch.stack(predictions).mean(0))
    return torch.cat(result, dim=1), states


def _assert_coefficients(layer, expected):
    torch.testing.assert_close(layer.weight, expected[0], rtol=2e-9, atol=2e-10)
    torch.testing.assert_close(layer.bias, expected[1], rtol=2e-9, atol=2e-10)


def _equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _equal(a, b)
    else:
        assert actual == expected


@pytest.mark.parametrize("preprocessing", ["standard", "off"])
@pytest.mark.parametrize("two_dimensional", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_hidden_units_and_public_scaling_match_manual_fit_and_forecast(
    preprocessing, two_dimensional, nested
):
    model, internal, memory, final = _build(preprocessing=preprocessing, nested=nested)
    values = _series(two_dimensional)
    hidden = 2.5 * values + 40 + (0.3 * values).sin()
    public = 0.4 * values + values.sin() + 5
    reference = _reference(values, hidden, public, normalize=preprocessing == "standard")
    original_values, original_hidden = values.clone(), hidden.clone()
    parameter_ids = tuple(id(value) for value in model.parameters())
    assert model.internal_readout_names == ("stage",)
    assert model.output_names == ("prediction",)
    assert model.fit(values, public, readout_targets={"stage": hidden}) is model
    assert model.internal_readout_names == ("stage",)
    assert tuple(id(value) for value in model.parameters()) == parameter_ids
    _assert_coefficients(internal, reference["first"])
    _assert_coefficients(final, reference["last"])
    assert memory.state is None
    torch.testing.assert_close(values, original_values, rtol=0, atol=0)
    torch.testing.assert_close(hidden, original_hidden, rtol=0, atol=0)

    actual = model(values)
    assert isinstance(actual, torch.Tensor)
    expected = reference["prediction"]
    torch.testing.assert_close(
        actual, expected.squeeze(0) if two_dimensional else expected, rtol=2e-9, atol=2e-10
    )
    torch.testing.assert_close(memory.state, reference["states"][:, -1], rtol=2e-9, atol=2e-10)
    history = values[-5:] if two_dimensional else values[:, -5:]
    if nested:
        with pytest.raises(ValueError, match="hidden inside graph operation"):
            model.forecast(history, horizon=4, execution="history")
        return
    expected_forecast, expected_state = _forecast(reference, history, 4)
    actual_forecast = model.forecast(history, horizon=4)
    torch.testing.assert_close(
        actual_forecast,
        expected_forecast.squeeze(0) if two_dimensional else expected_forecast,
        rtol=3e-8,
        atol=3e-9,
    )
    torch.testing.assert_close(memory.state, expected_state.squeeze(1), rtol=3e-8, atol=3e-9)


@pytest.mark.parametrize("preprocessing", ["standard", "off"])
def test_implicit_lead_clips_hidden_labels_and_training_statistics(preprocessing):
    model, internal, _, final = _build(preprocessing=preprocessing, lead=3)
    values = _series(False)
    hidden = 2.5 * values + 40
    hidden[:, :3] = -1e7
    hidden[:, -3:] = 1e8
    stop = values.shape[1] - 3
    reference = _reference(
        values,
        hidden,
        values[:, 3:],
        normalize=preprocessing == "standard",
        stop=stop,
        implicit=True,
    )
    model.fit(values, readout_targets={"stage": hidden})
    _assert_coefficients(internal, reference["first"])
    _assert_coefficients(final, reference["last"])
    if preprocessing == "standard":
        torch.testing.assert_close(model.input_scalers[0]._stat0, reference["mean_x"])
        torch.testing.assert_close(model.output_scalers[0]._stat0, reference["mean_x"])
    torch.testing.assert_close(model._history_0, values[:, -6:], rtol=0, atol=0)
    assert model.forecast(horizon=3).shape == (2, 3, 1)


def test_unnamed_nested_internal_head_has_a_discoverable_graph_path():
    model, internal, _, _ = _build(nested=True, name=None)
    path = next(name for name, module in model.graph.named_modules() if module is internal)
    assert model.internal_readout_names == (path,)
    values = _series(True)
    model.fit(values, values + 1, readout_targets={path: 2 * values + 40})
    assert internal.is_fitted
    assert isinstance(model(values[:5]), torch.Tensor)


@pytest.mark.parametrize("preprocessing", ["standard", "off"])
def test_coupled_member_names_units_and_mean_predictions_are_independent(preprocessing):
    first, a, memory_a, out_a = _build(preprocessing=preprocessing)
    second, b, memory_b, out_b = _build(preprocessing=preprocessing)
    model = rd.Model(
        graph=CoupledEnsembleESNModel([first.graph, second.graph]),
        input_names=("state",),
        output_names=("prediction",),
        preprocessing=preprocessing,
        washout=3,
    )
    values = _series(False)
    public = 0.4 * values + values.sin() + 5
    hidden_a, hidden_b = 2 * values + 40, -3 * values - 15
    expected_a = _reference(values, hidden_a, public, normalize=preprocessing == "standard")
    expected_b = _reference(values, hidden_b, public, normalize=preprocessing == "standard")
    assert model.internal_readout_names == ("models.0.stage", "models.1.stage")
    model.fit(
        values,
        public,
        readout_targets={"models.1.stage": hidden_b, "models.0.stage": hidden_a},
    )
    _assert_coefficients(a, expected_a["first"])
    _assert_coefficients(b, expected_b["first"])
    _assert_coefficients(out_a, expected_a["last"])
    _assert_coefficients(out_b, expected_b["last"])
    expected = (expected_a["prediction"] + expected_b["prediction"]) / 2
    torch.testing.assert_close(model(values), expected, rtol=2e-9, atol=2e-10)
    assert model.output_names == ("prediction",)
    history = values[:, -5:]
    prediction, states = _coupled_forecast((expected_a, expected_b), history, 4)
    torch.testing.assert_close(model.forecast(history, horizon=4), prediction, rtol=3e-8, atol=3e-9)
    for memory, state in zip((memory_a, memory_b), states):
        torch.testing.assert_close(memory.state, state.squeeze(1), rtol=3e-8, atol=3e-9)


@pytest.mark.parametrize(
    "invalid",
    ["missing", "empty", "unknown", "extra", "time", "batch", "features", "layout", "nonfinite"],
)
@pytest.mark.parametrize("refit", [False, True])
def test_invalid_hidden_targets_reject_before_graph_execution_without_mutation(invalid, refit):
    model, _, _, _ = _build()
    values = _series(True)
    public, hidden = values + 1, 2 * values + 40
    if refit:
        model.fit(values, public, readout_targets={"stage": hidden})
    before = copy.deepcopy(model.state_dict())
    targets = {"stage": hidden.clone()}
    if invalid == "missing":
        targets = None
    elif invalid == "empty":
        targets = {}
    elif invalid == "unknown":
        targets = {"other": hidden}
    elif invalid == "extra":
        targets["other"] = hidden
    elif invalid == "time":
        targets["stage"] = hidden[:-1]
    elif invalid == "batch":
        targets["stage"] = hidden.unsqueeze(0).repeat(2, 1, 1)
    elif invalid == "features":
        targets["stage"] = hidden.repeat(1, 2)
    elif invalid == "layout":
        targets["stage"] = hidden.unsqueeze(0)
    else:
        targets["stage"][4] = float("nan")
    calls = []
    handle = model.graph.register_forward_pre_hook(lambda module, args: calls.append(True))
    try:
        with pytest.raises((ValueError, TypeError), match="readout_targets"):
            model.fit(values, public, readout_targets=targets)
    finally:
        handle.remove()
    assert calls == []
    _equal(model.state_dict(), before)
    assert model.is_fitted == refit


@pytest.mark.parametrize("failing", ["internal", "terminal"])
def test_failed_internal_refit_preserves_weights_scalers_history_state_and_gradients(failing):
    model, internal, memory, final = _build()
    values = _series(False)
    public, hidden = values + 1, 2 * values + 40
    model.fit(values, public, readout_targets={"stage": hidden})
    source = values[:, :5].clone().requires_grad_()
    prediction = model(source)
    live_state = memory.state
    assert live_state is not None and live_state.grad_fn is not None
    before = copy.deepcopy(model.state_dict())
    parameter_ids = tuple(id(value) for value in model.parameters())
    (internal if failing == "internal" else final).fail = True
    with pytest.raises(RuntimeError, match="independent hidden-stage failure"):
        model.fit(values + 3, public + 2, readout_targets={"stage": hidden - 4})
    _equal(model.state_dict(), before)
    assert memory.state is live_state
    assert tuple(id(value) for value in model.parameters()) == parameter_ids
    prediction.square().sum().backward()
    assert source.grad is not None and bool((source.grad != 0).any())
    assert not internal._forward_pre_hooks and not final._forward_pre_hooks


@pytest.mark.parametrize("coupled", [False, True])
def test_invalid_head_invocations_fail_before_custom_preprocessing_fit(coupled):
    fit_calls = []

    class SpyUnits(nn.Module):
        supports_framewise = True

        def __init__(self):
            super().__init__()
            self.register_buffer("scale", torch.ones(1, dtype=torch.float64))

        def fit(self, training):
            fit_calls.append(training.shape)
            return self

        def forward(self, values):
            return values * self.scale

        def inverse(self, values):
            return values / self.scale

    source = rd.reservoir_input(1, dtype=torch.float64)
    shared = _ConditionalRidge(1, 1, name="shared").double()
    invalid = rd.ESNModel(source, shared(shared(source)))
    if coupled:
        valid, _, _, _ = _build()
        graph = CoupledEnsembleESNModel([valid.graph, invalid])
    else:
        graph = invalid
    model = rd.Model(
        graph=graph,
        input_names=("state",),
        output_names=("prediction",),
        input_transforms={"state": SpyUnits()},
        washout=3,
    )
    values = _series(True)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="shared.*positions|invoked more than once"):
        model.fit(
            values,
            values + 1,
            readout_targets={"models.0.stage": 2 * values + 40} if coupled else None,
        )
    assert fit_calls == []
    _equal(model.state_dict(), before)
