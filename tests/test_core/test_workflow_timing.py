"""Independent physical-unit, sample-time and state oracles for the model workflow."""

import copy

import pytest
import torch
from torch import nn

from resdag.core import reservoir_input
from resdag.core.workflow import Model, Preprocessing
from resdag.layers import Concatenate, ESNLayer, RidgeReadoutLayer


def _increment_model(preprocessing: Preprocessing) -> Model:
    inp = reservoir_input(1, dtype=torch.float64)
    readout = RidgeReadoutLayer(1, 1, alpha=0.0, name="internal_readout").double()
    return Model(
        inputs={"position": inp},
        outputs={"position_next": readout(inp)},
        washout=0,
        preprocessing=preprocessing,
    )


@pytest.mark.parametrize(
    "preprocessing", ["standard", "minmax", "noncentered", "meanpreserving", "off"]
)
def test_physical_increment_recurrence_has_exact_forecast_boundary(
    preprocessing: Preprocessing,
) -> None:
    """Learning u -> u+1 must advance in physical units under every scaling rule."""
    observed = torch.arange(8, dtype=torch.float64).reshape(-1, 1)
    model = _increment_model(preprocessing).fit(observed)
    expected = torch.tensor([[8.0], [9.0], [10.0], [11.0]], dtype=torch.float64)
    torch.testing.assert_close(model.forecast(horizon=4), expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(
        model.forecast(history=observed, horizon=1), expected[:1], rtol=1e-10, atol=1e-10
    )
    # Every historical input predicts its next integer. The boundary prediction
    # 8 belongs once to the returned history/forecast sequence, followed by 9.
    with_warmup = model.forecast(history=observed, horizon=4, return_warmup=True)
    torch.testing.assert_close(
        with_warmup,
        torch.arange(1, 12, dtype=torch.float64).reshape(-1, 1),
        rtol=1e-10,
        atol=1e-10,
    )


def test_named_multihead_driver_forecast_follows_physical_sample_indices() -> None:
    """The recurrence x[n+1]=x[n]+d[n] fixes each driver and forecast index."""
    state = reservoir_input(1, dtype=torch.float64)
    forcing = reservoir_input(1, dtype=torch.float64)
    features = Concatenate()(state, forcing)
    prediction = RidgeReadoutLayer(2, 1, alpha=0.0, name="internal_prediction").double()
    auxiliary = RidgeReadoutLayer(2, 1, alpha=0.0, name="internal_auxiliary").double()
    model = Model(
        inputs={"state": state, "forcing": forcing},
        outputs={"prediction": prediction(features), "energy": auxiliary(features)},
        washout=0,
        preprocessing="off",
    )
    values = torch.tensor([[0.0], [1.0], [3.0], [6.0], [10.0], [15.0]], dtype=torch.float64)
    drivers = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]], dtype=torch.float64)
    # Deliberately reverse both mappings relative to constructor order. Internal
    # readout names differ from public output names as well.
    model.fit(
        {"forcing": drivers, "state": values},
        targets={"energy": values * 2, "prediction": values + drivers},
    )
    history = {"forcing": drivers[:4], "state": values[:4]}
    single = model.forecast(history=history, horizon=1)
    assert isinstance(single, dict)
    torch.testing.assert_close(
        single["prediction"], torch.tensor([[10.0]], dtype=torch.float64), rtol=1e-10, atol=1e-10
    )
    result = model.forecast(history=history, future_inputs={"forcing": drivers[4:]}, horizon=3)
    assert isinstance(result, dict)
    assert tuple(result) == ("prediction", "energy")
    torch.testing.assert_close(
        result["prediction"],
        torch.tensor([[10.0], [15.0], [21.0]], dtype=torch.float64),
        rtol=1e-10,
        atol=1e-10,
    )
    torch.testing.assert_close(
        result["energy"],
        torch.tensor([[12.0], [20.0], [30.0]], dtype=torch.float64),
        rtol=1e-10,
        atol=1e-10,
    )


def _recurrent_model() -> tuple[Model, ESNLayer]:
    inp = reservoir_input(1, dtype=torch.float64)
    reservoir = ESNLayer(3, feedback_size=1, seed=17, dtype=torch.float64)
    reservoir.detach_state_between_calls = False
    readout = RidgeReadoutLayer(3, 1, alpha=1e-3, name="internal_readout").double()
    return (
        Model(inputs={"state": inp}, outputs={"next": readout(reservoir(inp))}, washout=2),
        reservoir,
    )


def test_scaled_stateful_forward_preserves_hooks_and_gradients_across_chunks() -> None:
    model, reservoir = _recurrent_model()
    training = torch.sin(torch.arange(25, dtype=torch.float64) / 4).reshape(-1, 1)
    model.fit(training).eval()
    reference = copy.deepcopy(model)
    encoder = nn.Linear(2, 1, dtype=torch.float64)
    with torch.no_grad():
        encoder.weight.fill_(0.3)
        encoder.bias.fill_(0.1)
    source = torch.linspace(-1, 1, 12, dtype=torch.float64).reshape(1, 6, 2).requires_grad_()
    encoded = encoder(source)
    calls: list[None] = []
    handle = reservoir.register_forward_hook(lambda *_: calls.append(None))
    try:
        first = model(encoded[:, :3])
        second = model(encoded[:, 3:])
    finally:
        handle.remove()
    assert isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor)
    second.square().sum().backward()
    assert calls == [None, None]
    assert source.grad is not None and source.grad[:, :3].abs().sum() > 0
    assert encoder.weight.grad is not None and encoder.weight.grad.abs().sum() > 0
    expected = reference(encoded.detach())
    torch.testing.assert_close(torch.cat((first, second), dim=1), expected)
    stored = model.get_state()
    assert stored
    for name, state in reference.get_state().items():
        torch.testing.assert_close(stored[name], state)
    stateless, _ = model.forward_stateless(encoded.detach())
    torch.testing.assert_close(stateless, expected)
    for name, state in model.get_state().items():
        torch.testing.assert_close(state, stored[name], rtol=0, atol=0)


def test_different_explicit_target_scaling_rejects_before_state_hooks_or_modes_change() -> None:
    model, reservoir = _recurrent_model()
    training = torch.arange(8, dtype=torch.float64).reshape(-1, 1)
    model.fit(training, targets={"next": training + 100})
    model(training)
    stored = model.get_state()
    assert stored
    model.train()
    model.graph.eval()
    reservoir.train()
    modes = [module.training for module in model.modules()]
    calls: list[None] = []
    handle = reservoir.register_forward_hook(lambda *_: calls.append(None))
    try:
        with pytest.raises(ValueError, match="feedback bridge"):
            model.forecast(history=training, horizon=3)
    finally:
        handle.remove()
    assert calls == []
    assert [module.training for module in model.modules()] == modes
    for name, state in model.get_state().items():
        torch.testing.assert_close(state, stored[name], rtol=0, atol=0)
