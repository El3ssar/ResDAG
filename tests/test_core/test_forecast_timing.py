"""Exact sample indices across core, ensemble, preparation and continuation."""

import pytest
import torch

from resdag.core import ESNModel, Input
from resdag.data import prepare_esn_data
from resdag.ensemble import CoupledEnsembleESNModel
from resdag.layers import CGReadoutLayer, Concatenate, ESNLayer
from resdag.models.parallel_esn import ParallelESNModel


def _increment_model(*, driven=False):
    feedback = Input((5, 1))
    driver = Input((5, 1)) if driven else None
    reservoir = ESNLayer(4, feedback_size=1, spectral_radius=0.5, seed=2)
    state = reservoir(feedback)
    features = Concatenate()(feedback, driver, state) if driven else Concatenate()(feedback, state)
    readout = CGReadoutLayer(6 if driven else 5, 1, bias=True, name="output")
    output = readout(features)
    model = ESNModel((feedback, driver) if driven else feedback, output)
    with torch.no_grad():
        readout.weight.zero_()
        readout.weight[0, 0] = 1
        if driven:
            readout.weight[0, 1] = 1
        readout.bias.fill_(0 if driven else 1)
        readout._set_fitted(True)
    return model


def _parallel_increment_model():
    reservoir = ESNLayer(1, feedback_size=1, activation="identity", bias=False, seed=2)
    readout = CGReadoutLayer(1, 1, bias=True)
    with torch.no_grad():
        reservoir.cell.weight_hh.zero_()
        reservoir.cell.weight_feedback.fill_(1)
        readout.weight.fill_(1)
        readout.bias.fill_(1)
        readout._set_fitted(True)
    return ParallelESNModel(reservoir, readout, input_dim=2, core_width=1, halo=0)


@pytest.fixture(params=["core", "coupled", "parallel"])
def model(request):
    if request.param == "coupled":
        return CoupledEnsembleESNModel([_increment_model(), _increment_model()])
    if request.param == "parallel":
        return _parallel_increment_model()
    return _increment_model()


def _history(model):
    width = 2 if isinstance(model, ParallelESNModel) else 1
    return torch.arange(5, dtype=torch.float32).view(1, 5, 1).expand(2, 5, width).clone()


def test_integer_sample_oracle_and_seam(model):
    observed = _history(model)
    expected = (
        torch.arange(5, 9, dtype=torch.float32).view(1, 4, 1).expand(2, 4, observed.shape[-1])
    )
    assert torch.equal(model.forecast(observed, horizon=4), expected)
    full = model.forecast(observed, horizon=4, return_warmup=True)
    assert torch.equal(full, torch.cat((observed[:, 1:], expected), dim=1))
    legacy = model.forecast(observed, horizon=4, alignment="legacy")
    assert torch.equal(legacy, expected + 1)


@pytest.mark.parametrize("horizon", [0, -1, True, 1.5])
def test_all_forecast_families_reject_invalid_counts(model, horizon):
    with pytest.raises(ValueError, match="horizon must be a positive integer"):
        model.forecast(_history(model), horizon=horizon)


def test_empty_history_continuation_consumes_final_prediction_once(model):
    observed = _history(model)
    full = model.forecast(observed, horizon=7)
    final_states = model.get_state()
    first = model.forecast(observed, horizon=3)
    continued = model.forecast(
        observed[:, :0], horizon=4, initial_feedback=first[:, -1:], reset=False
    )
    assert torch.equal(torch.cat((first, continued), dim=1), full)
    for name, state in model.get_state().items():
        torch.testing.assert_close(state, final_states[name], atol=1e-6, rtol=1e-6)
    with pytest.raises(ValueError, match="empty warmup"):
        model.forecast(observed[:, :0], horizon=1)


def test_explicit_seed_is_additional_observation(model):
    observed = _history(model)
    seed = torch.full_like(observed[:, :1], 20)
    expected = torch.full_like(seed, 21)
    assert torch.equal(model.forecast(observed, horizon=1, initial_feedback=seed), expected)
    full = model.forecast(observed, horizon=1, initial_feedback=seed, return_warmup=True)
    assert full.shape[1] == observed.shape[1] + 1
    assert torch.equal(full[:, -1:], expected)


@pytest.mark.parametrize("kind", ["core", "coupled"])
def test_exact_driver_pairing_and_one_step_without_future_driver(kind):
    models = [_increment_model(driven=True), _increment_model(driven=True)]
    model = models[0] if kind == "core" else CoupledEnsembleESNModel(models)
    # x[t+1] = x[t] + d[t], including at the warmup/forecast seam.
    feedback = torch.tensor([0.0, 1.0, 3.0, 6.0]).view(1, 4, 1)
    driver = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(1, 4, 1)
    future = torch.tensor([5.0, 6.0]).view(1, 2, 1)
    assert torch.equal(model.forecast((feedback, driver), horizon=1), torch.tensor([[[10.0]]]))
    actual = model.forecast((feedback, driver), future, horizon=3)
    assert torch.equal(actual, torch.tensor([[[10.0], [15.0], [21.0]]]))
    with pytest.raises(ValueError, match="at least 2"):
        model.forecast((feedback, driver), future[:, :1], horizon=3)
    seed = torch.tensor([[[10.0]]])
    explicit = model.forecast((feedback, driver), future, horizon=2, initial_feedback=seed)
    assert torch.equal(explicit, torch.tensor([[[15.0], [21.0]]]))


@pytest.mark.parametrize("alignment", ["next", "legacy"])
def test_prepared_history_forecast_is_exactly_held_out_validation(alignment):
    data = torch.arange(50, dtype=torch.float32).view(1, 50, 1)
    _, train, target, history, val = prepare_esn_data(
        data, warmup_steps=5, train_steps=20, val_steps=10, alignment=alignment
    )
    assert torch.equal(target, train + 1)
    assert history[0, -1, 0] == (25 if alignment == "next" else 24)
    assert val[0, 0, 0] == 26  # final supervised target 25 stays out of validation
    assert torch.equal(_increment_model().forecast(history, horizon=10, alignment=alignment), val)
