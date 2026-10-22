"""Per-port transforms agree with public raw-family state and feedback execution."""

import copy
import io

import pytest
import torch

import resdag as rd
from resdag.ensemble import CoupledEnsembleESNModel
from resdag.models import classic_esn, parallel_esn


def _model(family):
    if family == "parallel":
        graph = parallel_esn(
            4,
            core_width=2,
            halo=1,
            reservoir_size=5,
            seed=12,
            dtype=torch.float64,
            readout="svd",
            readout_alpha=1e-5,
        )
    else:
        members = [
            classic_esn(
                5,
                4,
                4,
                seed=seed,
                dtype=torch.float64,
                readout="svd",
                readout_alpha=1e-5,
                readout_name=f"member_{seed}",
            )
            for seed in (12, 13)
        ]
        graph = members[0] if family == "symbolic" else CoupledEnsembleESNModel(members)
    return rd.Model(
        graph=graph,
        input_names=("state",),
        output_names=("prediction",),
        washout=2,
        preprocessing="off",
        input_transforms={"state": rd.Standardize(4)},
        output_transforms={"prediction": rd.Standardize(4)},
        feedback=[rd.Feedback("prediction", "state", delay=2)],
    )


def _training():
    values = torch.randn(2, 28, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(63))
    values = values * torch.arange(1, 5).double() + torch.arange(1, 5).double() * 8
    return values, 0.6 * values + 17


def _states_close(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("family", ["symbolic", "parallel", "coupled"])
def test_custom_transform_forward_and_explicit_state_gradients_match_raw_family(family):
    values, targets = _training()
    model = _model(family)
    input_transform, output_transform = model.input_scalers[0], model.output_scalers[0]
    parameters = tuple(map(id, model.parameters()))
    model.fit(values, targets).eval()
    assert tuple(map(id, model.parameters())) == parameters
    assert model.input_scalers[0] is input_transform
    assert model.output_scalers[0] is output_transform
    # Independent per-port fits force a physical-unit bridge rather than a
    # fortuitously shared normalization for the autoregressive connection.
    assert not torch.equal(input_transform.mean, output_transform.mean)
    raw = copy.deepcopy(model.graph).eval()
    raw.reset_reservoirs()
    source = values[:, -6:].clone().requires_grad_()
    reference = source.detach().clone().requires_grad_()
    expected = output_transform.inverse(raw(input_transform(reference)))
    actual = model(source)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    _states_close(model.get_state(), raw.get_state())
    actual_gradient = torch.autograd.grad(actual.square().sum(), source)[0]
    expected_gradient = torch.autograd.grad(expected.square().sum(), reference)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-11, atol=1e-11)
    assert bool((actual_gradient != 0).any())

    # Start both explicit-state paths from the same nonzero raw-family state.
    initial = {key: state.detach().clone() * 0.3 for key, state in raw.get_state().items()}
    stored = {key: state.clone() for key, state in model.get_state().items()}
    source = values[:, :4].clone().requires_grad_()
    reference = source.detach().clone().requires_grad_()
    raw_output, raw_final = raw.forward_stateless(input_transform(reference), states=initial)
    expected = output_transform.inverse(raw_output)
    actual, final = model.forward_stateless(source, states=initial)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    _states_close(final, raw_final)
    _states_close(model.get_state(), stored)
    actual_gradient = torch.autograd.grad(actual.square().sum(), source)[0]
    expected_gradient = torch.autograd.grad(expected.square().sum(), reference)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("family", ["symbolic", "parallel", "coupled"])
def test_delayed_custom_unit_bridge_matches_manual_raw_stateless_replay_and_checkpoint(family):
    values, targets = _training()
    model = _model(family).fit(values, targets).eval()
    raw = copy.deepcopy(model.graph).eval()
    history, horizon = values[:, -7:], 5
    incoming, outgoing = model.input_scalers[0], model.output_scalers[0]
    with torch.no_grad():
        warm, states = raw.forward_stateless(incoming(history))
        physical_warm = outgoing.inverse(warm)
        emissions = list(physical_warm.split(1, dim=1))
        expected = [emissions[-1]]
        for _ in range(horizon - 1):
            # Delay two consumes the emission two graph ticks earlier. It is
            # converted through physical units before entering the raw graph.
            raw_output, states = raw.forward_stateless(incoming(emissions[-2]), states=states)
            emission = outgoing.inverse(raw_output)
            emissions.append(emission)
            expected.append(emission)
        expected = torch.cat(expected, dim=1)
    actual = model.forecast(history, horizon=horizon)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
    _states_close(model.get_state(), states)

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = _model(family)
    parameters = tuple(map(id, restored.parameters()))
    restored.load_state_dict(torch.load(buffer, weights_only=True, map_location="cpu"))
    assert tuple(map(id, restored.parameters())) == parameters
    torch.testing.assert_close(
        restored.forecast(history, horizon=horizon), expected, rtol=1e-11, atol=1e-11
    )
