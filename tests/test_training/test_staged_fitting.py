"""Independent recurrence oracles for fitting readouts separated by memory."""

import pytest
import torch
from torch import nn

import resdag as rd
from resdag.layers import IncrementalRidgeReadoutLayer, RidgeReadoutLayer


def _linear_memory(gain: float) -> rd.ESNLayer:
    layer = rd.ESNLayer(1, feedback_size=1, activation="identity", bias=False, dtype=torch.float64)
    with torch.no_grad():
        layer.weight_hh.fill_(gain)
        layer.weight_feedback.fill_(1)
    return layer


def _recurrence(sequence: torch.Tensor, gain: float) -> torch.Tensor:
    state = torch.zeros_like(sequence[:, 0])
    values = []
    for sample in sequence.unbind(1):
        state = gain * state + sample
        values.append(state)
    return torch.stack(values, dim=1)


@pytest.mark.parametrize("nested_upstream", [False, True])
@pytest.mark.parametrize("downstream_cls", [RidgeReadoutLayer, IncrementalRidgeReadoutLayer])
def test_recurrent_cascade_matches_known_stagewise_solution(nested_upstream, downstream_cls):
    x = rd.reservoir_input(1, dtype=torch.float64)
    upstream = RidgeReadoutLayer(1, 1, alpha=0, bias=False, name="upstream").double()
    middle = downstream_cls(1, 1, alpha=0, bias=False, name="middle").double()
    final = RidgeReadoutLayer(1, 1, alpha=0, bias=False, name="output").double()
    first, second = _linear_memory(0.7), _linear_memory(0.3)
    upstream_node = nn.Sequential(upstream)(x) if nested_upstream else upstream(x)
    model = rd.ESNModel(x, final(second(middle(first(upstream_node)))))
    series = torch.linspace(0.2, 2, 26, dtype=torch.float64).reshape(2, 13, 1)
    warmup, train = series[:, :4], series[:, 4:]
    first_expected = _recurrence(2 * series, 0.7)
    second_expected = _recurrence(3 * first_expected, 0.3)
    rd.ESNTrainer(model).fit(
        warmup,
        train,
        {
            "upstream": 2 * train,
            "middle": 3 * first_expected[:, 4:],
            "output": 4 * second_expected[:, 4:],
        },
    )
    torch.testing.assert_close(upstream.weight, torch.full_like(upstream.weight, 2))
    torch.testing.assert_close(middle.weight, torch.full_like(middle.weight, 3))
    torch.testing.assert_close(final.weight, torch.full_like(final.weight, 4))
    torch.testing.assert_close(first.state, first_expected[:, -1])
    torch.testing.assert_close(second.state, second_expected[:, -1])
    assert all(not module._forward_pre_hooks for module in (upstream, middle, final))


def test_direct_readout_chain_uses_one_combined_warmup_training_prefix():
    x = rd.reservoir_input(1, dtype=torch.float64)
    upstream = RidgeReadoutLayer(1, 1, alpha=0, bias=False, name="upstream").double()
    final = RidgeReadoutLayer(1, 1, alpha=0, bias=False, name="output").double()
    model = rd.ESNModel(x, final(upstream(x)))
    calls = []
    handle = model.register_forward_pre_hook(lambda module, args: calls.append(args[0].shape[1]))
    warmup = torch.ones(1, 3, 1, dtype=torch.float64)
    train = torch.linspace(0.1, 1, 7, dtype=torch.float64).reshape(1, 7, 1)
    try:
        rd.ESNTrainer(model).fit(warmup, train, {"upstream": 2 * train, "output": 6 * train})
    finally:
        handle.remove()
    assert calls == [10]
    torch.testing.assert_close(upstream.weight, torch.full_like(upstream.weight, 2))
    torch.testing.assert_close(final.weight, torch.full_like(final.weight, 3))


def test_staged_fit_removes_hooks_and_restores_overrides_after_failure():
    class FailingReadout(RidgeReadoutLayer):
        def fit(self, states, targets):
            raise RuntimeError("intentional stage failure")

    x = rd.reservoir_input(1, dtype=torch.float64)
    upstream = RidgeReadoutLayer(1, 1, name="upstream").double()
    final = FailingReadout(1, 1, name="output").double()
    model = rd.ESNModel(x, final(_linear_memory(0.5)(upstream(x))))
    warmup = torch.ones(1, 3, 1, dtype=torch.float64)
    train = torch.ones(1, 7, 1, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="intentional stage failure"):
        rd.ESNTrainer(model).fit(
            warmup, train, {"upstream": train, "output": train}, state_noise=0.01
        )
    assert upstream.state_noise == final.state_noise == 0
    assert not upstream._forward_pre_hooks and not final._forward_pre_hooks


def test_a_shared_readout_cannot_be_fitted_in_different_recurrent_stages():
    x = rd.reservoir_input(1, dtype=torch.float64)
    shared = RidgeReadoutLayer(1, 1, name="output").double()
    memory = _linear_memory(0.5)
    model = rd.ESNModel(x, shared(memory(shared(x))))
    data = torch.ones(1, 3, 1, dtype=torch.float64)
    with pytest.raises(ValueError, match="shared across different recurrent fitting stages"):
        rd.ESNTrainer(model).fit(data, data, data)
    assert memory.state is None
    assert not shared._forward_pre_hooks


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("downstream_cls", [RidgeReadoutLayer, IncrementalRidgeReadoutLayer])
def test_sequential_cascade_matches_explicit_graph_and_recurrence_oracle(nested, downstream_cls):
    series = torch.linspace(0.2, 2, 26, dtype=torch.float64).reshape(2, 13, 1)
    warmup, train = series[:, :4], series[:, 4:]
    first_expected = _recurrence(2 * series, 0.7)
    second_expected = _recurrence(3 * first_expected, 0.3)
    results = []
    for sequential in (False, True):
        x = rd.reservoir_input(1, dtype=torch.float64)
        upstream = RidgeReadoutLayer(1, 1, alpha=0, bias=False, name="upstream").double()
        middle = downstream_cls(1, 1, alpha=0, bias=False, name="middle").double()
        final = RidgeReadoutLayer(1, 1, alpha=0, bias=False, name="output").double()
        first, second = _linear_memory(0.7), _linear_memory(0.3)
        with torch.no_grad():
            upstream.weight.fill_(0.25)
            middle.weight.fill_(0.5)
        if sequential:
            block = (
                nn.Sequential(nn.Sequential(upstream, first), nn.Identity(), middle, second)
                if nested
                else nn.Sequential(upstream, first, middle, second)
            )
            output = final(block(x))
        else:
            output = final(second(middle(first(upstream(x)))))
        model = rd.ESNModel(x, output)
        rd.ESNTrainer(model).fit(
            warmup,
            train,
            {
                "upstream": 2 * train,
                "middle": 3 * first_expected[:, 4:],
                "output": 4 * second_expected[:, 4:],
            },
        )
        for readout, gain in ((upstream, 2), (middle, 3), (final, 4)):
            torch.testing.assert_close(readout.weight, torch.full_like(readout.weight, gain))
            assert not readout._forward_pre_hooks
        torch.testing.assert_close(first.state, first_expected[:, -1])
        torch.testing.assert_close(second.state, second_expected[:, -1])
        results.append((upstream.weight, middle.weight, final.weight, first.state, second.state))
    for explicit, sequential in zip(*results):
        torch.testing.assert_close(explicit, sequential)


@pytest.mark.parametrize("overridden_sequential", [False, True])
def test_opaque_mixed_composite_is_rejected_before_any_training_mutation(overridden_sequential):
    class OpaqueCascade(nn.Module):
        def __init__(self, upstream, memory):
            super().__init__()
            self.upstream, self.memory = upstream, memory

        def forward(self, inputs):
            return self.memory(self.upstream(inputs))

    class OverriddenSequential(nn.Sequential):
        def forward(self, inputs):
            return super().forward(inputs)

    x = rd.reservoir_input(1, dtype=torch.float64)
    upstream = RidgeReadoutLayer(1, 1, name="upstream").double()
    final = RidgeReadoutLayer(1, 1, name="output").double()
    memory = _linear_memory(0.7)
    wrapper = OverriddenSequential if overridden_sequential else OpaqueCascade
    model = rd.ESNModel(x, final(wrapper(upstream, memory)(x)))
    memory.set_state(torch.full((1, 1), 7.0, dtype=torch.float64))
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    original_state = memory.state.clone()
    original_modes = [module.training for module in model.modules()]
    calls = []
    handle = model.register_forward_pre_hook(lambda module, args: calls.append(True))
    data = torch.ones(1, 3, 1, dtype=torch.float64)
    try:
        with pytest.raises(ValueError, match="separate functional graph nodes.*Sequential"):
            rd.ESNTrainer(model).fit(
                data, data, {"upstream": 2 * data, "output": 3 * data}, state_noise=0.01
            )
    finally:
        handle.remove()
    assert not calls
    torch.testing.assert_close(memory.state, original_state, rtol=0, atol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
    assert [module.training for module in model.modules()] == original_modes
    assert not upstream.is_fitted and not final.is_fitted
    assert upstream.state_noise == final.state_noise == 0
    assert not upstream._forward_pre_hooks and not final._forward_pre_hooks
