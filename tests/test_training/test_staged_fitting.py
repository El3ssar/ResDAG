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


def test_direct_readout_chain_preserves_single_warmup_and_training_pass():
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
    assert calls == [3, 7]
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
