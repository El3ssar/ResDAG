"""Explicit DAG execution releases dead features without changing semantics."""

from __future__ import annotations

import copy
import weakref

import pytest
import torch
from torch import nn

import resdag as rd


class _PrefixSum(nn.Module):
    def forward(self, values):
        return values.cumsum(dim=1)


def _branched_graph(*, temporal=False):
    source = rd.reservoir_input(2, dtype=torch.float64)
    encoded = nn.Linear(2, 3, dtype=torch.float64)(source)
    features = _PrefixSum()(encoded) if temporal else encoded
    left = rd.ESNLayer(5, feedback_size=3, trainable=True, seed=3, dtype=torch.float64)(features)
    right = rd.ESNLayer(4, feedback_size=3, trainable=True, seed=7, dtype=torch.float64)(features)
    left = nn.Linear(5, 2, dtype=torch.float64)(left)
    right = nn.Linear(4, 2, dtype=torch.float64)(right)
    skip = nn.Linear(3, 2, dtype=torch.float64)(encoded)
    graph = rd.ESNModel(source, (left + right + skip, nn.Tanh()(left - right)))
    for module in graph.modules():
        if isinstance(module, rd.ESNLayer):
            module.detach_state_between_calls = False
    return graph


@pytest.mark.parametrize("temporal", [False, True])
@pytest.mark.parametrize("batch", [1, 3])
def test_values_final_states_and_gradients_match_ordinary_branched_graph(temporal, batch):
    torch.manual_seed(17)
    graph = _branched_graph(temporal=temporal)
    reference = copy.deepcopy(graph)
    inputs = torch.randn(batch, 7, 2, dtype=torch.float64, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()
    states = {
        name: torch.randn(batch, module.cell.state_size, dtype=torch.float64, requires_grad=True)
        for name, module in graph.named_modules()
        if isinstance(module, rd.ESNLayer)
    }
    reference_states = {
        name: value.detach().clone().requires_grad_() for name, value in states.items()
    }
    original_states = {name: value.detach().clone() for name, value in states.items()}
    reference.set_reservoir_states(reference_states)
    expected = reference(reference_inputs)
    expected_states = reference.get_reservoir_states()
    actual, final = graph.forward_stateless(inputs, states=states)
    for result, wanted in zip(actual, expected):
        torch.testing.assert_close(result, wanted, rtol=0, atol=0)
    for name, state in final.items():
        torch.testing.assert_close(state, expected_states[name], rtol=0, atol=0)
        torch.testing.assert_close(states[name], original_states[name], rtol=0, atol=0)
    assert all(
        module.state is None for module in graph.modules() if isinstance(module, rd.ESNLayer)
    )
    sum(value.square().sum() for value in (*actual, *final.values())).backward()
    sum(value.square().sum() for value in (*expected, *expected_states.values())).backward()
    torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=1e-13, atol=1e-13)
    for name, state in states.items():
        torch.testing.assert_close(state.grad, reference_states[name].grad, rtol=1e-13, atol=1e-13)
    expected_parameters = dict(reference.named_parameters())
    for name, parameter in graph.named_parameters():
        assert parameter.grad is not None
        torch.testing.assert_close(
            parameter.grad, expected_parameters[name].grad, rtol=1e-13, atol=1e-13
        )


def test_explicit_chunk_continuation_keeps_values_states_and_upstream_gradients():
    torch.manual_seed(23)
    graph = _branched_graph()
    reference = copy.deepcopy(graph)
    data = torch.randn(2, 9, 2, dtype=torch.float64, requires_grad=True)
    full_data = data.detach().clone().requires_grad_()
    first, state = graph.forward_stateless(data[:, :4])
    second, final = graph.forward_stateless(data[:, 4:], states=state)
    actual = tuple(torch.cat((a, b), dim=1) for a, b in zip(first, second))
    expected, expected_final = reference.forward_stateless(full_data)
    for result, wanted in zip(actual, expected):
        torch.testing.assert_close(result, wanted, rtol=1e-13, atol=1e-13)
    for name, value in final.items():
        torch.testing.assert_close(value, expected_final[name], rtol=1e-13, atol=1e-13)
    sum(value.square().sum() for value in actual).backward()
    sum(value.square().sum() for value in expected).backward()
    torch.testing.assert_close(data.grad, full_data.grad, rtol=1e-12, atol=1e-12)
    assert data.grad[:, :4].abs().sum() > 0


def test_dead_feature_tensor_is_freed_before_later_operations_in_no_grad_mode():
    observed = []
    reference = {}

    class Inspect(nn.Module):
        def forward(self, values):
            if "feature" in reference:
                observed.append(reference["feature"]() is None)
            return values

    class TrackedLinear(nn.Linear):
        def forward(self, values):
            output = super().forward(values)
            reference["feature"] = weakref.ref(output)
            return output

    source = rd.reservoir_input(2, dtype=torch.float64)
    encoded = TrackedLinear(2, 16, dtype=torch.float64)
    hidden = nn.Tanh()(encoded(source))
    graph = rd.ESNModel(source, nn.Linear(16, 2, dtype=torch.float64)(Inspect()(hidden)))
    observed.clear()
    with torch.no_grad():
        graph.forward_stateless(torch.randn(2, 7, 2, dtype=torch.float64))
    assert observed == [True]


def test_shared_consumers_and_retained_outputs_are_not_released_early():
    reference = {}
    observed = []

    class Inspect(nn.Module):
        def forward(self, values):
            if "feature" in reference:
                observed.append(reference["feature"]() is not None)
            return values.square()

    class TrackedLinear(nn.Linear):
        def forward(self, values):
            output = super().forward(values)
            reference["feature"] = weakref.ref(output)
            return output

    source = rd.reservoir_input(2, dtype=torch.float64)
    encoder = TrackedLinear(2, 4, dtype=torch.float64)
    hidden = encoder(source)
    graph = rd.ESNModel(source, (nn.Tanh()(hidden), Inspect()(hidden), hidden))
    observed.clear()
    with torch.no_grad():
        actual, _ = graph.forward_stateless(torch.randn(2, 5, 2, dtype=torch.float64))
    assert observed == [True]
    assert reference["feature"]() is actual[2]
    torch.testing.assert_close(actual[0], actual[2].tanh(), rtol=0, atol=0)
    torch.testing.assert_close(actual[1], actual[2].square(), rtol=0, atol=0)


def test_multioutput_views_keep_alias_storage_and_gradients_after_parent_release():
    class Split(nn.Module):
        def forward(self, values):
            return values.chunk(2, dim=-1)

    source = rd.reservoir_input(4, dtype=torch.float64)
    left, right = Split()(source)
    graph = rd.ESNModel(source, (left, right, left + right))
    data = torch.randn(2, 5, 4, dtype=torch.float64, requires_grad=True)
    actual, final = graph.forward_stateless(data)
    expected = data.chunk(2, dim=-1)
    assert final == {}
    for result, wanted in zip(actual[:2], expected):
        assert result.untyped_storage().data_ptr() == data.untyped_storage().data_ptr()
        torch.testing.assert_close(result, wanted, rtol=0, atol=0)
    torch.testing.assert_close(actual[2], expected[0] + expected[1], rtol=0, atol=0)
    sum(value.sum() for value in actual).backward()
    torch.testing.assert_close(data.grad, torch.full_like(data, 2), rtol=0, atol=0)
