"""Independent regression/recurrence oracles for combined-prefix DAG fitting."""

import copy

import pytest
import torch
from torch import nn

import resdag as rd


class _ObservedRidge(rd.RidgeReadoutLayer):
    """Record the actual regression support without inspecting trainer internals."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fit_pairs = []

    def fit(self, states, targets):
        self.fit_pairs.append((states.detach().clone(), targets.detach().clone()))
        return super().fit(states, targets)


def _ridge(states, targets, *, alpha, bias, sigma=0.0):
    """Solve independently, including expected centered state-noise covariance."""
    x = states.reshape(-1, states.shape[-1])
    y = targets.reshape(-1, targets.shape[-1])
    if bias:
        mean_x, mean_y = x.mean(0), y.mean(0)
        centered_x, centered_y = x - mean_x, y - mean_y
    else:
        centered_x, centered_y = x, y
    penalty = alpha + (x.shape[0] - int(bias)) * sigma**2
    coefficients = torch.linalg.solve(
        centered_x.T @ centered_x + penalty * torch.eye(x.shape[-1], dtype=x.dtype),
        centered_x.T @ centered_y,
    )
    intercept = mean_y - mean_x @ coefficients if bias else None
    return coefficients.T, intercept


def _map(values, weight, bias):
    result = values @ weight.T
    return result if bias is None else result + bias


def _memory(recurrent, feedback):
    layer = rd.ESNLayer(
        len(recurrent),
        feedback_size=len(feedback[0]),
        activation="identity",
        bias=False,
        dtype=torch.float64,
    )
    with torch.no_grad():
        layer.weight_hh.copy_(torch.tensor(recurrent, dtype=torch.float64))
        layer.weight_feedback.copy_(torch.tensor(feedback, dtype=torch.float64))
    return layer


def _recurrence(values, recurrent, feedback):
    """Manual discrete dynamics, independent of reservoir forwarding/state APIs."""
    recurrent = torch.tensor(recurrent, dtype=values.dtype)
    feedback = torch.tensor(feedback, dtype=values.dtype)
    state = values.new_zeros(values.shape[0], recurrent.shape[0])
    result = []
    for sample in values.unbind(1):
        state = state @ recurrent.T + sample @ feedback.T
        result.append(state)
    return torch.stack(result, dim=1)


def _assert_readout(layer, expected, states, targets):
    weight, bias = expected
    torch.testing.assert_close(layer.weight, weight, rtol=2e-11, atol=2e-12)
    if bias is not None:
        torch.testing.assert_close(layer.bias, bias, rtol=2e-11, atol=2e-12)
    assert len(layer.fit_pairs) == 1
    actual_states, actual_targets = layer.fit_pairs[0]
    torch.testing.assert_close(actual_states, states, rtol=2e-11, atol=2e-12)
    torch.testing.assert_close(actual_targets, targets, rtol=0, atol=0)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("warmup", [0, 5])
@pytest.mark.parametrize("sigma", [0.0, 0.13])
def test_regularized_cascade_matches_manual_weights_states_and_noise(nested, warmup, sigma):
    source = rd.reservoir_input(1, dtype=torch.float64)
    specs = ((0.07, True, "a"), (0.11, False, "b"), (0.03, True, "c"))
    heads = [
        _ObservedRidge(1, 1, alpha=alpha, bias=bias, name=name).double()
        for alpha, bias, name in specs
    ]
    first = _memory([[0.7]], [[1.0]])
    second = _memory([[-0.2]], [[1.0]])
    if nested:
        output = heads[2](nn.Sequential(nn.Sequential(heads[0], first), heads[1], second)(source))
    else:
        output = heads[2](second(heads[1](first(heads[0](source)))))
    model = rd.ESNModel(source, output)
    sequence = torch.sin(torch.linspace(0.2, 6, 38, dtype=torch.float64)).reshape(2, 19, 1)
    sequence = sequence + torch.tensor([0.0, 2.0], dtype=torch.float64).reshape(2, 1, 1)
    train = sequence[:, warmup:]
    targets = (2 * train + 1, train.cos() - 3, 0.4 * train + train.sin())

    fitted_a = _ridge(train, targets[0], alpha=0.07, bias=True, sigma=sigma)
    states_a = _recurrence(_map(sequence, *fitted_a), [[0.7]], [[1.0]])
    fitted_b = _ridge(states_a[:, warmup:], targets[1], alpha=0.11, bias=False, sigma=sigma)
    states_b = _recurrence(_map(states_a, *fitted_b), [[-0.2]], [[1.0]])
    fitted_c = _ridge(states_b[:, warmup:], targets[2], alpha=0.03, bias=True, sigma=sigma)
    original_rng = torch.random.get_rng_state().clone()

    rd.ESNTrainer(model).fit(
        sequence[:, :warmup],
        train,
        dict(zip(("a", "b", "c"), targets)),
        state_noise=sigma,
    )

    _assert_readout(heads[0], fitted_a, train, targets[0])
    _assert_readout(heads[1], fitted_b, states_a[:, warmup:], targets[1])
    _assert_readout(heads[2], fitted_c, states_b[:, warmup:], targets[2])
    torch.testing.assert_close(first.state, states_a[:, -1], rtol=2e-11, atol=2e-12)
    torch.testing.assert_close(second.state, states_b[:, -1], rtol=2e-11, atol=2e-12)
    assert [head.state_noise for head in heads] == [0, 0, 0]
    assert torch.equal(torch.random.get_rng_state(), original_rng)


def test_shared_branch_merge_and_driver_match_independent_regressions():
    state, driver = (rd.reservoir_input(1, dtype=torch.float64) for _ in range(2))
    source_memory = _memory([[0.5]], [[0.8, -0.6]])
    shared = rd.Concatenate()(source_memory(rd.Concatenate()(state, driver)), state, driver)
    first = _ObservedRidge(3, 1, alpha=0.07, name="first").double()
    branch = _ObservedRidge(3, 2, alpha=0.11, name="branch").double()
    merged = rd.Concatenate()(first(shared), branch(shared))
    recurrent = [[0.15, 0.05], [-0.1, 0.35]]
    feedback = [[0.3, -0.2, 0.1], [-0.4, 0.5, 0.2]]
    downstream = _memory(recurrent, feedback)
    final = _ObservedRidge(5, 2, alpha=0.03, name="final").double()
    model = rd.ESNModel((state, driver), final(rd.Concatenate()(downstream(merged), merged)))
    warmup = 5
    values = torch.linspace(-1, 2, 38, dtype=torch.float64).reshape(2, 19, 1)
    driving = 0.4 * values.square() + values.cos()
    feature_state = _recurrence(torch.cat((values, driving), dim=-1), [[0.5]], [[0.8, -0.6]])
    source_features = torch.cat((feature_state, values, driving), dim=-1)
    train, train_driver = values[:, warmup:], driving[:, warmup:]
    first_target = 2 * train + train_driver
    branch_target = torch.cat(
        (-0.5 * train + train_driver, 0.2 * train + train_driver.square()), -1
    )
    final_target = torch.cat((train.sin() - train_driver, train.cos() + 0.3 * train_driver), -1)

    fit_first = _ridge(source_features[:, warmup:], first_target, alpha=0.07, bias=True)
    fit_branch = _ridge(source_features[:, warmup:], branch_target, alpha=0.11, bias=True)
    merge_features = torch.cat(
        (_map(source_features, *fit_first), _map(source_features, *fit_branch)), -1
    )
    last_state = _recurrence(merge_features, recurrent, feedback)
    final_features = torch.cat((last_state, merge_features), -1)
    fit_final = _ridge(final_features[:, warmup:], final_target, alpha=0.03, bias=True)

    rd.ESNTrainer(model).fit(
        (values[:, :warmup], driving[:, :warmup]),
        (train, train_driver),
        {"first": first_target, "branch": branch_target, "final": final_target},
    )

    _assert_readout(first, fit_first, source_features[:, warmup:], first_target)
    _assert_readout(branch, fit_branch, source_features[:, warmup:], branch_target)
    _assert_readout(final, fit_final, final_features[:, warmup:], final_target)
    torch.testing.assert_close(downstream.state, last_state[:, -1], rtol=2e-11, atol=2e-12)


class _PrefixSum(nn.Module):
    def forward(self, values):
        return values.cumsum(1)


@pytest.mark.parametrize("warmup", [0, 4])
def test_temporal_context_before_and_between_readouts_includes_warmup(warmup):
    source = rd.reservoir_input(1, dtype=torch.float64)
    first = _ObservedRidge(1, 1, alpha=0.1, bias=False, name="first").double()
    final = _ObservedRidge(1, 1, alpha=0.2, bias=False, name="final").double()
    model = rd.ESNModel(source, final(_PrefixSum()(first(_PrefixSum()(source)))))
    values = torch.linspace(0.2, 2, 26, dtype=torch.float64).reshape(2, 13, 1)
    first_features = values.cumsum(1)
    first_target = 2 * first_features[:, warmup:] + values[:, warmup:].sin()
    final_target = 0.3 * values[:, warmup:].sin() + 0.2 * values[:, warmup:]
    fit_first = _ridge(first_features[:, warmup:], first_target, alpha=0.1, bias=False)
    final_features = _map(first_features, *fit_first).cumsum(1)
    fit_final = _ridge(final_features[:, warmup:], final_target, alpha=0.2, bias=False)

    rd.ESNTrainer(model).fit(
        values[:, :warmup],
        values[:, warmup:],
        {"first": first_target, "final": final_target},
    )

    _assert_readout(first, fit_first, first_features[:, warmup:], first_target)
    _assert_readout(final, fit_final, final_features[:, warmup:], final_target)


def test_downstream_failure_restores_noise_modes_and_preserves_user_hooks():
    class FailingReadout(_ObservedRidge):
        def fit(self, states, targets):
            raise RuntimeError("independent downstream failure")

    source = rd.reservoir_input(1, dtype=torch.float64)
    first = _ObservedRidge(1, 1, name="first", state_noise=0.03).double()
    final = FailingReadout(1, 1, name="final", state_noise=0.07).double()
    model = rd.ESNModel(source, final(_PrefixSum()(first(source))))
    first.eval()
    final.train()
    original_modes = tuple(module.training for module in model.modules())
    user_hook = first.register_forward_pre_hook(lambda module, args: None)
    original_hooks = dict(first._forward_pre_hooks)
    values = torch.linspace(0.2, 2, 13, dtype=torch.float64).reshape(1, 13, 1)
    try:
        with pytest.raises(RuntimeError, match="independent downstream failure"):
            rd.ESNTrainer(model).fit(
                values[:, :4],
                values[:, 4:],
                {"first": values[:, 4:], "final": values[:, 4:]},
                state_noise=0.13,
            )
        assert first.state_noise == 0.03
        assert final.state_noise == 0.07
        assert dict(first._forward_pre_hooks) == original_hooks
        assert not final._forward_pre_hooks
        assert tuple(module.training for module in model.modules()) == original_modes
    finally:
        user_hook.remove()


def test_user_hooks_change_features_and_outputs_seen_by_downstream_fitting():
    source = rd.reservoir_input(1, dtype=torch.float64)
    first = _ObservedRidge(1, 1, alpha=0.07, name="first").double()
    final = _ObservedRidge(1, 1, alpha=0.03, name="final").double()
    model = rd.ESNModel(source, final(_PrefixSum()(first(source))))
    before = first.register_forward_pre_hook(lambda module, args: (2 * args[0],))
    after = first.register_forward_hook(lambda module, args, output: output + 4)
    original_pre_hooks = dict(first._forward_pre_hooks)
    original_post_hooks = dict(first._forward_hooks)
    values = torch.linspace(0.2, 2, 26, dtype=torch.float64).reshape(2, 13, 1)
    warmup = 4
    first_target = 3 * values[:, warmup:] + 1
    final_target = 7 * values[:, warmup:] + values[:, warmup:].sin()
    fit_first = _ridge(2 * values[:, warmup:], first_target, alpha=0.07, bias=True)
    final_features = (_map(2 * values, *fit_first) + 4).cumsum(1)
    fit_final = _ridge(final_features[:, warmup:], final_target, alpha=0.03, bias=True)
    try:
        rd.ESNTrainer(model).fit(
            values[:, :warmup],
            values[:, warmup:],
            {"first": first_target, "final": final_target},
        )
        _assert_readout(first, fit_first, 2 * values[:, warmup:], first_target)
        _assert_readout(final, fit_final, final_features[:, warmup:], final_target)
        assert dict(first._forward_pre_hooks) == original_pre_hooks
        assert dict(first._forward_hooks) == original_post_hooks
    finally:
        before.remove()
        after.remove()


@pytest.mark.parametrize("nested", [False, True])
def test_one_readout_at_two_distinct_positions_is_rejected_before_fit(nested):
    source = rd.reservoir_input(1, dtype=torch.float64)
    shared = _ObservedRidge(1, 1, alpha=0, bias=False, name="shared").double()
    output = nn.Sequential(shared, shared)(source) if nested else shared(shared(source))
    model = rd.ESNModel(source, output)
    weights = shared.weight.detach().clone()
    values = torch.arange(1.0, 11.0, dtype=torch.float64).reshape(1, 10, 1)
    with pytest.raises(ValueError, match="shared.*positions|invoked more than once"):
        rd.ESNTrainer(model).fit(values[:, :3], values[:, 3:], 2 * values[:, 3:])
    assert shared.fit_pairs == []
    assert not shared.is_fitted
    torch.testing.assert_close(shared.weight, weights, rtol=0, atol=0)
    assert not shared._forward_pre_hooks


def _assert_state_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_state_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for item, prior in zip(actual, expected):
            _assert_state_equal(item, prior)
    else:
        assert actual == expected


@pytest.mark.parametrize("framewise", [False, True])
@pytest.mark.parametrize("refit", [False, True])
def test_public_failed_fit_with_inplace_layer_preserves_caller_and_model(framewise, refit):
    class InPlace(nn.Module):
        def __init__(self):
            super().__init__()
            self.supports_framewise = framewise

        def forward(self, values):
            return values.add_(7)

    class ConditionalReadout(rd.RidgeReadoutLayer):
        fail = False

        def fit(self, states, targets):
            if self.fail:
                raise RuntimeError("independent public failure")
            return super().fit(states, targets)

    source = rd.reservoir_input(1, dtype=torch.float64)
    head = ConditionalReadout(1, 1, alpha=0.01).double()
    model = rd.Model(
        inputs={"state": source},
        outputs={"prediction": head(InPlace()(source))},
        preprocessing="off",
        washout=3,
    )
    if refit:
        initial = torch.arange(20.0, dtype=torch.float64).reshape(1, 20, 1)
        model.fit(initial.clone(), targets=2 * initial)
    head.fail = True
    original_model_state = copy.deepcopy(model.state_dict())
    original_parameters = tuple(id(value) for value in model.parameters())
    base = torch.arange(52.0, dtype=torch.float64).reshape(2, 13, 2)
    values = base[..., :1]
    targets = 2 * values + 1
    original_base, original_targets = base.clone(), targets.clone()
    with pytest.raises(RuntimeError, match="independent public failure"):
        model.fit(values, targets=targets)
    torch.testing.assert_close(base, original_base, rtol=0, atol=0)
    torch.testing.assert_close(targets, original_targets, rtol=0, atol=0)
    _assert_state_equal(model.state_dict(), original_model_state)
    assert tuple(id(value) for value in model.parameters()) == original_parameters
    assert model.is_fitted == refit
    assert not head._forward_pre_hooks
