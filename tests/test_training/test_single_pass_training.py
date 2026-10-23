"""One complete-prefix fitting traversal with independent recurrence oracles."""

import copy
import io
import weakref
from types import MethodType

import pytest
import torch
from torch import nn

import resdag as rd


class CountingReadout(rd.RidgeReadoutLayer):
    def __init__(self, name):
        super().__init__(1, 1, alpha=0, bias=False, name=name)
        self.fit_calls = []

    def fit(self, states, targets):
        self.fit_calls.append((tuple(states.shape), tuple(targets.shape)))
        return super().fit(states, targets)


def _memory(gain):
    layer = rd.ESNLayer(1, feedback_size=1, activation="identity", bias=False, dtype=torch.float64)
    with torch.no_grad():
        layer.cell.weight_hh.fill_(gain)
        layer.cell.weight_feedback.fill_(1)
    return layer


def _scan(values, gain):
    state = torch.zeros_like(values[:, 0])
    outputs = []
    for sample in values.unbind(1):
        state = gain * state + sample
        outputs.append(state)
    return torch.stack(outputs, 1)


def _series():
    return torch.linspace(0.2, 2, 26, dtype=torch.float64).reshape(2, 13, 1)


@pytest.mark.parametrize("depth", [1, 2, 4, 8])
def test_cascade_fits_every_head_once_and_propagates_each_reservoir_once(depth):
    source = rd.reservoir_input(1, dtype=torch.float64)
    node = source
    layers, heads, expected_states, targets = [], [], [], {}
    values = _series()
    propagated = values
    for index in range(depth):
        recurrent_gain, readout_gain = 0.1 + 0.05 * index, 1.2 + 0.1 * index
        memory = _memory(recurrent_gain)
        head = CountingReadout(f"head_{index}").double()
        with torch.no_grad():
            head.weight.fill_(-0.25 * (index + 1))
        propagated = _scan(propagated, recurrent_gain)
        expected_states.append(propagated[:, -1].clone())
        propagated = propagated * readout_gain
        targets[head.name] = propagated[:, 4:]
        node = head(memory(node))
        layers.append(memory)
        heads.append((head, readout_gain))
    model = rd.ESNModel(source, node)
    calls = [[] for _ in layers]
    handles = [
        layer.register_forward_pre_hook(
            lambda module, args, observed=observed: observed.append(tuple(args[0].shape))
        )
        for layer, observed in zip(layers, calls)
    ]
    try:
        rd.ESNTrainer(model).fit(values[:, :4], values[:, 4:], targets)
    finally:
        for handle in handles:
            handle.remove()
    assert calls == [[(2, 13, 1)] for _ in layers]
    for (head, gain), layer, state in zip(heads, layers, expected_states):
        assert head.fit_calls == [((2, 9, 1), (2, 9, 1))]
        torch.testing.assert_close(head.weight, torch.full_like(head.weight, gain))
        torch.testing.assert_close(layer.state, state)


def test_temporal_features_include_the_entire_warmup_prefix():
    class Cumulative(nn.Module):
        def forward(self, values):
            return values.cumsum(1)

    source = rd.reservoir_input(1, dtype=torch.float64)
    head = rd.RidgeReadoutLayer(1, 1, alpha=0, name="output").double()
    model = rd.ESNModel(source, head(Cumulative()(source)))
    values = torch.arange(1, 7, dtype=torch.float64).reshape(1, 6, 1)
    targets = 2 * values.cumsum(1)[:, 2:]
    rd.ESNTrainer(model).fit(values[:, :2], values[:, 2:], targets)
    torch.testing.assert_close(head.weight, torch.tensor([[2.0]], dtype=torch.float64))
    torch.testing.assert_close(head.bias, torch.zeros_like(head.bias), atol=1e-12, rtol=0)
    torch.testing.assert_close(model(values)[:, 2:], targets)


def test_nested_sequential_preserves_hooks_and_effective_readout_features():
    source = rd.reservoir_input(1, dtype=torch.float64)
    upstream, final = CountingReadout("upstream").double(), CountingReadout("output").double()
    memory = _memory(0.3)
    inner = nn.Sequential(upstream, memory)
    outer = nn.Sequential(inner, nn.Identity(), final)
    model = rd.ESNModel(source, outer(source))
    values = _series()
    expected_features = _scan(6 * values, 0.3)
    calls = {name: [] for name in ("model", "outer", "inner", "upstream", "memory", "final")}
    handles = []
    for name, module in (
        ("model", model),
        ("outer", outer),
        ("inner", inner),
        ("upstream", upstream),
        ("memory", memory),
        ("final", final),
    ):
        handles.append(
            module.register_forward_pre_hook(
                lambda module, args, name=name: calls[name].append(tuple(args[0].shape))
            )
        )
    # Existing user pre-hooks must run before fitting consumes the features.
    handles.append(upstream.register_forward_pre_hook(lambda module, args: (2 * args[0],)))
    try:
        rd.ESNTrainer(model).fit(
            values[:, :4],
            values[:, 4:],
            {"upstream": 6 * values[:, 4:], "output": 4 * expected_features[:, 4:]},
        )
    finally:
        for handle in handles:
            handle.remove()
    assert calls == {name: [(2, 13, 1)] for name in calls}
    torch.testing.assert_close(upstream.weight, torch.full_like(upstream.weight, 3))
    torch.testing.assert_close(final.weight, torch.full_like(final.weight, 4))
    torch.testing.assert_close(memory.state, expected_features[:, -1])
    assert len(upstream.fit_calls) == len(final.fit_calls) == 1


def test_shared_feature_branch_is_computed_once_for_independent_heads():
    class ObservedScale(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, values):
            self.calls.append(tuple(values.shape))
            return 2 * values

    source = rd.reservoir_input(1, dtype=torch.float64)
    memory, scale = _memory(0.2), ObservedScale()
    features = scale(memory(source))
    first, second = CountingReadout("first").double(), CountingReadout("second").double()
    model = rd.ESNModel(source, (first(features), second(features)))
    scale.calls.clear()
    values = _series()
    expected = 2 * _scan(values, 0.2)
    rd.ESNTrainer(model).fit(
        values[:, :4],
        values[:, 4:],
        {"first": 3 * expected[:, 4:], "second": 5 * expected[:, 4:]},
    )
    assert scale.calls == [(2, 13, 1)]
    assert len(first.fit_calls) == len(second.fit_calls) == 1
    torch.testing.assert_close(first.weight, torch.full_like(first.weight, 3))
    torch.testing.assert_close(second.weight, torch.full_like(second.weight, 5))


def test_repeated_readout_instance_is_rejected_before_training_mutation():
    source = rd.reservoir_input(1, dtype=torch.float64)
    memory, shared = _memory(0.2), CountingReadout("output").double()
    first = shared(memory(source))
    model = rd.ESNModel(source, shared(nn.Identity()(first)))
    memory.set_state(torch.full((2, 1), 7.0, dtype=torch.float64))
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    state = memory.state.clone()
    original_forward = model.forward
    values = _series()
    with pytest.raises(ValueError, match="[Rr]eadout|instance|shared|repeated"):
        rd.ESNTrainer(model).fit(values[:, :4], values[:, 4:], values[:, 4:])
    assert not shared.fit_calls
    assert model.forward is original_forward
    torch.testing.assert_close(memory.state, state, rtol=0, atol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, weights[name], rtol=0, atol=0)


def test_failure_restores_forward_user_hooks_and_noise_settings():
    class FailingReadout(CountingReadout):
        def fit(self, states, targets):
            raise RuntimeError("intentional single-pass failure")

    source = rd.reservoir_input(1, dtype=torch.float64)
    upstream, final = CountingReadout("upstream").double(), FailingReadout("output").double()
    model = rd.ESNModel(source, final(_memory(0.2)(upstream(source))))
    upstream.state_noise, final.state_noise = 0.02, 0.03
    upstream.noise_draws, final.noise_draws = 2, 3
    upstream.noise_seed, final.noise_seed = 17, 19
    noise = [(head.state_noise, head.noise_draws, head.noise_seed) for head in (upstream, final)]
    original_forward = model.forward
    observed = []
    handle = upstream.register_forward_pre_hook(lambda module, args: observed.append(True))
    user_hook_keys = tuple(upstream._forward_pre_hooks)
    values = _series()
    try:
        with pytest.raises(RuntimeError, match="intentional single-pass failure"):
            rd.ESNTrainer(model).fit(
                values[:, :4],
                values[:, 4:],
                {"upstream": values[:, 4:], "output": values[:, 4:]},
                state_noise=0.1,
                noise_draws=4,
                noise_seed=23,
            )
        assert model.forward is original_forward
        assert tuple(upstream._forward_pre_hooks) == user_hook_keys
        assert not final._forward_pre_hooks
        assert [
            (head.state_noise, head.noise_draws, head.noise_seed) for head in (upstream, final)
        ] == noise
        assert len(observed) == 1
        # Trainer fit hooks are gone; a normal forward no longer calls fit().
        assert model(values).shape == values.shape
    finally:
        handle.remove()


def test_inplace_custom_transform_does_not_modify_caller_warmup_or_training():
    class InplaceShift(nn.Module):
        def forward(self, values):
            return values.add_(1)

    source = rd.reservoir_input(1, dtype=torch.float64)
    head = CountingReadout("output").double()
    model = rd.ESNModel(source, head(InplaceShift()(source)))
    values = _series()
    warmup, train = values[:, :4], values[:, 4:]
    before = values.clone()
    rd.ESNTrainer(model).fit(warmup, train, 3 * (train + 1))
    torch.testing.assert_close(values, before, rtol=0, atol=0)
    torch.testing.assert_close(head.weight, torch.full_like(head.weight, 3))


def test_branch_features_live_until_last_consumer_and_are_then_released():
    class Features(nn.Module):
        def forward(self, values):
            return values + 1

    source = rd.reservoir_input(1, dtype=torch.float64)
    producer = Features()
    features = producer(source)
    first, second = CountingReadout("first").double(), CountingReadout("second").double()
    merge = rd.Concatenate()
    output = rd.RidgeReadoutLayer(2, 1, alpha=0.01, name="output").double()
    model = rd.ESNModel(source, output(merge(first(features), second(features))))
    observed = {}
    handles = [
        producer.register_forward_hook(
            lambda module, args, value: observed.update(feature=weakref.ref(value))
        ),
        second.register_forward_pre_hook(
            lambda module, args: observed.update(at_second=observed["feature"]() is not None)
        ),
        merge.register_forward_pre_hook(
            lambda module, args: observed.update(at_merge=observed["feature"]() is not None)
        ),
    ]
    values = _series()
    try:
        rd.ESNTrainer(model).fit(
            values[:, :4],
            values[:, 4:],
            {
                "first": 2 * (values[:, 4:] + 1),
                "second": 3 * (values[:, 4:] + 1),
                "output": 4 * (values[:, 4:] + 1),
            },
        )
    finally:
        for handle in handles:
            handle.remove()
    assert observed["at_second"]
    assert not observed["at_merge"]
    assert observed["feature"]() is None


@pytest.mark.parametrize("instance_forward", [False, True])
@pytest.mark.parametrize("compiled_attribute", [False, True])
def test_graph_prehook_failure_restores_exact_forward_and_compile_attributes(
    instance_forward, compiled_attribute
):
    source = rd.reservoir_input(1, dtype=torch.float64)
    head = CountingReadout("output").double()
    model = rd.ESNModel(source, head(source))
    if not instance_forward:
        del model.forward
    if compiled_attribute:

        def compiled(*args, **kwargs):
            raise AssertionError("cached compiled forward must not bypass fitting")

        model.__dict__["_compiled_call_impl"] = compiled
    else:
        model.__dict__.pop("_compiled_call_impl", None)
    names = ("forward", "_compiled_call_impl", "_resdag_fitting_execution_active")
    before = {name: model.__dict__[name] for name in names if name in model.__dict__}

    def fail(module, args):
        raise RuntimeError("graph prehook failure")

    handle = model.register_forward_pre_hook(fail)
    values = _series()
    try:
        with pytest.raises(RuntimeError, match="graph prehook failure"):
            rd.ESNTrainer(model).fit(values[:, :4], values[:, 4:], values[:, 4:])
    finally:
        handle.remove()
    assert {name for name in names if name in model.__dict__} == set(before)
    for name, value in before.items():
        assert model.__dict__[name] is value
    assert not head._forward_pre_hooks
    assert not head.fit_calls


@pytest.mark.parametrize("start", [True, False, 0.5, -1, 13])
def test_complete_prefix_rejects_invalid_supervised_start_before_mutation(start):
    source = rd.reservoir_input(1, dtype=torch.float64)
    head = CountingReadout("output").double()
    model = rd.ESNModel(source, head(source))
    values = _series()
    with pytest.raises(ValueError, match="fit_start"):
        rd.ESNTrainer(model)._fit_complete_prefix(
            (values,), {"output": values[:, 4:]}, fit_start=start
        )
    assert not head.fit_calls
    assert not head.is_fitted


def test_readout_posthooks_feed_their_actual_outputs_to_downstream_recurrence():
    source = rd.reservoir_input(1, dtype=torch.float64)
    upstream, final = CountingReadout("upstream").double(), CountingReadout("output").double()
    memory = _memory(0.3)
    model = rd.ESNModel(source, final(memory(upstream(source))))
    values = _series()
    expected = _scan(2 * values + 7, 0.3)
    outputs = []
    handles = [
        upstream.register_forward_hook(lambda module, args, output: output + 7),
        model.register_forward_hook(lambda module, args, output: outputs.append(output.clone())),
    ]
    try:
        rd.ESNTrainer(model).fit(
            values[:, :4],
            values[:, 4:],
            {"upstream": 2 * values[:, 4:], "output": 3 * expected[:, 4:]},
        )
    finally:
        for handle in handles:
            handle.remove()
    assert len(outputs) == 1
    torch.testing.assert_close(outputs[0], 3 * expected)
    torch.testing.assert_close(memory.state, expected[:, -1])


def test_tuple_unpack_releases_unused_outputs_and_preserves_used_branches():
    class Split(nn.Module):
        def forward(self, value):
            return value + 1, value + 2, value + 3

    source = rd.reservoir_input(1, dtype=torch.float64)
    split = Split()
    first, unused, third = split(source)
    a, b = CountingReadout("a").double(), CountingReadout("b").double()
    model = rd.ESNModel(source, (a(first), b(third)))
    observed = {}
    handles = [
        split.register_forward_hook(
            lambda module, args, outputs: observed.update(unused=weakref.ref(outputs[1]))
        ),
        a.register_forward_pre_hook(
            lambda module, args: observed.update(unused_alive=observed["unused"]() is not None)
        ),
    ]
    values = _series()
    try:
        rd.ESNTrainer(model).fit(
            values[:, :4],
            values[:, 4:],
            {"a": 2 * (values[:, 4:] + 1), "b": 3 * (values[:, 4:] + 3)},
        )
    finally:
        for handle in handles:
            handle.remove()
    assert not observed["unused_alive"]
    assert observed["unused"]() is None
    torch.testing.assert_close(a.weight, torch.full_like(a.weight, 2))
    torch.testing.assert_close(b.weight, torch.full_like(b.weight, 3))


def test_module_compile_callable_is_restored_after_successful_fitting():
    source = rd.reservoir_input(1, dtype=torch.float64)
    head = CountingReadout("output").double()
    model = rd.ESNModel(source, head(source))
    model.compile(backend="eager")
    compiled = model.__dict__["_compiled_call_impl"]
    original_forward = model.forward
    values = _series()
    rd.ESNTrainer(model).fit(values[:, :4], values[:, 4:], 2 * values[:, 4:])
    assert model.__dict__["_compiled_call_impl"] is compiled
    assert model.forward is original_forward
    torch.testing.assert_close(head.weight, torch.full_like(head.weight, 2))


def test_compiled_wrapper_is_rejected_before_any_fit_mutation():
    source = rd.reservoir_input(1, dtype=torch.float64)
    head = CountingReadout("output").double()
    model = rd.ESNModel(source, head(source))
    wrapped = torch.compile(model, backend="eager")
    values = _series()
    with pytest.raises(TypeError, match="original ESNModel.*torch.compile"):
        rd.ESNTrainer(wrapped).fit(values[:, :4], values[:, 4:], 2 * values[:, 4:])
    assert not head.fit_calls
    assert not head.is_fitted


@pytest.mark.parametrize("kind", ["generated", "interpreted", "deepcopy", "pickle"])
def test_standard_declared_graph_forwards_survive_forward_validation(kind):
    source = rd.reservoir_input(1, dtype=torch.float64)
    head = CountingReadout("output").double()
    model = rd.ESNModel(source, head(source), enable_forward_codegen=kind != "interpreted")
    if kind == "deepcopy":
        model = copy.deepcopy(model)
    elif kind == "pickle":
        data = io.BytesIO()
        torch.save(model, data)
        data.seek(0)
        model = torch.load(data, weights_only=False)
    values = _series()
    rd.ESNTrainer(model).fit(values[:, :4], values[:, 4:], 2 * values[:, 4:])
    actual = next(layer for layer in model.modules() if isinstance(layer, CountingReadout))
    assert len(actual.fit_calls) == 1
    torch.testing.assert_close(actual.weight, torch.full_like(actual.weight, 2))


@pytest.mark.parametrize("kind", ["function", "bound_method", "subclass", "other_model"])
def test_custom_outer_forward_is_rejected_before_model_or_input_mutation(kind):
    class CustomModel(rd.ESNModel):
        def forward(self, values):
            return values + 7

    source = rd.reservoir_input(1, dtype=torch.float64)
    memory, head = _memory(0.3), CountingReadout("output").double()
    cls = CustomModel if kind == "subclass" else rd.ESNModel
    model = cls(source, head(memory(source)), enable_forward_codegen=kind != "subclass")
    if kind == "function":
        model.forward = lambda values: values + 7
    elif kind == "bound_method":

        def custom(self, values):
            return values + 7

        model.forward = MethodType(custom, model)
    elif kind == "other_model":
        model.forward = copy.deepcopy(model).forward
    memory.set_state(torch.full((2, 1), 9.0, dtype=torch.float64))
    before = {name: value.clone() for name, value in model.state_dict().items()}
    old_state, old_forward = memory.state.clone(), model.forward
    had_forward_attribute = "forward" in model.__dict__
    values = _series()
    with pytest.raises(ValueError, match="custom top-level forward.*graph operations.*hooks"):
        rd.ESNTrainer(model).fit(values[:, :4], values[:, 4:], 2 * values[:, 4:])
    assert ("forward" in model.__dict__) == had_forward_attribute
    assert model.forward == old_forward
    assert not head.fit_calls
    assert not head._forward_pre_hooks
    torch.testing.assert_close(memory.state, old_state, rtol=0, atol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
