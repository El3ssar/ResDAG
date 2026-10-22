"""Full trained-DAG composition with explicit or independent recurrent state."""

import copy

import pytest
import torch
from torch import nn

from resdag import ESNModel, ESNTrainer, reservoir_input
from resdag.ensemble import CoupledEnsembleESNModel
from resdag.layers import Concatenate, ESNLayer, ExtendedSquare, NGReservoirLayer, RidgeReadoutLayer
from resdag.models import classic_esn, coupled_ensemble_esn, parallel_esn


def _branched_model() -> ESNModel:
    inp = reservoir_input(2)
    fast = ESNLayer(10, feedback_size=2, leak_rate=0.8, seed=3)(inp)
    slow = ESNLayer(8, feedback_size=2, leak_rate=0.2, seed=4)(inp)
    features = Concatenate()(ExtendedSquare()(fast), slow)
    primary = RidgeReadoutLayer(28, 2, name="output")(features)
    auxiliary = nn.Linear(28, 1)(features)
    return ESNModel(inp, (primary, auxiliary))


def test_full_graph_matches_stateful_outputs_and_preserves_stored_state() -> None:
    model = _branched_model()
    x = torch.randn(2, 13, 2)
    model(x)
    before = model.get_reservoir_states()
    independent, states = model.forward_stateless(x)
    after = model.get_reservoir_states()
    for name in before:
        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
    model.reset_reservoirs()
    expected = model(x)
    for actual, reference in zip(independent, expected):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for name, state in model.get_reservoir_states().items():
        torch.testing.assert_close(states[name], state, rtol=0, atol=0)


def test_explicit_states_continue_parallel_transformed_multihead_graph() -> None:
    model = _branched_model()
    x = torch.randn(2, 19, 2)
    full, full_states = model.forward_stateless(x)
    first, states = model.forward_stateless(x[:, :7])
    second, final = model.forward_stateless(x[:, 7:], states=states)
    for a, b, reference in zip(first, second, full):
        torch.testing.assert_close(torch.cat((a, b), dim=1), reference)
    for name, state in full_states.items():
        torch.testing.assert_close(final[name], state)
    assert model.get_reservoir_states() == {}


@pytest.mark.parametrize(
    "entry",
    [
        "stateless",
        "adapter",
        "stateful_adapter",
        "forecast_one",
        "forecast_many",
        "flat",
        "coupled_adapter",
        "coupled_forecast",
    ],
)
def test_repeated_reservoir_instance_is_rejected_before_state_changes(entry) -> None:
    inp = reservoir_input(2)
    shared = ESNLayer(2, feedback_size=2, seed=3)
    first = shared(inp)
    second = shared(first)
    model = ESNModel(inp, second)
    x = torch.randn(2, 6, 2)
    # Existing ordinary module execution remains available for legacy graphs.
    model(x)
    before = model.get_state()
    with pytest.raises(ValueError, match="same reservoir layer instance.*chunk boundaries"):
        if entry == "stateless":
            model.forward_stateless(x)
        elif entry == "adapter":
            model.as_module()
        elif entry == "stateful_adapter":
            model.as_module(stateful=True)
        elif entry == "forecast_one":
            model.forecast(x, horizon=1)
        elif entry == "forecast_many":
            model.forecast(x, horizon=3)
        elif entry == "flat":
            model._get_flat_step()
        elif entry == "coupled_adapter":
            CoupledEnsembleESNModel([model]).as_module()
        else:
            CoupledEnsembleESNModel([model]).forecast(x, horizon=1)
    for name, state in model.get_state().items():
        torch.testing.assert_close(state, before[name], rtol=0, atol=0)


def test_distinct_reservoir_instances_can_share_weights_and_keep_causal_state() -> None:
    inp = reservoir_input(2)
    left = ESNLayer(2, feedback_size=2, seed=3)
    right = ESNLayer(2, feedback_size=2, seed=4)
    right.cell.weight_hh = left.cell.weight_hh
    right.cell.weight_feedback = left.cell.weight_feedback
    right.cell.bias_h = left.cell.bias_h
    model = ESNModel(inp, right(left(inp)))
    x = torch.randn(2, 9, 2)
    full, full_states = model.forward_stateless(x)
    first, state = model.forward_stateless(x[:, :4])
    second, final = model.forward_stateless(x[:, 4:], states=state)
    assert len(full_states) == 2
    assert left.cell.weight_hh is right.cell.weight_hh
    torch.testing.assert_close(torch.cat((first, second), dim=1), full)
    for name in full_states:
        torch.testing.assert_close(final[name], full_states[name])
    flat = model._get_flat_step()
    states = [layer.cell.init_state(2, x.device, x.dtype) for layer in flat.reservoir_layers]
    frames = []
    for frame in x.split(1, dim=1):
        outputs, states = flat.step((frame,), states)
        frames.append(outputs[0])
    torch.testing.assert_close(torch.cat(frames, dim=1), full)


@pytest.mark.parametrize("duplicate_model", [False, True])
def test_ensemble_members_cannot_share_a_reservoir_instance(duplicate_model) -> None:
    first_input = reservoir_input(2)
    shared = ESNLayer(2, feedback_size=2, seed=3)
    first = ESNModel(first_input, shared(first_input))
    second_input = reservoir_input(2)
    second = first if duplicate_model else ESNModel(second_input, shared(second_input))
    with pytest.raises(ValueError, match="members 0 and 1 share a reservoir layer instance"):
        CoupledEnsembleESNModel([first, second])


def test_ensemble_members_can_share_weights_with_independent_reservoir_state() -> None:
    first_input, second_input = reservoir_input(2), reservoir_input(2)
    left = ESNLayer(2, feedback_size=2, seed=3)
    right = ESNLayer(2, feedback_size=2, seed=4)
    right.cell.weight_hh = left.cell.weight_hh
    right.cell.weight_feedback = left.cell.weight_feedback
    right.cell.bias_h = left.cell.bias_h
    model = CoupledEnsembleESNModel(
        [ESNModel(first_input, left(first_input)), ESNModel(second_input, right(second_input))]
    )
    x = torch.randn(2, 7, 2)
    full, _ = model.forward_stateless(x)
    first, state = model.forward_stateless(x[:, :3])
    second, state = model.forward_stateless(x[:, 3:], states=state)
    assert len(state) == 2
    torch.testing.assert_close(torch.cat((first, second), dim=1), full)


def test_explicit_state_keeps_gradient_across_chunks_and_upstream_encoder() -> None:
    inp = reservoir_input(2)
    reservoir = ESNLayer(8, feedback_size=2, seed=5)(inp)
    model = ESNModel(inp, nn.Linear(8, 1)(reservoir))
    encoder = nn.Linear(3, 2)
    source = torch.randn(2, 11, 3, requires_grad=True)
    x = encoder(source)
    _, state = model.forward_stateless(x[:, :6])
    result, _ = model.forward_stateless(x[:, 6:], states=state)
    result.square().sum().backward()
    assert source.grad is not None
    assert source.grad[:, :6].abs().sum() > 0
    assert encoder.weight.grad is not None and encoder.weight.grad.abs().sum() > 0
    assert model.get_reservoir_states() == {}


def test_ngrc_explicit_state_uses_delay_buffer_shape() -> None:
    inp = reservoir_input(2)
    reservoir = NGReservoirLayer(input_dim=2, k=2, s=1, p=2)
    features = reservoir(inp)
    model = ESNModel(inp, features)
    x = torch.randn(2, 10, 2)
    result, states = model.forward_stateless(x)
    assert next(iter(states.values())).dim() == 3
    model.reset_reservoirs()
    torch.testing.assert_close(result, model(x))


def test_trained_model_adapter_preserves_readout_and_weights(tmp_path) -> None:
    inp = reservoir_input(2)
    reservoir = ESNLayer(10, feedback_size=2, seed=6)(inp)
    model = ESNModel(inp, RidgeReadoutLayer(20, 2, name="output")(ExtendedSquare()(reservoir)))
    series = torch.randn(2, 30, 2)
    ESNTrainer(model).fit((series[:, :5],), (series[:, 5:-1],), {"output": series[:, 6:]})
    adapter = model.as_module()
    first = adapter(series[:, -6:])
    second = adapter(series[:, -6:])
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    expected, _ = model.forward_stateless(series[:, -6:])
    torch.testing.assert_close(first, expected)
    assert next(adapter.parameters()) is next(model.parameters())
    path = tmp_path / "adapter.pt"
    torch.save(adapter.state_dict(), path)
    copied = copy.deepcopy(model).as_module()
    copied.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(copied(series[:, -6:]), first)
    copied = copied.double()
    assert copied(series[:, -6:].double()).dtype == torch.float64


def test_adapter_stateful_mode_and_whole_model_pickle(tmp_path) -> None:
    model = _branched_model()
    adapter = model.as_module(stateful=True)
    x = torch.randn(2, 8, 2)
    first = adapter(x)
    second = adapter(x)
    assert not torch.allclose(first[0], second[0])
    adapter.reset_state()
    torch.testing.assert_close(adapter(x)[0], first[0])
    # Populate both execution caches before serializing.
    model.forward_stateless(x)
    model._get_flat_step()
    path = tmp_path / "complete.pt"
    torch.save(adapter, path)
    restored = torch.load(path, weights_only=False)
    adapter.reset_state()
    restored.reset_state()
    torch.testing.assert_close(restored(x)[0], adapter(x)[0])


@pytest.mark.parametrize("bad", [{}, {"wrong": torch.zeros(2, 8)}])
def test_explicit_state_keys_must_match_model(bad) -> None:
    model = _branched_model()
    with pytest.raises(ValueError, match="state keys"):
        model.forward_stateless(torch.randn(2, 8, 2), states=bad)


def test_cached_plan_rebuilds_after_adding_an_output() -> None:
    inp = reservoir_input(2)
    hidden = ESNLayer(8, feedback_size=2, seed=7)(inp)
    output = nn.Linear(8, 2)(hidden)
    model = ESNModel(inp, output)
    x = torch.randn(2, 8, 2)
    model.forward_stateless(x)
    model._get_flat_step()
    model.add_output(hidden)
    result, _ = model.forward_stateless(x)
    assert isinstance(result, tuple) and len(result) == 2
    assert model._get_flat_step().n_outputs == 2


def _family_model(kind: str):
    if kind == "dag":
        return classic_esn(12, 2, 2, seed=12)
    if kind == "parallel":
        return parallel_esn(input_dim=2, reservoir_size=12, core_width=1, halo=1, seed=12)
    return coupled_ensemble_esn(
        2, model_factory=classic_esn, reservoir_size=12, feedback_size=2, output_size=2, seed=12
    )


@pytest.mark.parametrize("kind", ["dag", "parallel", "coupled"])
def test_all_model_families_share_named_state_and_independent_adapter(kind) -> None:
    model = _family_model(kind)
    x = torch.randn(2, 13, 2)
    full, full_state = model.forward_stateless(x)
    first, state = model.forward_stateless(x[:, :5])
    second, final = model.forward_stateless(x[:, 5:], states=state)
    torch.testing.assert_close(torch.cat((first, second), dim=1), full)
    for name in full_state:
        torch.testing.assert_close(final[name], full_state[name])
    assert model.get_state() == {}
    model.set_state(state)
    for name, tensor in model.get_state().items():
        torch.testing.assert_close(tensor, state[name])
    torch.testing.assert_close(model(x[:, 5:]), second)
    before = model.get_state()
    adapter = model.as_module()
    torch.testing.assert_close(adapter(x), full)
    torch.testing.assert_close(adapter(x), full)
    for name, tensor in model.get_state().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
    adapter.eval()
    assert not model.training
    adapter.train()
    assert model.training
    clone = copy.deepcopy(model).as_module()
    clone.load_state_dict(adapter.state_dict())
    torch.testing.assert_close(clone(x), full)
    clone.double()
    assert clone(x.double()).dtype == torch.float64


@pytest.mark.parametrize("kind", ["dag", "parallel", "coupled"])
def test_all_model_families_compose_with_upstream_trainable_modules(kind) -> None:
    model = _family_model(kind)
    pipeline = nn.Sequential(nn.Linear(3, 2), model.as_module(), nn.Linear(2, 1))
    x = torch.randn(2, 8, 3, requires_grad=True)
    pipeline(x).square().sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert pipeline[0].weight.grad is not None and pipeline[0].weight.grad.abs().sum() > 0
    assert model.get_state() == {}


@pytest.mark.parametrize("kind", ["dag", "parallel", "coupled"])
def test_all_model_families_validate_explicit_input_contract(kind) -> None:
    model = _family_model(kind)
    with pytest.raises(ValueError, match="3-D"):
        model.forward_stateless(torch.randn(2, 2))
    with pytest.raises(ValueError, match="features"):
        model.forward_stateless(torch.randn(2, 3, 4))
    with pytest.raises(ValueError, match="dtype"):
        model.forward_stateless(torch.randn(2, 3, 2, dtype=torch.float64))
    with pytest.raises(TypeError, match="floating"):
        model.forward_stateless(torch.zeros(2, 3, 2, dtype=torch.long))
    with pytest.raises(ValueError, match="expects"):
        model.forward_stateless(torch.randn(2, 3, 2), torch.randn(2, 3, 2))


@pytest.mark.parametrize("kind", ["dag", "parallel", "coupled"])
def test_empty_explicit_chunk_preserves_supplied_states(kind) -> None:
    model = _family_model(kind)
    x = torch.randn(2, 4, 2)
    _, state = model.forward_stateless(x)
    output, final = model.forward_stateless(x[:, :0], states=state)
    assert output.shape == (2, 0, 2)
    for name, tensor in state.items():
        torch.testing.assert_close(final[name], tensor, rtol=0, atol=0)
    assert model.get_state() == {}


def test_coupled_canonical_state_does_not_change_legacy_state_layout() -> None:
    model = _family_model("coupled")
    model(torch.randn(2, 6, 2))
    legacy = model.get_reservoir_states()
    canonical = model.get_state()
    assert isinstance(legacy, list) and len(legacy) == 2
    assert all(name.startswith("models.") for name in canonical)
    model.reset_reservoirs()
    model.set_reservoir_states(legacy)
    for name, state in model.get_state().items():
        torch.testing.assert_close(state, canonical[name])
    with pytest.raises(KeyError, match="unexpected"):
        model.set_state({**canonical, "bad": torch.zeros(2, 12)})


def test_symbolic_tuple_unpack_and_multiple_input_streams_share_one_plan() -> None:
    class Split(nn.Module):
        supports_framewise = True

        def forward(self, value):
            return value.chunk(2, dim=-1)

    feedback = reservoir_input(2)
    driver = reservoir_input(2)
    hidden = ESNLayer(8, feedback_size=2, input_size=2, seed=31)(feedback, driver)
    first, second = Split()(hidden)
    model = ESNModel((feedback, driver), (nn.Linear(4, 2)(first), second))
    x, d = torch.randn(2, 6, 2), torch.randn(2, 6, 2)
    expected = model(x, d)
    model.reset_reservoirs()
    actual, states = model.forward_stateless(x, d)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference)
    flat = model._get_flat_step()
    initial = [layer.cell.init_state(2, x.device, x.dtype) for layer in flat.reservoir_layers]
    step, step_states = flat.step((x[:, :1], d[:, :1]), initial)
    one, one_state = model.forward_stateless(x[:, :1], d[:, :1])
    for result, reference in zip(step, one):
        torch.testing.assert_close(result, reference)
    for result, reference in zip(step_states, one_state.values()):
        torch.testing.assert_close(result, reference)
    assert len(states) == 1
