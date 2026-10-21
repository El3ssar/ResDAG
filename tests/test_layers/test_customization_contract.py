"""Cross-family contracts for public initialization hooks and independent gates."""

import copy
import io

import pytest
import torch
from torch.nn import functional

import resdag as rd
from resdag.layers.cells.gated_cell import GatedReservoirCell

FAMILIES = [
    rd.ESNLayer,
    rd.ES2NLayer,
    rd.ResESNLayer,
    rd.EuSNLayer,
    rd.GatedReservoirLayer,
    rd.LIFReservoirLayer,
]


def constant(rows, cols, value=0.125, *, dtype=None, device=None):
    return torch.full((rows, cols), value, dtype=dtype, device=device)


@pytest.mark.parametrize("family", FAMILIES)
def test_hooks_are_available_on_each_existing_family(family):
    layer = family(
        8,
        feedback_size=2,
        recurrent_initializer=("signed_permutation", {"gain": 0.5}),
        bias_initializer=(constant, {"value": 0.25}),
        initial_state_initializer=(constant, {"value": 0.125}),
        dtype=torch.float64,
        seed=3,
    )
    assert all(p.dtype == torch.float64 for p in layer.parameters())
    torch.testing.assert_close(layer.cell.bias_h, torch.full((8,), 0.25, dtype=torch.float64))
    state = layer.cell.init_state(3, torch.device("cpu"), torch.float64)
    assert state.shape == (3, layer.cell.state_size)
    torch.testing.assert_close(state, torch.full_like(state, 0.125))
    state[0].zero_()
    assert torch.all(state[1] == 0.125)  # reset states never share writable storage
    fresh = layer.cell.init_state(1, torch.device("cpu"), torch.float64)
    assert torch.all(fresh == 0.125)
    output = layer(torch.randn(3, 4, 2, dtype=torch.float64))
    assert output.shape == (3, 4, 8)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("family", FAMILIES)
def test_alias_preserves_existing_seeded_initialization(family):
    common = dict(reservoir_size=8, feedback_size=2, seed=9)
    old = family(**common, topology=("simple_cycle", {"weight": 0.4}))
    new = family(**common, recurrent_initializer=("simple_cycle", {"weight": 0.4}))
    for name, value in old.state_dict().items():
        torch.testing.assert_close(value, new.state_dict()[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="topology|recurrent_initializer"):
        family(**common, topology="simple_cycle", recurrent_initializer="orthogonal")
    with pytest.raises(ValueError, match="bias"):
        family(**common, bias=False, bias_initializer="zero")


@pytest.mark.parametrize("family", FAMILIES)
def test_reset_template_round_trips_and_legacy_checkpoint_clears_it(family):
    common = dict(reservoir_size=8, feedback_size=2, seed=1)
    configured = family(**common, initial_state_initializer=constant)
    plain = family(**common)
    plain.load_state_dict(configured.state_dict())
    state = plain.cell.init_state(2, torch.device("cpu"), torch.float32)
    assert torch.all(state == 0.125)
    legacy = family(**common).state_dict()
    plain.load_state_dict(legacy)
    assert not plain.cell.init_state(2, torch.device("cpu"), torch.float32).any()
    restored = copy.deepcopy(configured)
    state[0].zero_()
    assert torch.all(restored.cell.init_state(2, torch.device("cpu"), torch.float32) == 0.125)


def test_independent_gate_projection_matches_equation_and_sequence_projection():
    cell = GatedReservoirCell(
        5,
        feedback_size=2,
        input_size=3,
        recurrent_initializer=("signed_permutation", {"gain": 0.8}),
        gate_recurrent_initializer=("signed_permutation", {"gain": 0.2}),
        feedback_initializer=(constant, {"value": 0.1}),
        input_initializer=(constant, {"value": 0.2}),
        gate_feedback_initializer=(constant, {"value": 0.3}),
        gate_input_initializer=(constant, {"value": 0.4}),
        bias_initializer=(constant, {"value": 0.5}),
        gate_bias_initializer=(constant, {"value": 1.5}),
        activation="tanh",
        dtype=torch.float64,
        seed=7,
    )
    fb, driver = torch.randn(2, 4, 2).double(), torch.randn(2, 4, 3).double()
    state = torch.randn(2, 5).double()
    projected = cell.project_inputs([fb, driver])
    for t in range(4):
        candidate = torch.tanh(
            functional.linear(fb[:, t], cell.weight_feedback)
            + functional.linear(driver[:, t], cell.weight_input)
            + functional.linear(state, cell.weight_hh)
            + cell.bias_h
        )
        gate = torch.sigmoid(
            functional.linear(fb[:, t], cell.weight_gate_feedback)
            + functional.linear(driver[:, t], cell.weight_gate_input)
            + functional.linear(state, cell.weight_hz)
            + cell.bias_z
        )
        expected = gate * state + (1 - gate) * candidate
        actual, next_state = cell.step(projected[:, t], state)
        direct, _ = cell([fb[:, t], driver[:, t]], state)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(direct, expected)
        state = next_state


@pytest.mark.parametrize("gate_role", ["feedback", "input"])
def test_one_gate_projection_override_reuses_other_candidate_projection(gate_role):
    kwargs = {f"gate_{gate_role}_initializer": (constant, {"value": 0.2})}
    layer = rd.GatedReservoirLayer(6, feedback_size=2, input_size=2, seed=3, **kwargs)
    x, driver = torch.randn(2, 5, 2), torch.randn(2, 5, 2)
    out = layer(x, driver)
    assert out.shape == (2, 5, 6)
    assert torch.isfinite(out).all()
    buffer = io.BytesIO()
    torch.save(layer, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=False)
    layer.reset_state()
    restored.reset_state()
    torch.testing.assert_close(layer(x, driver), restored(x, driver))


def test_gate_parameter_gradients_and_upstream_gradients():
    layer = rd.GatedReservoirLayer(
        6,
        feedback_size=2,
        gate_feedback_initializer="normal",
        gate_recurrent_initializer="orthogonal",
        trainable=True,
        seed=3,
    )
    x = torch.randn(2, 7, 2, requires_grad=True)
    layer(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())


def test_factories_forward_new_initialization_controls():
    model = rd.classic_esn(
        reservoir_size=10,
        feedback_size=2,
        output_size=2,
        recurrent_initializer=("signed_permutation", {"gain": 0.8}),
        bias_initializer=torch.nn.init.zeros_,
        initial_state_initializer=constant,
        dtype=torch.float64,
        seed=2,
    ).double()
    data = torch.randn(1, 32, 2, dtype=torch.float64)
    rd.ESNTrainer(model).fit(data[:, :5], data[:, 5:-1], data[:, 6:])
    predicted = model.forecast(data, horizon=3)
    assert predicted.shape == (1, 3, 2)
    assert torch.isfinite(predicted).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal unavailable")
@pytest.mark.parametrize("family", FAMILIES)
def test_direct_metal_construction_and_hooks_match_cpu(family):
    options = dict(
        reservoir_size=8,
        feedback_size=2,
        recurrent_initializer=("signed_permutation", {"gain": 0.5}),
        bias_initializer=constant,
        initial_state_initializer=constant,
        seed=7,
    )
    cpu = family(**options)
    metal = family(**options, device="mps")
    # RNG algorithms need only agree per backend; compare identical weights.
    cpu.load_state_dict({name: value.cpu() for name, value in metal.state_dict().items()})
    data = torch.randn(2, 9, 2)
    expected = cpu(data)
    actual = metal(data.to("mps")).cpu()
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    assert all(parameter.device.type == "mps" for parameter in metal.parameters())


@pytest.mark.parametrize("factory", [rd.gated_esn, rd.lsm, rd.es2n, rd.resesn, rd.eusn])
def test_each_factory_constructs_consistent_double_graph(factory):
    model = factory(
        reservoir_size=8,
        feedback_size=2,
        output_size=2,
        recurrent_initializer="simple_cycle",
        dtype=torch.float64,
        seed=7,
    )
    assert all(parameter.dtype == torch.float64 for parameter in model.parameters())
    output, _ = model.forward_stateless(torch.randn(2, 5, 2, dtype=torch.float64))
    assert output.shape == (2, 5, 2) and output.dtype == torch.float64


def test_deep_and_hybrid_factories_construct_consistent_double_graph():
    deep = rd.deep_esn(
        reservoir_sizes=[8, 8],
        feedback_size=2,
        output_size=2,
        recurrent_initializer="simple_cycle",
        dtype=torch.float64,
        seed=7,
    )
    expert = torch.nn.Linear(2, 2)
    hybrid = rd.hybrid_esn(
        expert,
        reservoir_size=8,
        feedback_size=2,
        output_size=2,
        recurrent_initializer="simple_cycle",
        dtype=torch.float64,
        seed=7,
    )
    for model in (deep, hybrid):
        output, _ = model.forward_stateless(torch.randn(2, 5, 2, dtype=torch.float64))
        assert output.shape == (2, 5, 2) and output.dtype == torch.float64


@pytest.mark.parametrize("family", [rd.GatedReservoirLayer, rd.LIFReservoirLayer])
def test_historical_full_pickle_can_then_load_new_reset_template(family):
    old = family(6, feedback_size=2, seed=3)
    old.cell._buffers.pop("_initial_state")
    old.cell._load_state_dict_pre_hooks.clear()
    old.cell.__dict__.pop("initial_state_initializer", None)
    old.cell.__dict__.pop("_separate_gate_drive", None)
    buffer = io.BytesIO()
    torch.save(old, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=False)
    new = family(6, feedback_size=2, seed=3, initial_state_initializer=constant)
    restored.load_state_dict(new.state_dict())
    state = restored.cell.init_state(2, torch.device("cpu"), torch.float32)
    assert torch.all(state == 0.125)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal unavailable")
@pytest.mark.parametrize("family", [rd.GatedReservoirLayer, rd.LIFReservoirLayer])
def test_standalone_hook_families_support_default_metal_device(family):
    with torch.device("mps"):
        layer = family(6, feedback_size=2, seed=7, initial_state_initializer=constant)
        output = layer(torch.randn(2, 5, 2))
    assert output.device.type == "mps" and torch.isfinite(output).all()
