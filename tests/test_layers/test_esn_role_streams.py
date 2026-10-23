"""Independent initialization roles, explicit RNG ownership and stored-weight parity."""

import copy

import pytest
import torch

import resdag as rd
from resdag.init.input_feedback import RandomInputInitializer
from resdag.init.topology import MatrixTopology

_C = 0x9E3779B9
_MODULUS = 2**63 - 1


def _draw(shape, seed, offset=0, scale=1.0):
    local_seed = seed if offset == 0 else (seed + offset) % _MODULUS
    generator = torch.Generator(device="cpu").manual_seed(local_seed)
    return torch.empty(shape, dtype=torch.float64).uniform_(-scale, scale, generator=generator)


def _cell(**options):
    arguments = dict(
        reservoir_size=8,
        feedback_size=8,
        input_size=8,
        spectral_radius=None,
        bias_scaling=0.25,
        seed=42,
        dtype=torch.float64,
    )
    arguments.update(options)
    return rd.ESNCell(**arguments)


@pytest.mark.parametrize("seed", [0, 42, 2**63 - 5])
@pytest.mark.parametrize("feedback", [None, "random"])
def test_defaults_match_independent_role_oracles_without_reusing_prefixes(seed, feedback):
    cell = _cell(seed=seed, feedback_initializer=feedback)
    expected = {
        "weight_feedback": _draw((8, 8), seed),
        "weight_input": _draw((8, 8), seed, 2 * _C),
        "weight_hh": _draw((8, 8), seed, 4 * _C),
        "bias_h": _draw((8,), seed, 6 * _C, scale=0.25),
    }
    for name, value in expected.items():
        torch.testing.assert_close(getattr(cell, name), value, rtol=0, atol=0)
    matrices = [cell.weight_feedback, cell.weight_input, cell.weight_hh]
    for index, matrix in enumerate(matrices):
        assert not torch.equal(matrix.flatten()[:8] * 0.25, cell.bias_h)
        for other in matrices[index + 1 :]:
            assert not torch.equal(matrix, other)


def _uniform_recurrent(n, seed=None):
    return _draw((n, n), seed)


@pytest.mark.parametrize(
    "recurrent",
    [
        _uniform_recurrent,
        "erdos_renyi",
        ("erdos_renyi", {"p": 0.3}),
        (torch.nn.init.uniform_, {"a": -1.0, "b": 1.0}),
    ],
)
def test_named_feedback_and_custom_recurrence_cannot_restart_default_bias_stream(recurrent):
    cell = _cell(feedback_initializer="random", recurrent_initializer=recurrent)
    torch.testing.assert_close(cell.bias_h, _draw((8,), 42, 6 * _C, scale=0.25), rtol=0, atol=0)
    assert not torch.equal(cell.bias_h, cell.weight_feedback.flatten()[:8] * 0.25)
    if recurrent is _uniform_recurrent or isinstance(recurrent, tuple) and callable(recurrent[0]):
        torch.testing.assert_close(cell.weight_hh, _draw((8, 8), 42, 4 * _C), rtol=0, atol=0)


@pytest.mark.parametrize("seed", [42, None])
@pytest.mark.parametrize("input_size,input_initializer", [(None, None), (1, None), (3, "normal")])
def test_optional_input_and_bias_do_not_shift_other_roles(seed, input_size, input_initializer):
    torch.manual_seed(2718)
    reference = _cell(
        seed=seed, feedback_initializer="random", recurrent_initializer=_uniform_recurrent
    )
    torch.manual_seed(2718)
    candidate = _cell(
        seed=seed,
        input_size=input_size,
        input_initializer=input_initializer,
        feedback_initializer="random",
        recurrent_initializer=_uniform_recurrent,
    )
    for name in ("weight_feedback", "weight_hh", "bias_h"):
        torch.testing.assert_close(
            getattr(candidate, name), getattr(reference, name), rtol=0, atol=0
        )
    torch.manual_seed(2718)
    no_bias = _cell(seed=seed, bias=False, feedback_initializer="normal")
    torch.manual_seed(2718)
    with_bias = _cell(seed=seed, feedback_initializer="normal")
    for name in ("weight_feedback", "weight_input", "weight_hh"):
        torch.testing.assert_close(getattr(no_bias, name), getattr(with_bias, name), rtol=0, atol=0)


@pytest.mark.parametrize("spec", ["tuple", "object"])
def test_explicit_initializer_and_recurrent_seeds_override_role_fallbacks(spec):
    feedback = ("random", {"seed": 7}) if spec == "tuple" else RandomInputInitializer(seed=7)
    recurrent = (
        (_uniform_recurrent, {"seed": 11})
        if spec == "tuple"
        else MatrixTopology(_uniform_recurrent, {"seed": 11})
    )
    one = _cell(feedback_initializer=feedback, recurrent_initializer=recurrent, seed=42)
    two = _cell(feedback_initializer=feedback, recurrent_initializer=recurrent, seed=999)
    torch.testing.assert_close(one.weight_feedback, _draw((8, 8), 7), rtol=0, atol=0)
    torch.testing.assert_close(one.weight_hh, _draw((8, 8), 11), rtol=0, atol=0)
    torch.testing.assert_close(one.weight_feedback, two.weight_feedback, rtol=0, atol=0)
    torch.testing.assert_close(one.weight_hh, two.weight_hh, rtol=0, atol=0)


def test_explicit_none_and_callback_generator_remain_owned_by_the_spec():
    observed = []

    def explicit_none(n, seed="not-supplied"):
        observed.append(seed)
        return torch.eye(n, dtype=torch.float64)

    _cell(recurrent_initializer=(explicit_none, {"seed": None}))
    assert observed == [None]
    generator = torch.Generator().manual_seed(123)
    expected = torch.empty(8, 8, dtype=torch.float64).uniform_(
        generator=torch.Generator().manual_seed(123)
    )
    state = generator.get_state().clone()
    cell = _cell(recurrent_initializer=(torch.nn.init.uniform_, {"generator": generator}))
    torch.testing.assert_close(cell.weight_hh, expected, rtol=0, atol=0)
    assert not torch.equal(state, generator.get_state())  # explicitly owned mutable callback RNG


def test_fixed_int_and_advanced_layer_generator_do_not_consume_caller_rng():
    torch.manual_seed(1234)
    generator = torch.Generator().manual_seed(42)
    torch.rand(13, generator=generator)
    global_state, local_state = torch.get_rng_state().clone(), generator.get_state().clone()
    one = _cell(seed=generator, feedback_initializer="random", recurrent_initializer="erdos_renyi")
    two = _cell(seed=42, feedback_initializer="random", recurrent_initializer="erdos_renyi")
    assert torch.equal(torch.get_rng_state(), global_state)
    assert torch.equal(generator.get_state(), local_state)
    for name in ("weight_feedback", "weight_input", "weight_hh", "bias_h"):
        torch.testing.assert_close(getattr(one, name), getattr(two, name), rtol=0, atol=0)


def test_none_draws_exactly_one_cpu_master_then_all_roles_are_local():
    torch.manual_seed(1234)
    expected_master = int(torch.randint(0, _MODULUS, (1,), device="cpu").item())
    expected_state = torch.get_rng_state().clone()
    torch.manual_seed(1234)
    cell = _cell(seed=None, feedback_initializer="random", recurrent_initializer="erdos_renyi")
    assert cell._seed_int == expected_master
    assert torch.equal(torch.get_rng_state(), expected_state)
    torch.testing.assert_close(
        cell.bias_h, _draw((8,), expected_master, 6 * _C, scale=0.25), rtol=0, atol=0
    )


class _Continuous(rd.ContinuousReservoirCell):
    def vector_field(self, state, projected):
        return self._recurrent(state) + projected


@pytest.mark.parametrize("cls", [rd.ESNCell, _Continuous, rd.EuSNCell, rd.ES2NCell, rd.ResESNCell])
def test_inherited_families_share_roles_without_input_or_skip_coupling(cls):
    options = dict(
        reservoir_size=8, feedback_size=2, seed=42, dtype=torch.float64, spectral_radius=None
    )
    one = cls(**options)
    two = cls(**options, input_size=3)
    for name in ("weight_feedback", "weight_hh", "bias_h"):
        torch.testing.assert_close(getattr(one, name), getattr(two, name), rtol=0, atol=0)
    torch.testing.assert_close(one.bias_h, _draw((8,), 42, 6 * _C), rtol=0, atol=0)
    if hasattr(one, "skip_matrix"):
        torch.testing.assert_close(one.skip_matrix, two.skip_matrix, rtol=0, atol=0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_construction_retains_cpu_master_and_weight_placement_policy():
    options = dict(reservoir_size=8, feedback_size=2, input_size=2, seed=42, spectral_radius=None)
    cpu = rd.ESNCell(**options, device="cpu")
    mps = rd.ESNCell(**options, device="mps")
    for name in ("weight_feedback", "weight_input", "weight_hh", "bias_h"):
        assert getattr(mps, name).device.type == "mps"
        torch.testing.assert_close(getattr(mps, name).cpu(), getattr(cpu, name), rtol=0, atol=0)


def test_copies_and_batch_changes_preserve_weights_and_upstream_gradients():
    layer = rd.ESNLayer(8, feedback_size=2, spectral_radius=None, seed=42, dtype=torch.float64)
    clone = copy.deepcopy(layer).double()
    weights = {name: value.detach().clone() for name, value in layer.named_parameters()}
    for batch in (1, 3):
        values = torch.randn(batch, 4, 2, dtype=torch.float64, requires_grad=True)
        output = layer(values)
        clone.reset_state()
        torch.testing.assert_close(output, clone(values), rtol=0, atol=0)
        output.square().sum().backward()
        assert values.grad is not None and torch.isfinite(values.grad).all()
    for name, value in layer.named_parameters():
        torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
        assert value.grad is None


def test_portable_checkpoint_retains_materialized_weights_not_new_seed_draws(tmp_path):
    graph = rd.classic_esn(
        8, 1, 1, topology=None, spectral_radius=None, seed=9, dtype=torch.float64, readout="svd"
    )
    layer = next(module for module in graph.modules() if isinstance(module, rd.ESNLayer))
    # Materialize the old sequential-generator convention, independently of the
    # new constructor. A checkpoint must preserve these exact stored tensors.
    generator = torch.Generator().manual_seed(9)
    with torch.no_grad():
        for value in (layer.weight_feedback, layer.weight_hh, layer.bias_h):
            value.uniform_(-1, 1, generator=generator)
    stored = {name: value.detach().clone() for name, value in layer.named_parameters()}
    model = rd.Model(graph=graph, washout=2)
    data = torch.sin(torch.arange(40, dtype=torch.float64)[:, None] * 0.1)
    model.fit(data)
    expected = model.forecast(data[-5:], horizon=4)
    path = tmp_path / "materialized.pt"
    model.save(path)
    restored = rd.load(path)
    restored_layer = next(
        module for module in restored.graph.modules() if isinstance(module, rd.ESNLayer)
    )
    for name, value in restored_layer.named_parameters():
        torch.testing.assert_close(value, stored[name], rtol=0, atol=0)
    torch.testing.assert_close(restored.forecast(data[-5:], horizon=4), expected, rtol=0, atol=0)
