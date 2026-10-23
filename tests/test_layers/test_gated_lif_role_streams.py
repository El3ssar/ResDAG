"""Independent gated/spiking roles and explicit caller RNG ownership."""

import copy

import pytest
import torch

import resdag as rd
from resdag.init.input_feedback import RandomInputInitializer
from resdag.init.topology import MatrixTopology

_C = 0x9E3779B9
_MODULUS = 2**63 - 1


def _draw(shape, seed, offset=0, scale=1.0, dtype=torch.float64):
    local_seed = seed if offset == 0 else (seed + offset) % _MODULUS
    generator = torch.Generator(device="cpu").manual_seed(local_seed)
    return torch.empty(shape, dtype=dtype).uniform_(-scale, scale, generator=generator)


def _cell(kind, **options):
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
    cls = rd.GatedReservoirCell if kind == "gated" else rd.LIFCell
    return cls(**arguments)


def _recurrent_name(kind):
    return "weight_hh" if kind == "gated" else "weight_rec"


def _recurrent_offset(kind):
    return 10 * _C if kind == "gated" else 4 * _C


def _uniform_recurrent(n, seed=None):
    return _draw((n, n), seed)


@pytest.mark.parametrize("kind", ["gated", "lif"])
@pytest.mark.parametrize("seed", [0, 42, 2**63 - 5])
@pytest.mark.parametrize("feedback", [None, "random"])
def test_default_roles_match_exact_independent_streams(kind, seed, feedback):
    cell = _cell(kind, seed=seed, feedback_initializer=feedback)
    expected = {
        "weight_feedback": _draw((8, 8), seed),
        "weight_input": _draw((8, 8), seed, 2 * _C),
        _recurrent_name(kind): _draw((8, 8), seed, _recurrent_offset(kind)),
        "bias_h": _draw((8,), seed, 5 * _C, scale=0.25),
    }
    if kind == "gated":
        expected["weight_hz"] = _draw((8, 8), seed, 4 * _C, scale=8**-0.5)
        torch.testing.assert_close(cell.bias_z, torch.full((8,), 2.0, dtype=torch.float64))
    for name, value in expected.items():
        torch.testing.assert_close(getattr(cell, name), value, rtol=0, atol=0)
    matrices = [cell.weight_feedback, cell.weight_input, getattr(cell, _recurrent_name(kind))]
    for index, matrix in enumerate(matrices):
        assert not torch.equal(matrix.flatten()[:8] * 0.25, cell.bias_h)
        for other in matrices[index + 1 :]:
            assert not torch.equal(matrix, other)


@pytest.mark.parametrize("kind", ["gated", "lif"])
@pytest.mark.parametrize("recurrent", [_uniform_recurrent, ("erdos_renyi", {"p": 0.3})])
def test_named_feedback_custom_recurrence_and_default_bias_do_not_share_draws(kind, recurrent):
    cell = _cell(kind, feedback_initializer="random", recurrent_initializer=recurrent)
    torch.testing.assert_close(cell.bias_h, _draw((8,), 42, 5 * _C, scale=0.25), rtol=0, atol=0)
    assert not torch.equal(cell.bias_h, cell.weight_feedback.flatten()[:8] * 0.25)
    if recurrent is _uniform_recurrent:
        torch.testing.assert_close(
            getattr(cell, _recurrent_name(kind)),
            _draw((8, 8), 42, _recurrent_offset(kind)),
            rtol=0,
            atol=0,
        )
    if kind == "gated":
        torch.testing.assert_close(
            cell.weight_hz, _draw((8, 8), 42, 4 * _C, scale=8**-0.5), rtol=0, atol=0
        )


@pytest.mark.parametrize("kind", ["gated", "lif"])
@pytest.mark.parametrize("seed", [42, None])
def test_optional_roles_and_initializer_choices_leave_other_streams_unchanged(kind, seed):
    torch.manual_seed(2718)
    reference = _cell(kind, seed=seed)
    for changes in (
        {"input_size": None},
        {"input_size": 3, "input_initializer": "normal"},
        {"bias": False},
        {"feedback_initializer": "normal"},
    ):
        torch.manual_seed(2718)
        candidate = _cell(kind, seed=seed, **changes)
        names = [_recurrent_name(kind)]
        if changes.get("bias") is not False:
            names.append("bias_h")
        if "feedback_initializer" not in changes:
            names.append("weight_feedback")
        if kind == "gated":
            names.append("weight_hz")
        for name in names:
            torch.testing.assert_close(
                getattr(candidate, name), getattr(reference, name), rtol=0, atol=0
            )


def test_gated_separate_projection_and_gate_overrides_do_not_shift_candidate_roles():
    one = _cell("gated")
    two = _cell(
        "gated",
        gate_feedback_initializer="random",
        gate_input_initializer="random",
        gate_recurrent_initializer=_uniform_recurrent,
        gate_bias_initializer="random",
        initial_state_initializer="random",
    )
    for name in ("weight_feedback", "weight_input", "weight_hh", "bias_h"):
        torch.testing.assert_close(getattr(two, name), getattr(one, name), rtol=0, atol=0)
    for name, shape, offset in (
        ("weight_gate_feedback", (8, 8), 8),
        ("weight_gate_input", (8, 8), 9),
        ("weight_hz", (8, 8), 4),
        ("bias_z", (8,), 6),
    ):
        torch.testing.assert_close(
            getattr(two, name), _draw(shape, 42, offset * _C), rtol=0, atol=0
        )
    torch.testing.assert_close(
        two.init_state(1, torch.device("cpu"), torch.float64),
        _draw((1, 8), 42, 7 * _C),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("kind", ["gated", "lif"])
@pytest.mark.parametrize("spec", ["tuple", "object"])
def test_explicit_initializer_seeds_have_priority_over_layer_roles(kind, spec):
    feedback = ("random", {"seed": 7}) if spec == "tuple" else RandomInputInitializer(seed=7)
    recurrent = (
        (_uniform_recurrent, {"seed": 11})
        if spec == "tuple"
        else MatrixTopology(_uniform_recurrent, {"seed": 11})
    )
    options = dict(
        feedback_initializer=feedback,
        recurrent_initializer=recurrent,
        bias_initializer=("random", {"seed": 13}),
        initial_state_initializer=("random", {"seed": 17}),
    )
    one, two = _cell(kind, seed=42, **options), _cell(kind, seed=999, **options)
    for name, seed in (("weight_feedback", 7), (_recurrent_name(kind), 11), ("bias_h", 13)):
        shape = (8,) if name == "bias_h" else (8, 8)
        torch.testing.assert_close(getattr(one, name), _draw(shape, seed), rtol=0, atol=0)
        torch.testing.assert_close(getattr(one, name), getattr(two, name), rtol=0, atol=0)
    torch.testing.assert_close(
        one.init_state(1, torch.device("cpu"), torch.float64),
        _draw((1, one.state_size), 17),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("kind", ["gated", "lif"])
def test_explicit_none_and_callback_generator_retain_initializer_ownership(kind):
    observed = []

    def explicit_none(n, seed="not-supplied"):
        observed.append(seed)
        return torch.eye(n, dtype=torch.float64)

    _cell(kind, recurrent_initializer=(explicit_none, {"seed": None}))
    assert observed == [None]
    generator = torch.Generator().manual_seed(123)
    before = generator.get_state().clone()
    expected = torch.empty(8, 8, dtype=torch.float64).uniform_(
        generator=torch.Generator().manual_seed(123)
    )
    cell = _cell(kind, recurrent_initializer=(torch.nn.init.uniform_, {"generator": generator}))
    torch.testing.assert_close(getattr(cell, _recurrent_name(kind)), expected, rtol=0, atol=0)
    assert not torch.equal(before, generator.get_state())


@pytest.mark.parametrize("kind", ["gated", "lif"])
def test_layer_generator_is_nonconsuming_and_uses_initial_seed(kind):
    torch.manual_seed(1234)
    generator = torch.Generator().manual_seed(42)
    torch.rand(13, generator=generator)
    global_state, local_state = torch.get_rng_state().clone(), generator.get_state().clone()
    options = dict(feedback_initializer="random", recurrent_initializer="erdos_renyi")
    one, two = _cell(kind, seed=generator, **options), _cell(kind, seed=42, **options)
    assert torch.equal(torch.get_rng_state(), global_state)
    assert torch.equal(generator.get_state(), local_state)
    for name, value in one.named_parameters():
        torch.testing.assert_close(value, dict(two.named_parameters())[name], rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["gated", "lif"])
def test_none_consumes_one_cpu_master_for_all_default_and_named_roles(kind):
    torch.manual_seed(1234)
    expected_master = int(torch.randint(0, _MODULUS, (1,), device="cpu").item())
    expected_state = torch.get_rng_state().clone()
    torch.manual_seed(1234)
    cell = _cell(
        kind,
        seed=None,
        feedback_initializer="random",
        recurrent_initializer="erdos_renyi",
        initial_state_initializer="random",
    )
    assert cell._seed_int == expected_master
    assert torch.equal(torch.get_rng_state(), expected_state)
    torch.testing.assert_close(
        cell.bias_h, _draw((8,), expected_master, 5 * _C, scale=0.25), rtol=0, atol=0
    )
    torch.testing.assert_close(
        cell.init_state(1, torch.device("cpu"), torch.float64),
        _draw((1, cell.state_size), expected_master, 7 * _C),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("kind", ["gated", "lif"])
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_construction_matches_cpu_without_consuming_supplied_generator(kind):
    generator = torch.Generator().manual_seed(42)
    torch.rand(13, generator=generator)
    before = generator.get_state().clone()
    options = dict(seed=generator, dtype=torch.float32, feedback_initializer="random")
    cpu, mps = _cell(kind, device="cpu", **options), _cell(kind, device="mps", **options)
    assert torch.equal(before, generator.get_state())
    for name, value in mps.named_parameters():
        assert value.device.type == "mps"
        torch.testing.assert_close(value.cpu(), dict(cpu.named_parameters())[name], rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["gated", "lif"])
def test_copies_batch_switches_and_gradients_preserve_materialized_weights(kind):
    cls = rd.GatedReservoirLayer if kind == "gated" else rd.LIFReservoirLayer
    options = {"activation": "tanh"} if kind == "gated" else {"output": "voltage"}
    layer = cls(8, feedback_size=2, spectral_radius=None, seed=42, dtype=torch.float64, **options)
    clone = copy.deepcopy(layer).double()
    weights = {name: value.detach().clone() for name, value in layer.named_parameters()}
    for batch in (1, 3):
        values = torch.randn(batch, 4, 2, dtype=torch.float64, requires_grad=True)
        layer.reset_state()
        clone.reset_state()
        output = layer(values)
        torch.testing.assert_close(output, clone(values), rtol=0, atol=0)
        output.square().sum().backward()
        assert values.grad is not None and torch.isfinite(values.grad).all()
        assert torch.count_nonzero(values.grad) > 0
    for name, value in layer.named_parameters():
        torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
        assert value.grad is None


@pytest.mark.parametrize("kind", ["gated", "lif"])
def test_portable_roundtrip_preserves_old_materialized_weights_and_forecasts(kind, tmp_path):
    cls = rd.GatedReservoirLayer if kind == "gated" else rd.LIFReservoirLayer
    options = {"activation": "tanh"} if kind == "gated" else {"output": "voltage"}
    layer = cls(8, feedback_size=1, spectral_radius=None, seed=9, dtype=torch.float64, **options)
    # Deliberately materialize sequential draws from the old convention. A load
    # must restore stored tensors, rather than regenerating new independent ones.
    generator = torch.Generator().manual_seed(9)
    with torch.no_grad():
        for value in layer.parameters():
            value.uniform_(-0.5, 0.5, generator=generator)
    stored = {name: value.detach().clone() for name, value in layer.named_parameters()}
    source = rd.Input((4, 1), dtype=torch.float64)
    prediction = rd.SVDReadoutLayer(8, 1, alpha=1e-5).double()(layer(source))
    model = rd.Model(inputs={"signal": source}, outputs={"prediction": prediction}, washout=2)
    data = torch.sin(torch.arange(40, dtype=torch.float64)[:, None] * 0.1)
    model.fit(data)
    expected = model.forecast(data[-5:], horizon=4)
    path = tmp_path / f"{kind}-materialized.pt"
    model.save(path)
    restored = rd.load(path)
    restored_layer = next(module for module in restored.graph.modules() if isinstance(module, cls))
    for name, value in restored_layer.named_parameters():
        torch.testing.assert_close(value, stored[name], rtol=0, atol=0)
    torch.testing.assert_close(restored.forecast(data[-5:], horizon=4), expected, rtol=0, atol=0)
