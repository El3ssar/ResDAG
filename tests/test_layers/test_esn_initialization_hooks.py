"""First-class ESN parameter-role hooks retain all initializer spec formats."""

import copy

import pytest
import torch

from resdag import ESNLayer


def _ramp(rows, cols, *, dtype=None, device=None):
    return (
        torch.arange(rows * cols, dtype=dtype or torch.float64, device=device).reshape(rows, cols)
        / 100
    )


def test_recurrent_alias_preserves_legacy_values_and_rejects_ambiguity():
    a = ESNLayer(5, 2, topology="simple_cycle", seed=4)
    b = ESNLayer(5, 2, recurrent_initializer="simple_cycle", seed=4)
    for key in a.state_dict():
        assert torch.equal(a.state_dict()[key], b.state_dict()[key])
    with pytest.raises(ValueError, match="topology.*recurrent_initializer"):
        ESNLayer(5, 2, topology="simple_cycle", recurrent_initializer="simple_cycle")


def test_bias_and_initial_state_are_configurable_and_state_is_batch_independent():
    layer = ESNLayer(
        5,
        2,
        bias_initializer=(torch.nn.init.constant_, {"val": 0.125}),
        initial_state_initializer=_ramp,
        seed=7,
        dtype=torch.float64,
    )
    torch.testing.assert_close(layer.bias_h, torch.full((5,), 0.125, dtype=torch.float64))
    state = layer.cell.init_state(3, torch.device("cpu"), torch.float64)
    torch.testing.assert_close(state, _ramp(1, 5).expand(3, -1))
    state[0, 0] = 100
    assert state[1, 0] == 0
    assert layer.cell.init_state(1, torch.device("cpu"), torch.float64)[0, 0] == 0
    destination = ESNLayer(5, 2, seed=8).double()
    destination.load_state_dict(layer.state_dict())
    torch.testing.assert_close(
        destination.cell.init_state(2, torch.device("cpu"), torch.float64),
        _ramp(1, 5).expand(2, -1),
    )
    destination.load_state_dict(ESNLayer(5, 2).state_dict())
    assert destination.cell._initial_state is None


@pytest.mark.parametrize("name", ["bias_initializer", "initial_state_initializer"])
def test_random_role_hooks_reproducible_with_independent_seeds(name):
    kwargs = {name: "normal"}
    a = ESNLayer(8, 2, seed=5, **kwargs)
    b = ESNLayer(8, 2, seed=5, **kwargs)
    assert a.state_dict().keys() == b.state_dict().keys()
    for key, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[key])


def test_role_hooks_validation():
    with pytest.raises(ValueError, match="bias=True"):
        ESNLayer(5, 2, bias=False, bias_initializer="zero")
    for name in ("bias_initializer", "initial_state_initializer"):
        with pytest.raises(ValueError, match="finite"):
            ESNLayer(5, 2, **{name: (torch.nn.init.constant_, {"val": float("nan")})})
    with pytest.raises(TypeError, match="floating"):
        ESNLayer(5, 2, dtype=torch.int32)


def test_true_float64_construction_and_buffer_moves():
    a = ESNLayer(
        5,
        2,
        seed=3,
        dtype=torch.float64,
        bias_initializer=_ramp,
        initial_state_initializer=_ramp,
        leak_rate=[0.1, 0.2, 0.3, 0.4, 0.5],
    )
    assert all(p.dtype == torch.float64 for p in a.parameters())
    assert a.cell._leak_rate_vec.dtype == torch.float64
    assert not torch.equal(a.weight_hh, a.weight_hh.float().double())
    b = copy.deepcopy(a).float()
    assert b.cell._initial_state.dtype == torch.float32
    assert b.cell._leak_rate_vec.dtype == torch.float32


def test_full_pickle_from_before_optional_hooks_remains_usable():
    import io

    source = ESNLayer(5, 2, seed=7)
    inputs = torch.randn(2, 4, 2, generator=torch.Generator().manual_seed(9))
    expected = source(inputs).clone()
    source.reset_state()
    for name in ("_intrinsic_gain", "_intrinsic_bias", "_recurrent_gain", "_initial_state"):
        source.cell._buffers.pop(name)
    for name in (
        "bias_initializer",
        "initial_state_initializer",
        "recurrent_initializer",
        "_factory_kwargs",
        "_has_intrinsic",
        "_has_recurrent_gain",
    ):
        delattr(source.cell, name)
    stream = io.BytesIO()
    torch.save(source, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=False)
    assert restored.cell._intrinsic_gain is None
    assert restored.cell._initial_state is None
    torch.testing.assert_close(restored(inputs), expected, atol=0, rtol=0)
    restored.leak_rate = [0.2] * 5
    assert restored.leak_rate.shape == (5,)


def test_mps_rejects_unsupported_precision_and_sparse_requests_early():
    with pytest.raises(ValueError, match="MPS.*float64"):
        ESNLayer(5, 2, dtype=torch.float64, device="mps")
    with pytest.raises(ValueError, match="sparse=True.*MPS"):
        ESNLayer(5, 2, sparse=True, device="mps")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_direct_mps_initialization_uses_reproducible_cpu_draws_and_moves_buffers():
    kwargs = dict(
        seed=7,
        bias_initializer="normal",
        initial_state_initializer="normal",
        recurrent_initializer="orthogonal",
        leak_rate=[0.5] * 5,
    )
    cpu = ESNLayer(5, 2, **kwargs)
    native = ESNLayer(5, 2, device="mps", **kwargs)
    for key, value in cpu.state_dict().items():
        assert torch.equal(value, native.state_dict()[key].cpu())
    output = native(torch.ones(2, 3, 2, device="mps"))
    assert output.device.type == "mps" and bool(torch.isfinite(output).all())
