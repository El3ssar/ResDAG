"""Live layer writes must change the executed cell, including derived values."""

import io
import math

import pytest
import torch

from resdag.layers import ESNLayer, EuSNLayer, LIFReservoirLayer
from resdag.layers.cells import ESNCell


def test_eusn_epsilon_and_dt_change_the_integrated_interval():
    layer = EuSNLayer(4, feedback_size=1, epsilon=0.1, seed=7)
    state = torch.ones(1, 4)
    drive = torch.full((1, 4), 0.2)
    field = layer.cell.vector_field(state, drive)
    layer.epsilon = 0.25
    assert layer.cell.dt == layer.dt == layer.epsilon == 0.25
    output, next_state = layer.cell.step(drive, state)
    torch.testing.assert_close(output, state + 0.25 * field)
    torch.testing.assert_close(next_state, output)
    layer.dt = 0.5
    assert layer.cell.epsilon == 0.5
    assert "epsilon" not in layer.__dict__ and "dt" not in layer.__dict__


@pytest.mark.parametrize("name", ["tau_mem", "tau_syn", "dt"])
def test_lif_time_changes_update_both_decays(name):
    layer = LIFReservoirLayer(4, feedback_size=1, synaptic_current=True, seed=7)
    setattr(layer, name, 2.0)
    assert getattr(layer.cell, name) == 2.0
    assert layer.cell.beta == math.exp(-layer.cell.dt / layer.cell.tau_mem)
    assert layer.cell.alpha == math.exp(-layer.cell.dt / layer.cell.tau_syn)
    layer.v_threshold = 0.4
    assert layer.cell.v_threshold == 0.4
    assert name not in layer.__dict__ and "v_threshold" not in layer.__dict__


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
@pytest.mark.parametrize("name", ["tau_mem", "tau_syn", "dt"])
def test_invalid_lif_time_write_is_atomic(name, value):
    layer = LIFReservoirLayer(4, feedback_size=1, seed=7)
    before = (
        layer.cell.dt,
        layer.cell.tau_mem,
        layer.cell.tau_syn,
        layer.cell.beta,
        layer.cell.alpha,
    )
    with pytest.raises(ValueError, match="positive and finite"):
        setattr(layer, name, value)
    assert before == (
        layer.cell.dt,
        layer.cell.tau_mem,
        layer.cell.tau_syn,
        layer.cell.beta,
        layer.cell.alpha,
    )


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_eusn_time_write_is_atomic(value):
    layer = EuSNLayer(4, feedback_size=1, seed=7)
    before = layer.epsilon
    with pytest.raises(ValueError, match="positive and finite"):
        layer.epsilon = value
    assert layer.cell.dt == before


def test_legacy_plain_time_attributes_remain_readable_after_full_pickle():
    for layer, names in [
        (EuSNLayer(4, feedback_size=1, seed=7), ("dt",)),
        (LIFReservoirLayer(4, feedback_size=1, seed=7), ("dt", "tau_mem", "tau_syn")),
    ]:
        for name in names:
            layer.cell.__dict__[name] = layer.cell.__dict__.pop(f"_{name}")
        stream = io.BytesIO()
        torch.save(layer, stream)
        stream.seek(0)
        restored = torch.load(stream, weights_only=False)
        restored.dt = 0.2
        assert restored.cell.dt == 0.2
        if hasattr(restored.cell, "tau_mem"):
            assert restored.cell.beta == math.exp(-0.2 / restored.cell.tau_mem)


@pytest.mark.parametrize("replace", [False, True])
def test_sparse_recurrent_operator_tracks_live_weight_edits(replace):
    cell = ESNCell(8, feedback_size=1, topology="simple_cycle", sparse=True, seed=7)
    state = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    before = cell._recurrent(state)
    with torch.no_grad():
        if replace:
            cell.weight_hh = torch.nn.Parameter(torch.eye(8) * 2, requires_grad=False)
        else:
            cell.weight_hh.copy_(torch.eye(8) * 2)
    after = cell._recurrent(state)
    torch.testing.assert_close(after, state * 2)
    assert not torch.equal(before, after)


def test_sparse_cache_refresh_preserves_calibration_gain_and_dtype():
    layer = ESNLayer(8, feedback_size=1, topology="simple_cycle", sparse=True, seed=7)
    cell = layer.cell
    cell._recurrent_gain = torch.arange(1, 9, dtype=torch.float32)
    cell._has_recurrent_gain = True
    layer.double()
    with torch.no_grad():
        cell.weight_hh.copy_(torch.eye(8, dtype=torch.float64))
    state = torch.ones(1, 8, dtype=torch.float64)
    torch.testing.assert_close(cell._recurrent(state), cell._recurrent_gain.unsqueeze(0))
    assert cell._weight_hh_csr.dtype == torch.float64


def test_auto_sparse_density_change_keeps_calibration_gain(monkeypatch):
    cell = ESNCell(8, feedback_size=1, topology="simple_cycle", sparse=False, seed=7)
    cell._sparse_requested = "auto"
    monkeypatch.setattr(cell, "_auto_min_size", lambda: 1)
    cell._maybe_build_sparse_weight()
    assert cell.sparse
    cell._recurrent_gain = torch.full((8,), 2.0)
    cell._has_recurrent_gain = True
    with torch.no_grad():
        cell.weight_hh.fill_(1)
    torch.testing.assert_close(cell._recurrent(torch.ones(1, 8)), torch.full((1, 8), 16.0))
    assert not cell.sparse


def test_sparse_inference_parameter_without_version_refreshes_after_mutation():
    with torch.inference_mode():
        cell = ESNCell(4, feedback_size=1, topology="simple_cycle", sparse=True, seed=7)
        state = torch.ones(1, 4)
        cell._recurrent(state)
        cell.weight_hh.copy_(torch.eye(4) * 3)
        torch.testing.assert_close(cell._recurrent(state), torch.full((1, 4), 3.0))


@pytest.mark.parametrize("replace", [False, True])
def test_sparse_operator_falls_back_to_dense_for_actual_trainable_weights(replace):
    cell = ESNCell(4, feedback_size=1, topology="simple_cycle", sparse=True, seed=7)
    if replace:
        cell.weight_hh = torch.nn.Parameter(torch.eye(4))
    else:
        cell.weight_hh.requires_grad_(True)
    state = torch.ones(1, 4)
    cell._recurrent(state).sum().backward()
    assert not cell.sparse
    torch.testing.assert_close(cell.weight_hh.grad, torch.ones(4, 4))
