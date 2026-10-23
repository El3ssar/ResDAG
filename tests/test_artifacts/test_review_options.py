"""Independent constructor-option and placement regressions for model artifacts."""

import pytest
import torch
from torch import nn

import resdag as rd
from resdag.artifacts import save_model


@pytest.mark.parametrize("trusted", [False, True])
def test_layernorm_without_bias_retains_exact_parameter_schema(tmp_path, trusted):
    original = nn.LayerNorm(3, eps=0.003, bias=False).double()
    with torch.no_grad():
        original.weight.copy_(torch.tensor([0.4, 1.3, -0.8], dtype=torch.float64))
    save_model(original, tmp_path / "layernorm.pt", trusted=trusted)
    restored = rd.load(tmp_path / "layernorm.pt", trusted=trusted)
    assert restored.bias is None
    assert tuple(restored.state_dict()) == ("weight",)
    assert restored.eps == original.eps
    values = torch.tensor([[1.0, 3.0, -2.0], [4.0, -1.0, 0.5]], dtype=torch.float64)
    torch.testing.assert_close(restored(values), original(values), rtol=0, atol=0)


@pytest.mark.parametrize("trusted", [False, True])
def test_prelu_constructor_initialization_survives_parameter_reset(tmp_path, trusted):
    original = nn.PReLU(3, init=0.8).double()
    with torch.no_grad():
        original.weight.copy_(torch.tensor([0.3, 0.4, 0.5], dtype=torch.float64))
    save_model(original, tmp_path / "prelu.pt", trusted=trusted)
    restored = rd.load(tmp_path / "prelu.pt", trusted=trusted)
    assert restored.init == 0.8
    values = torch.tensor([[-1.0, -2.0, -3.0], [1.0, -1.0, 4.0]], dtype=torch.float64)
    torch.testing.assert_close(restored(values), original(values), rtol=0, atol=0)
    original.reset_parameters()
    restored.reset_parameters()
    torch.testing.assert_close(restored.weight, original.weight, rtol=0, atol=0)
    torch.testing.assert_close(restored(values), original(values), rtol=0, atol=0)


@pytest.mark.parametrize("trusted", [False, True])
def test_converted_graph_precision_does_not_use_stale_construction_placeholders(tmp_path, trusted):
    source = rd.reservoir_input(2, dtype=torch.float32)
    linear = nn.Linear(2, 2)
    original = rd.ESNModel(source, linear(source)).double()
    assert original.inputs[0].v.dtype == torch.float32
    values = torch.arange(20, dtype=torch.float64).reshape(2, 5, 2)
    expected = original(values)
    save_model(original, tmp_path / "converted.pt", trusted=trusted)
    restored = rd.load(tmp_path / "converted.pt", trusted=trusted)
    assert next(restored.parameters()).dtype == torch.float64
    torch.testing.assert_close(restored(values), expected, rtol=0, atol=0)


@pytest.mark.parametrize("trusted", [False, True])
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_buffer_only_graph_placement_comes_from_restored_buffers(tmp_path, trusted, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    source = rd.reservoir_input(2, dtype=torch.float32, device=device)
    values = torch.arange(16.0, device=device).reshape(1, 8, 2)
    units = rd.Standardize(2).to(device)
    units.fit(values)
    original = rd.ESNModel(source, units(source))
    assert list(original.parameters()) == []
    save_model(original, tmp_path / "buffers.pt", trusted=trusted)
    restored = rd.load(tmp_path / "buffers.pt", trusted=trusted, map_location=device)
    assert {buffer.device.type for buffer in restored.buffers()} == {device}
    torch.testing.assert_close(restored(values), original(values), rtol=0, atol=0)
    if device == "mps":
        on_cpu = rd.load(tmp_path / "buffers.pt", trusted=trusted, map_location="cpu")
        assert {buffer.device.type for buffer in on_cpu.buffers()} == {"cpu"}
        torch.testing.assert_close(
            on_cpu(values.cpu()), original(values).cpu(), rtol=3e-6, atol=3e-6
        )
