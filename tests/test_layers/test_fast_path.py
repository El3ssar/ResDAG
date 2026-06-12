"""Equivalence tests for the projected fast path.

`BaseReservoirLayer.forward` uses `cell.project_inputs()` + `cell.step()`
when the cell supports it. These tests pin the contract: the fast path must
produce the same states as driving `cell.forward()` one step at a time.
"""

import pytest
import torch

from resdag.layers import ESNLayer, NGReservoir
from resdag.layers.cells import ESNCell


def _manual_per_step(cell: ESNCell, inputs: list[torch.Tensor]) -> torch.Tensor:
    """Reference: drive the cell with per-step forward(), no fast path."""
    batch, seq_len, _ = inputs[0].shape
    state = cell.init_state(batch, inputs[0].device, inputs[0].dtype)
    outputs = []
    for t in range(seq_len):
        inputs_t = [stream[:, t, :] for stream in inputs]
        output, state = cell(inputs_t, state)
        outputs.append(output)
    return torch.stack(outputs, dim=1)


class TestFastPathEquivalence:
    @pytest.mark.parametrize("leak_rate", [1.0, 0.3])
    @pytest.mark.parametrize("bias", [True, False])
    def test_feedback_only(self, device, leak_rate, bias):
        torch.manual_seed(0)
        layer = ESNLayer(
            reservoir_size=64,
            feedback_size=5,
            spectral_radius=0.9,
            leak_rate=leak_rate,
            bias=bias,
        ).to(device)
        feedback = torch.randn(4, 30, 5, device=device)

        layer.reset_state()
        fast = layer(feedback)
        reference = _manual_per_step(layer.cell, [feedback])

        assert torch.allclose(fast, reference, atol=1e-6)

    def test_with_driving_input(self, device):
        torch.manual_seed(0)
        layer = ESNLayer(
            reservoir_size=64,
            feedback_size=3,
            input_size=7,
            spectral_radius=0.9,
            leak_rate=0.5,
        ).to(device)
        feedback = torch.randn(2, 25, 3, device=device)
        driver = torch.randn(2, 25, 7, device=device)

        layer.reset_state()
        fast = layer(feedback, driver)
        reference = _manual_per_step(layer.cell, [feedback, driver])

        assert torch.allclose(fast, reference, atol=1e-6)

    def test_validation_still_raises_in_fast_path(self):
        layer = ESNLayer(reservoir_size=16, feedback_size=3)

        with pytest.raises(ValueError, match="Feedback size mismatch"):
            layer(torch.randn(1, 10, 4))

        with pytest.raises(ValueError, match="without input_size"):
            layer(torch.randn(1, 10, 3), torch.randn(1, 10, 2))

    def test_ngcell_has_no_projection(self):
        """NG-RC keeps the per-step fallback (delay-buffer state)."""
        layer = NGReservoir(input_dim=3, k=2, s=1, p=2)
        assert layer.cell.project_inputs([torch.randn(1, 10, 3)]) is None

        out = layer(torch.randn(1, 10, 3))
        assert out.shape == (1, 10, layer.cell.feature_dim)

    def test_single_step_forward_unchanged(self):
        """cell.forward on one step equals project+step composition."""
        torch.manual_seed(1)
        cell = ESNCell(reservoir_size=32, feedback_size=4, spectral_radius=0.8, leak_rate=0.7)
        fb_t = torch.randn(3, 4)
        state = cell.init_state(3, fb_t.device, fb_t.dtype)

        out_fwd, _ = cell([fb_t], state)
        proj = cell.project_inputs([fb_t])
        out_step, _ = cell.step(proj, state)

        assert torch.allclose(out_fwd, out_step, atol=1e-6)
