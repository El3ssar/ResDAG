"""Unit tests for GRUCell and GRULayer."""

import pytest
import torch

from resdag.layers.cells.gru_cell import GRUCell
from resdag.layers.reservoirs.gru import GRULayer


# ---------------------------------------------------------------------------
# GRUCell tests
# ---------------------------------------------------------------------------


class TestGRUCellInstantiation:
    """Test GRUCell construction and buffer registration."""

    def test_basic_instantiation(self) -> None:
        cell = GRUCell(input_dim=10, hidden_dim=50)
        assert cell.input_dim == 10
        assert cell.hidden_dim == 50
        assert cell.state_size == 50

    def test_input_weight_shapes(self) -> None:
        cell = GRUCell(input_dim=8, hidden_dim=32)
        assert cell.W_z.shape == (32, 8)
        assert cell.W_r.shape == (32, 8)
        assert cell.W_h.shape == (32, 8)

    def test_recurrent_weight_shapes(self) -> None:
        cell = GRUCell(input_dim=8, hidden_dim=32)
        assert cell.U_z.shape == (32, 32)
        assert cell.U_r.shape == (32, 32)
        assert cell.U_h.shape == (32, 32)

    def test_bias_shapes_when_enabled(self) -> None:
        cell = GRUCell(input_dim=8, hidden_dim=32, bias=True)
        assert cell.b_z is not None and cell.b_z.shape == (32,)
        assert cell.b_r is not None and cell.b_r.shape == (32,)
        assert cell.b_h is not None and cell.b_h.shape == (32,)

    def test_bias_none_when_disabled(self) -> None:
        cell = GRUCell(input_dim=8, hidden_dim=32, bias=False)
        assert cell.b_z is None
        assert cell.b_r is None
        assert cell.b_h is None

    def test_no_trainable_parameters(self) -> None:
        """All weights must be buffers, not nn.Parameters."""
        cell = GRUCell(input_dim=10, hidden_dim=50)
        assert sum(1 for _ in cell.parameters()) == 0

    def test_all_buffers_registered(self) -> None:
        cell = GRUCell(input_dim=10, hidden_dim=50)
        buffer_names = {name for name, _ in cell.named_buffers()}
        assert {"W_z", "W_r", "W_h", "U_z", "U_r", "U_h"}.issubset(buffer_names)

    def test_recurrent_matrices_independent(self) -> None:
        """U_z, U_r, U_h must not share storage."""
        torch.manual_seed(0)
        cell = GRUCell(input_dim=4, hidden_dim=16)
        assert not torch.allclose(cell.U_z, cell.U_r)
        assert not torch.allclose(cell.U_z, cell.U_h)
        assert not torch.allclose(cell.U_r, cell.U_h)

    def test_init_state(self) -> None:
        cell = GRUCell(input_dim=10, hidden_dim=50)
        h0 = cell.init_state(batch_size=4, device=torch.device("cpu"), dtype=torch.float32)
        assert h0.shape == (4, 50)
        assert torch.all(h0 == 0.0)


class TestGRUCellSpectralRadius:
    """Verify spectral radius is applied to each recurrent matrix."""

    @pytest.mark.parametrize("target_sr", [0.5, 0.9, 1.2])
    def test_spectral_radius_U_z(self, target_sr: float) -> None:
        cell = GRUCell(input_dim=4, hidden_dim=30, spectral_radius=target_sr)
        eigvals = torch.linalg.eigvals(cell.U_z)
        actual_sr = torch.max(torch.abs(eigvals)).item()
        assert abs(actual_sr - target_sr) < 0.01

    @pytest.mark.parametrize("target_sr", [0.5, 0.9, 1.2])
    def test_spectral_radius_U_r(self, target_sr: float) -> None:
        cell = GRUCell(input_dim=4, hidden_dim=30, spectral_radius=target_sr)
        eigvals = torch.linalg.eigvals(cell.U_r)
        actual_sr = torch.max(torch.abs(eigvals)).item()
        assert abs(actual_sr - target_sr) < 0.01

    @pytest.mark.parametrize("target_sr", [0.5, 0.9, 1.2])
    def test_spectral_radius_U_h(self, target_sr: float) -> None:
        cell = GRUCell(input_dim=4, hidden_dim=30, spectral_radius=target_sr)
        eigvals = torch.linalg.eigvals(cell.U_h)
        actual_sr = torch.max(torch.abs(eigvals)).item()
        assert abs(actual_sr - target_sr) < 0.01


class TestGRUCellSparsity:
    """Test that sparsity masks are applied."""

    def test_sparsity_reduces_nonzeros(self) -> None:
        torch.manual_seed(42)
        dense = GRUCell(input_dim=4, hidden_dim=64, sparsity=0.0)
        sparse = GRUCell(input_dim=4, hidden_dim=64, sparsity=0.8)
        dense_nnz = (dense.U_z != 0).sum().item()
        sparse_nnz = (sparse.U_z != 0).sum().item()
        assert sparse_nnz < dense_nnz

    def test_zero_sparsity_no_forced_zeros(self) -> None:
        cell = GRUCell(input_dim=4, hidden_dim=32, sparsity=0.0)
        # With a full random matrix, all entries should typically be non-zero
        assert (cell.U_z == 0).sum().item() == 0


class TestGRUCellForward:
    """Test single-step forward pass."""

    def test_output_shape_single_input(self) -> None:
        cell = GRUCell(input_dim=10, hidden_dim=50)
        x = torch.randn(4, 10)
        h = torch.zeros(4, 50)
        new_h = cell([x], h)
        assert new_h.shape == (4, 50)

    def test_output_shape_concatenated_inputs(self) -> None:
        """Cell should concatenate inputs[0] and inputs[1] to form x."""
        cell = GRUCell(input_dim=15, hidden_dim=50)
        fb = torch.randn(4, 10)
        drv = torch.randn(4, 5)
        h = torch.zeros(4, 50)
        new_h = cell([fb, drv], h)
        assert new_h.shape == (4, 50)

    def test_hidden_state_bounded_tanh_dominated(self) -> None:
        """GRU output is a convex mix of prev state and tanh candidate."""
        cell = GRUCell(input_dim=10, hidden_dim=100, w_in_scale=0.1, spectral_radius=0.1)
        x = torch.randn(8, 10)
        h = torch.zeros(8, 100)
        for _ in range(20):
            h = cell([x], h)
        # With small weights the state should stay bounded
        assert torch.all(torch.isfinite(h))

    def test_state_changes_over_time(self) -> None:
        cell = GRUCell(input_dim=5, hidden_dim=20)
        x = torch.randn(2, 5)
        h = torch.zeros(2, 20)
        h_prev = h.clone()
        h = cell([x], h)
        assert not torch.allclose(h, h_prev)

    def test_no_grad_on_output(self) -> None:
        """Buffers should not accumulate gradients."""
        cell = GRUCell(input_dim=5, hidden_dim=20)
        x = torch.randn(2, 5, requires_grad=True)
        h = torch.zeros(2, 20)
        new_h = cell([x], h)
        new_h.sum().backward()
        # x should have a gradient; cell buffers should not
        assert x.grad is not None
        assert cell.W_z.grad is None
        assert cell.U_z.grad is None

    def test_gate_bias_init_shifts_output(self) -> None:
        """gate_bias_init shifts gate activations, changing output."""
        torch.manual_seed(0)
        cell_neutral = GRUCell(input_dim=5, hidden_dim=20, gate_bias_init=0.0)
        torch.manual_seed(0)
        cell_open = GRUCell(input_dim=5, hidden_dim=20, gate_bias_init=5.0)
        x = torch.randn(2, 5)
        h = torch.zeros(2, 20)
        out_neutral = cell_neutral([x], h)
        out_open = cell_open([x], h)
        assert not torch.allclose(out_neutral, out_open)


# ---------------------------------------------------------------------------
# GRULayer tests
# ---------------------------------------------------------------------------


class TestGRULayerInstantiation:
    """Test GRULayer construction."""

    def test_basic_instantiation(self) -> None:
        layer = GRULayer(input_dim=10, hidden_dim=100)
        assert layer.hidden_dim == 100
        assert layer.state_size == 100
        assert layer.state is None

    def test_cell_type(self) -> None:
        layer = GRULayer(input_dim=10, hidden_dim=50)
        assert isinstance(layer.cell, GRUCell)

    def test_no_trainable_parameters(self) -> None:
        layer = GRULayer(input_dim=10, hidden_dim=50)
        assert sum(1 for _ in layer.parameters()) == 0

    def test_kwargs_forwarded_to_cell(self) -> None:
        layer = GRULayer(
            input_dim=10, hidden_dim=50, w_in_scale=0.5, spectral_radius=0.7, bias=False
        )
        assert layer.cell._has_bias is False
        # Spectral radius on U_z
        eigvals = torch.linalg.eigvals(layer.cell.U_z)
        actual_sr = torch.max(torch.abs(eigvals)).item()
        assert abs(actual_sr - 0.7) < 0.01


class TestGRULayerForward:
    """Test forward pass over sequences."""

    def test_output_shape_feedback_only(self) -> None:
        layer = GRULayer(input_dim=10, hidden_dim=200)
        feedback = torch.randn(4, 50, 10)
        states = layer(feedback)
        assert states.shape == (4, 50, 200)

    def test_output_shape_with_driving_input(self) -> None:
        """input_dim = feedback_dim + driving_dim."""
        layer = GRULayer(input_dim=15, hidden_dim=200)
        feedback = torch.randn(4, 50, 10)
        driving = torch.randn(4, 50, 5)
        states = layer(feedback, driving)
        assert states.shape == (4, 50, 200)

    def test_invalid_feedback_dimension_raises(self) -> None:
        layer = GRULayer(input_dim=10, hidden_dim=50)
        with pytest.raises(ValueError, match="Feedback must be 3D"):
            layer(torch.randn(4, 10))

    def test_driving_batch_mismatch_raises(self) -> None:
        layer = GRULayer(input_dim=15, hidden_dim=50)
        fb = torch.randn(2, 10, 10)
        drv = torch.randn(3, 10, 5)
        with pytest.raises(ValueError, match="match feedback dimensions"):
            layer(fb, drv)

    def test_output_finite(self) -> None:
        layer = GRULayer(input_dim=10, hidden_dim=100)
        feedback = torch.randn(2, 30, 10)
        states = layer(feedback)
        assert torch.all(torch.isfinite(states))


class TestGRULayerStateful:
    """Test state management mirrors ESNLayer behaviour."""

    def test_state_initialised_after_forward(self) -> None:
        layer = GRULayer(input_dim=10, hidden_dim=50)
        assert layer.state is None
        layer(torch.randn(2, 5, 10))
        assert layer.state is not None
        assert layer.state.shape == (2, 50)

    def test_state_persists_across_calls(self) -> None:
        layer = GRULayer(input_dim=10, hidden_dim=50)
        fb1 = torch.randn(2, 5, 10)
        fb2 = torch.randn(2, 5, 10)
        layer(fb1)
        state_after_1 = layer.get_state()
        layer(fb2)
        state_after_2 = layer.get_state()
        assert not torch.allclose(state_after_1, state_after_2)

    def test_reset_state_none(self) -> None:
        layer = GRULayer(input_dim=10, hidden_dim=50)
        layer(torch.randn(2, 5, 10))
        layer.reset_state()
        assert layer.state is None

    def test_reset_state_with_batch_size(self) -> None:
        layer = GRULayer(input_dim=10, hidden_dim=50)
        layer(torch.randn(2, 5, 10))  # Ensure buffers exist on a device
        layer.reset_state(batch_size=3)
        assert layer.state is not None
        assert layer.state.shape == (3, 50)
        assert torch.all(layer.state == 0.0)

    def test_reset_and_replay_is_deterministic(self) -> None:
        layer = GRULayer(input_dim=10, hidden_dim=50)
        fb = torch.randn(2, 5, 10)
        out1 = layer(fb)
        layer.reset_state(batch_size=2)
        out2 = layer(fb)
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_get_state_returns_clone(self) -> None:
        layer = GRULayer(input_dim=10, hidden_dim=50)
        layer(torch.randn(2, 5, 10))
        s1 = layer.get_state()
        s2 = layer.get_state()
        assert s1 is not s2
        assert torch.allclose(s1, s2)

    def test_get_state_before_init_is_none(self) -> None:
        layer = GRULayer(input_dim=10, hidden_dim=50)
        assert layer.get_state() is None

    def test_set_state(self) -> None:
        layer = GRULayer(input_dim=10, hidden_dim=50)
        custom = torch.randn(2, 50)
        layer.set_state(custom)
        assert torch.allclose(layer.state, custom)

    def test_set_state_wrong_size_raises(self) -> None:
        layer = GRULayer(input_dim=10, hidden_dim=50)
        with pytest.raises(ValueError, match="State size mismatch"):
            layer.set_state(torch.randn(2, 40))

    def test_set_random_state(self) -> None:
        layer = GRULayer(input_dim=10, hidden_dim=50)
        layer.reset_state(batch_size=2)
        layer.set_random_state()
        assert layer.state is not None
        # Random state should not be all zeros
        assert not torch.all(layer.state == 0.0)

    def test_set_random_state_before_init_raises(self) -> None:
        layer = GRULayer(input_dim=10, hidden_dim=50)
        with pytest.raises(RuntimeError, match="Reservoir not initialized"):
            layer.set_random_state()


class TestGRULayerAttributeDelegation:
    """Test that unknown attributes are delegated to the inner cell."""

    def test_hidden_dim_delegated(self) -> None:
        layer = GRULayer(input_dim=10, hidden_dim=64)
        assert layer.hidden_dim == 64

    def test_input_dim_delegated(self) -> None:
        layer = GRULayer(input_dim=7, hidden_dim=32)
        assert layer.input_dim == 7

    def test_buffers_accessible(self) -> None:
        layer = GRULayer(input_dim=10, hidden_dim=32)
        assert layer.W_z.shape == (32, 10)
        assert layer.U_z.shape == (32, 32)

    def test_missing_attr_raises(self) -> None:
        layer = GRULayer(input_dim=10, hidden_dim=32)
        with pytest.raises(AttributeError):
            _ = layer.nonexistent_attribute


class TestGRULayerDevice:
    """Test device handling."""

    def test_cpu_output(self) -> None:
        layer = GRULayer(input_dim=10, hidden_dim=50)
        states = layer(torch.randn(2, 5, 10))
        assert states.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_output(self) -> None:
        layer = GRULayer(input_dim=10, hidden_dim=50).to("cuda")
        fb = torch.randn(2, 5, 10, device="cuda")
        states = layer(fb)
        assert states.device.type == "cuda"


class TestGRULayerRepr:
    """Test string representation."""

    def test_repr_contains_key_info(self) -> None:
        layer = GRULayer(input_dim=10, hidden_dim=100)
        r = repr(layer)
        assert "GRUCell" in r
        assert "input_dim=10" in r
        assert "hidden_dim=100" in r


class TestGRUCellVsESNDynamics:
    """Verify that GRU and ESN dynamics differ (they should)."""

    def test_gru_and_esn_differ(self) -> None:
        from resdag.layers import ESNLayer

        torch.manual_seed(0)
        gru = GRULayer(input_dim=5, hidden_dim=50)

        torch.manual_seed(0)
        esn = ESNLayer(reservoir_size=50, feedback_size=5)

        fb = torch.randn(2, 20, 5)
        gru_out = gru(fb)
        esn_out = esn(fb)

        # Different architectures must produce different outputs
        assert not torch.allclose(gru_out, esn_out)
