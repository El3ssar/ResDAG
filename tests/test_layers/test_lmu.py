"""Unit tests for LMUCell and LMULayer."""

import warnings

import numpy as np
import pytest
import torch

from resdag.layers.cells.lmu_cell import LMUCell, _compute_legendre_matrices, _discretize
from resdag.layers.reservoirs.lmu import LMULayer


# ---------------------------------------------------------------------------
# Legendre matrix computation
# ---------------------------------------------------------------------------


class TestLegendrematrices:
    """Test analytical computation of A and B matrices."""

    def test_diagonal_pattern(self) -> None:
        """Diagonal of A should be -(2i+1) for i=0..order-1."""
        A, _ = _compute_legendre_matrices(4)
        np.testing.assert_array_equal(np.diag(A), [-1, -3, -5, -7])

    def test_above_diagonal_positive(self) -> None:
        """Entries above the diagonal (i < j) should be (2i+1)."""
        A, _ = _compute_legendre_matrices(4)
        for i in range(4):
            for j in range(i + 1, 4):
                expected = 2 * i + 1
                assert A[i, j] == expected, f"A[{i},{j}] should be {expected}, got {A[i,j]}"

    def test_below_diagonal_alternating(self) -> None:
        """Entries below diagonal (i > j) should be (2i+1)*(-1)^(i-j)."""
        A, _ = _compute_legendre_matrices(4)
        for i in range(4):
            for j in range(i):
                expected = (2 * i + 1) * ((-1) ** (i - j))
                assert A[i, j] == expected, f"A[{i},{j}] should be {expected}, got {A[i,j]}"

    def test_B_alternating_sign(self) -> None:
        """B[i] should be (2i+1)*(-1)^i."""
        _, B = _compute_legendre_matrices(4)
        expected = np.array([[1], [-3], [5], [-7]], dtype=np.float64)
        np.testing.assert_array_equal(B, expected)

    def test_shape(self) -> None:
        """Verify shapes for various orders."""
        for order in [1, 4, 8, 16]:
            A, B = _compute_legendre_matrices(order)
            assert A.shape == (order, order)
            assert B.shape == (order, 1)

    def test_order_1(self) -> None:
        """Edge case: order=1."""
        A, B = _compute_legendre_matrices(1)
        assert A[0, 0] == -1.0
        assert B[0, 0] == 1.0


# ---------------------------------------------------------------------------
# Discretization
# ---------------------------------------------------------------------------


class TestDiscretization:
    """Test ZOH and Euler discretization."""

    def test_zoh_shape(self) -> None:
        """ZOH output shapes are correct."""
        A, B = _compute_legendre_matrices(8)
        A_bar, B_bar = _discretize(A, B, theta=1.0, dt=1.0, method="zoh")
        assert A_bar.shape == (8, 8)
        assert B_bar.shape == (8, 1)

    def test_euler_shape(self) -> None:
        """Euler output shapes are correct."""
        A, B = _compute_legendre_matrices(8)
        A_bar, B_bar = _discretize(A, B, theta=1.0, dt=1.0, method="euler")
        assert A_bar.shape == (8, 8)
        assert B_bar.shape == (8, 1)

    def test_invalid_method_raises(self) -> None:
        A, B = _compute_legendre_matrices(4)
        with pytest.raises(ValueError, match="Unknown discretization method"):
            _discretize(A, B, theta=1.0, dt=1.0, method="runge_kutta")  # type: ignore[arg-type]

    def test_zoh_output_dtype_float64(self) -> None:
        """Discretized matrices are float64 (for numerical precision)."""
        A, B = _compute_legendre_matrices(4)
        A_bar, B_bar = _discretize(A, B, theta=1.0, dt=1.0, method="zoh")
        assert A_bar.dtype == torch.float64
        assert B_bar.dtype == torch.float64

    def test_euler_vs_zoh_differ(self) -> None:
        """Euler and ZOH should produce different results."""
        A, B = _compute_legendre_matrices(8)
        A_bar_zoh, _ = _discretize(A, B, theta=1.0, dt=1.0, method="zoh")
        A_bar_euler, _ = _discretize(A, B, theta=1.0, dt=1.0, method="euler")
        assert not torch.allclose(A_bar_zoh, A_bar_euler)

    def test_small_dt_euler_close_to_zoh(self) -> None:
        """For very small dt, Euler and ZOH should converge."""
        A, B = _compute_legendre_matrices(4)
        A_bar_zoh, B_bar_zoh = _discretize(A, B, theta=1.0, dt=1e-4, method="zoh")
        A_bar_euler, B_bar_euler = _discretize(A, B, theta=1.0, dt=1e-4, method="euler")
        assert torch.allclose(A_bar_zoh, A_bar_euler, atol=1e-6)
        assert torch.allclose(B_bar_zoh, B_bar_euler, atol=1e-6)


# ---------------------------------------------------------------------------
# LMUCell instantiation
# ---------------------------------------------------------------------------


class TestLMUCellInstantiation:
    """Test LMUCell construction and buffer registration."""

    def test_default_params(self) -> None:
        """Default parameters create valid cell."""
        cell = LMUCell(input_dim=3)
        assert cell.input_dim == 3
        assert cell.order == 8
        assert cell.theta == 1.0
        assert cell._dt == 1.0
        assert cell._discretization == "zoh"
        assert not cell._nonlinear_hidden

    def test_state_size_linear(self) -> None:
        """state_size = input_dim * order when nonlinear_hidden=False."""
        cell = LMUCell(input_dim=3, order=8)
        assert cell.state_size == 24
        assert cell.memory_dim == 24
        assert cell.output_dim == 24

    def test_state_size_nonlinear(self) -> None:
        """state_size = input_dim * order + hidden_dim when nonlinear_hidden=True."""
        cell = LMUCell(input_dim=3, order=8, nonlinear_hidden=True, hidden_dim=32)
        assert cell.state_size == 24 + 32
        assert cell.memory_dim == 24
        assert cell.output_dim == 56

    def test_buffers_registered(self) -> None:
        """A_bar and B_bar should be registered as buffers (not parameters)."""
        cell = LMUCell(input_dim=3, order=8)
        buffer_names = {name for name, _ in cell.named_buffers()}
        assert "A_bar" in buffer_names
        assert "B_bar" in buffer_names

    def test_no_parameters_linear_mode(self) -> None:
        """In linear mode, cell should have no trainable parameters."""
        cell = LMUCell(input_dim=3, order=8)
        params = list(cell.parameters())
        assert len(params) == 0

    def test_no_parameters_nonlinear_mode(self) -> None:
        """In nonlinear mode, all weights are still buffers (non-trainable)."""
        cell = LMUCell(input_dim=3, order=8, nonlinear_hidden=True, hidden_dim=32)
        params = list(cell.parameters())
        assert len(params) == 0

    def test_nonlinear_buffers_registered(self) -> None:
        """W_h, W_m, W_x, b are registered as buffers when nonlinear_hidden=True."""
        cell = LMUCell(input_dim=3, order=8, nonlinear_hidden=True, hidden_dim=32)
        buffer_names = {name for name, _ in cell.named_buffers()}
        assert "W_h" in buffer_names
        assert "W_m" in buffer_names
        assert "W_x" in buffer_names
        assert "b" in buffer_names

    def test_nonlinear_buffer_shapes(self) -> None:
        """Nonlinear weight buffers have the correct shapes."""
        cell = LMUCell(input_dim=3, order=8, nonlinear_hidden=True, hidden_dim=32)
        assert cell.W_h.shape == (32, 32)
        assert cell.W_m.shape == (32, 24)
        assert cell.W_x.shape == (32, 3)
        assert cell.b.shape == (32,)

    def test_A_bar_shape(self) -> None:
        """A_bar has shape (order, order)."""
        cell = LMUCell(input_dim=5, order=16)
        assert cell.A_bar.shape == (16, 16)

    def test_B_bar_shape(self) -> None:
        """B_bar has shape (order, 1)."""
        cell = LMUCell(input_dim=5, order=16)
        assert cell.B_bar.shape == (16, 1)

    def test_euler_discretization(self) -> None:
        """Cell can be created with Euler discretization."""
        cell = LMUCell(input_dim=3, order=8, discretization="euler")
        assert cell._discretization == "euler"

    def test_invalid_discretization_raises(self) -> None:
        """Invalid discretization raises ValueError."""
        with pytest.raises(ValueError, match="Unknown discretization"):
            LMUCell(input_dim=3, order=8, discretization="runge_kutta")  # type: ignore[arg-type]

    def test_large_order_warning(self) -> None:
        """Order > 256 should emit a UserWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            LMUCell(input_dim=1, order=257)
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "257" in str(w[0].message)

    def test_w_x_scale_applied(self) -> None:
        """w_x_scale scales the W_x buffer."""
        torch.manual_seed(42)
        cell_scale1 = LMUCell(input_dim=3, order=4, nonlinear_hidden=True, w_x_scale=1.0)
        torch.manual_seed(42)
        cell_scale2 = LMUCell(input_dim=3, order=4, nonlinear_hidden=True, w_x_scale=2.0)
        assert torch.allclose(cell_scale2.W_x, cell_scale1.W_x * 2.0)

    def test_W_h_spectral_radius_lt_1(self) -> None:
        """W_h should be scaled so that its spectral radius < 1."""
        cell = LMUCell(input_dim=3, order=4, nonlinear_hidden=True, hidden_dim=64)
        eigenvalues = torch.linalg.eigvals(cell.W_h)
        sr = torch.max(torch.abs(eigenvalues)).real.item()
        assert sr <= 1.0


# ---------------------------------------------------------------------------
# LMUCell forward pass
# ---------------------------------------------------------------------------


class TestLMUCellForward:
    """Test single-step forward computation."""

    def test_output_shape_linear(self) -> None:
        """Output has shape (batch, state_size)."""
        cell = LMUCell(input_dim=3, order=8)
        x = torch.randn(4, 3)
        state = torch.zeros(4, cell.state_size)
        out = cell([x], state)
        assert out.shape == (4, 24)

    def test_output_shape_nonlinear(self) -> None:
        """Output has shape (batch, state_size) with nonlinear mode."""
        cell = LMUCell(input_dim=3, order=8, nonlinear_hidden=True, hidden_dim=32)
        x = torch.randn(4, 3)
        state = torch.zeros(4, cell.state_size)
        out = cell([x], state)
        assert out.shape == (4, 56)

    def test_linear_update_is_correct(self) -> None:
        """Memory update follows the discretized LTI formula."""
        cell = LMUCell(input_dim=1, order=4)
        x = torch.ones(1, 1)
        m = torch.zeros(1, 4)

        out = cell([x], m)

        # Manual computation: m_new = 0 @ A_bar.T + 1 * B_bar.T = B_bar.T
        expected = cell.B_bar.T.repeat(1, 1)  # shape (1, 4)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_multiple_inputs_concatenated(self) -> None:
        """Inputs from multiple streams are concatenated before LMU update."""
        cell = LMUCell(input_dim=5, order=4)
        fb = torch.randn(2, 3)
        drv = torch.randn(2, 2)
        state = torch.zeros(2, cell.state_size)

        out_two = cell([fb, drv], state)

        # Equivalent to feeding the concatenated tensor as a single input
        x_full = torch.cat([fb, drv], dim=-1)
        out_one = cell([x_full], state)

        assert torch.allclose(out_two, out_one)

    def test_memory_update_linearity(self) -> None:
        """Memory update is linear: superposition holds."""
        cell = LMUCell(input_dim=1, order=8)
        state = torch.zeros(1, cell.state_size)

        x1 = torch.tensor([[2.0]])
        x2 = torch.tensor([[3.0]])

        # Double and triple from zero
        m1 = cell([x1], state)
        m2 = cell([x2], state)

        # Linearity: cell([a*x1 + b*x2], 0) = a*cell([x1], 0) + b*cell([x2], 0)
        x_sum = x1 + x2
        m_sum = cell([x_sum], state)
        assert torch.allclose(m_sum, m1 + m2, atol=1e-5)

    def test_state_dimension_used_correctly(self) -> None:
        """Non-zero initial state propagates correctly through A_bar."""
        cell = LMUCell(input_dim=1, order=4)
        state_nonzero = torch.ones(1, cell.state_size)
        x = torch.zeros(1, 1)

        out = cell([x], state_nonzero)

        # With zero input, m_new = m @ A_bar.T
        m = state_nonzero.reshape(1, 1, 4)
        expected = (m @ cell.A_bar.T).reshape(1, 4)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_output_no_activation_on_memory(self) -> None:
        """Linear memory portion should not be clamped to [-1, 1]."""
        # Feed large input — memory should grow beyond [-1, 1] if linear
        cell = LMUCell(input_dim=1, order=4)
        x = torch.full((1, 1), 1000.0)
        state = torch.zeros(1, cell.state_size)
        out = cell([x], state)
        # At least one memory value should exceed 1 in magnitude for large input
        assert torch.any(torch.abs(out) > 1.0)

    def test_nonlinear_output_bounded(self) -> None:
        """Hidden portion is bounded by tanh to (-1, 1)."""
        cell = LMUCell(input_dim=3, order=8, nonlinear_hidden=True, hidden_dim=32)
        x = torch.randn(4, 3) * 100.0  # Large input to saturate
        state = torch.zeros(4, cell.state_size)
        out = cell([x], state)
        h = out[:, cell.memory_dim :]
        assert torch.all(h >= -1.0) and torch.all(h <= 1.0)

    def test_deterministic_with_fixed_seed(self) -> None:
        """Same inputs and state produce the same output."""
        cell = LMUCell(input_dim=3, order=8)
        x = torch.randn(2, 3)
        state = torch.zeros(2, cell.state_size)
        out1 = cell([x], state)
        out2 = cell([x], state)
        assert torch.allclose(out1, out2)

    def test_repr(self) -> None:
        """__repr__ contains key info."""
        cell = LMUCell(input_dim=3, order=8, theta=2.0, discretization="euler")
        r = repr(cell)
        assert "LMUCell" in r
        assert "input_dim=3" in r
        assert "order=8" in r
        assert "theta=2.0" in r
        assert "euler" in r


# ---------------------------------------------------------------------------
# LMULayer instantiation
# ---------------------------------------------------------------------------


class TestLMULayerInstantiation:
    """Test LMULayer construction."""

    def test_default_params(self) -> None:
        """Default parameters create a valid layer."""
        lmu = LMULayer(feedback_size=3)
        assert lmu.feedback_size == 3
        assert lmu.input_size is None

    def test_with_driving_input(self) -> None:
        """Layer with driving input configures input_dim correctly."""
        lmu = LMULayer(feedback_size=3, input_size=2)
        assert lmu.input_size == 2
        assert lmu.cell.input_dim == 5  # 3 + 2

    def test_cell_attribute_delegation(self) -> None:
        """Layer delegates unknown attributes to cell."""
        lmu = LMULayer(feedback_size=3, order=16, theta=5.0)
        assert lmu.order == 16
        assert lmu.theta == 5.0
        assert lmu.memory_dim == 3 * 16
        assert lmu.state_size == 48

    def test_state_initially_none(self) -> None:
        """State is None before first forward pass."""
        lmu = LMULayer(feedback_size=3)
        assert lmu.get_state() is None

    def test_repr_delegates_to_cell(self) -> None:
        """__repr__ of LMULayer includes LMUCell info."""
        lmu = LMULayer(feedback_size=3, order=8)
        assert "LMUCell" in repr(lmu)

    def test_no_trainable_params(self) -> None:
        """LMULayer (linear mode) has no trainable parameters."""
        lmu = LMULayer(feedback_size=3, order=8)
        assert sum(p.numel() for p in lmu.parameters()) == 0


# ---------------------------------------------------------------------------
# LMULayer forward pass
# ---------------------------------------------------------------------------


class TestLMULayerForward:
    """Test sequence forward pass behavior."""

    def test_output_shape_feedback_only(self) -> None:
        """Output has shape (batch, seq, state_size)."""
        lmu = LMULayer(feedback_size=3, order=8)
        x = torch.randn(4, 50, 3)
        out = lmu(x)
        assert out.shape == (4, 50, 24)

    def test_output_shape_with_driving(self) -> None:
        """Output shape is correct with feedback + driving input."""
        lmu = LMULayer(feedback_size=3, input_size=2, order=8)
        fb = torch.randn(4, 50, 3)
        drv = torch.randn(4, 50, 2)
        out = lmu(fb, drv)
        assert out.shape == (4, 50, 40)

    def test_output_shape_nonlinear(self) -> None:
        """Output includes hidden_dim when nonlinear_hidden=True."""
        lmu = LMULayer(feedback_size=3, order=8, nonlinear_hidden=True, hidden_dim=32)
        x = torch.randn(4, 50, 3)
        out = lmu(x)
        assert out.shape == (4, 50, 56)

    def test_requires_3d_input(self) -> None:
        """2D feedback raises ValueError."""
        lmu = LMULayer(feedback_size=3)
        with pytest.raises(ValueError, match="Feedback must be 3D"):
            lmu(torch.randn(4, 3))

    def test_feedback_size_mismatch_raises(self) -> None:
        """Wrong feedback feature dim raises error from cell level."""
        lmu = LMULayer(feedback_size=3)
        # Cell.input_dim = 3; passing 4 features triggers shape error in cat/matmul
        with pytest.raises(Exception):
            lmu(torch.randn(2, 10, 4))  # 4 != 3

    def test_batch_size_1(self) -> None:
        """Works with batch_size=1."""
        lmu = LMULayer(feedback_size=3, order=4)
        out = lmu(torch.randn(1, 10, 3))
        assert out.shape == (1, 10, 12)

    def test_seq_len_1(self) -> None:
        """Works with seq_len=1."""
        lmu = LMULayer(feedback_size=3, order=4)
        out = lmu(torch.randn(4, 1, 3))
        assert out.shape == (4, 1, 12)

    def test_output_dtype_float32(self) -> None:
        """Output dtype matches input dtype."""
        lmu = LMULayer(feedback_size=3, order=4)
        x = torch.randn(2, 10, 3, dtype=torch.float32)
        out = lmu(x)
        assert out.dtype == torch.float32

    def test_output_is_tensor(self) -> None:
        """Output is a plain Tensor (not tuple)."""
        lmu = LMULayer(feedback_size=3)
        out = lmu(torch.randn(2, 5, 3))
        assert isinstance(out, torch.Tensor)


# ---------------------------------------------------------------------------
# LMULayer state management
# ---------------------------------------------------------------------------


class TestLMULayerStateMgmt:
    """Test state persistence, reset, set, and get."""

    def test_state_initialized_after_forward(self) -> None:
        """State is not None after a forward pass."""
        lmu = LMULayer(feedback_size=3)
        lmu(torch.randn(2, 5, 3))
        assert lmu.get_state() is not None

    def test_state_shape_after_forward(self) -> None:
        """State shape is (batch, state_size)."""
        lmu = LMULayer(feedback_size=3, order=8)
        lmu(torch.randn(2, 5, 3))
        state = lmu.get_state()
        assert state is not None
        assert state.shape == (2, 24)

    def test_state_persists_across_calls(self) -> None:
        """State carries over between forward calls."""
        lmu = LMULayer(feedback_size=3, order=4)
        lmu(torch.randn(2, 5, 3))
        state_mid = lmu.get_state()

        lmu(torch.randn(2, 5, 3))
        state_end = lmu.get_state()

        # States should differ after processing more data
        assert not torch.allclose(state_mid, state_end)

    def test_reset_sets_state_none(self) -> None:
        """reset_state() without args sets state to None."""
        lmu = LMULayer(feedback_size=3)
        lmu(torch.randn(2, 5, 3))
        lmu.reset_state()
        assert lmu.state is None

    def test_reset_with_batch_size_zeros(self) -> None:
        """reset_state(batch_size=k) initializes zero state of correct shape."""
        lmu = LMULayer(feedback_size=3, order=8)
        lmu.reset_state(batch_size=4)
        assert lmu.state is not None
        assert lmu.state.shape == (4, 24)
        assert torch.allclose(lmu.state, torch.zeros(4, 24))

    def test_reset_then_forward_reproducible(self) -> None:
        """After reset, same input yields same output."""
        lmu = LMULayer(feedback_size=3, order=4)
        x = torch.randn(2, 10, 3)

        lmu.reset_state(batch_size=2)
        out1 = lmu(x)

        lmu.reset_state(batch_size=2)
        out2 = lmu(x)

        assert torch.allclose(out1, out2)

    def test_set_state(self) -> None:
        """set_state() injects custom state."""
        lmu = LMULayer(feedback_size=3, order=4)
        custom = torch.randn(2, lmu.cell.state_size)
        lmu.set_state(custom)
        assert torch.allclose(lmu.get_state(), custom)

    def test_set_state_wrong_size_raises(self) -> None:
        """set_state() with wrong last dimension raises ValueError."""
        lmu = LMULayer(feedback_size=3, order=4)
        bad = torch.randn(2, 999)
        with pytest.raises(ValueError, match="State size mismatch"):
            lmu.set_state(bad)

    def test_get_state_returns_clone(self) -> None:
        """get_state() returns an independent clone."""
        lmu = LMULayer(feedback_size=3, order=4)
        lmu(torch.randn(2, 5, 3))
        s1 = lmu.get_state()
        s2 = lmu.get_state()
        assert s1 is not s2
        assert torch.allclose(s1, s2)

    def test_set_random_state(self) -> None:
        """set_random_state() produces non-zero state."""
        lmu = LMULayer(feedback_size=3, order=4)
        lmu.reset_state(batch_size=2)
        lmu.set_random_state()
        state = lmu.get_state()
        assert state is not None
        assert not torch.allclose(state, torch.zeros_like(state))

    def test_set_random_state_before_init_raises(self) -> None:
        """set_random_state() before initialization raises RuntimeError."""
        lmu = LMULayer(feedback_size=3, order=4)
        with pytest.raises(RuntimeError, match="not initialized"):
            lmu.set_random_state()


# ---------------------------------------------------------------------------
# LMULayer: theta / dt effect
# ---------------------------------------------------------------------------


class TestLMULayerTheta:
    """Verify that theta and dt affect the output."""

    def test_different_theta_different_output(self) -> None:
        """Different theta values produce different outputs."""
        x = torch.randn(2, 20, 3)

        lmu1 = LMULayer(feedback_size=3, order=8, theta=1.0)
        lmu2 = LMULayer(feedback_size=3, order=8, theta=10.0)

        out1 = lmu1(x)
        out2 = lmu2(x)
        assert not torch.allclose(out1, out2)

    def test_different_dt_different_output(self) -> None:
        """Different dt values produce different outputs."""
        x = torch.randn(2, 20, 3)

        lmu1 = LMULayer(feedback_size=3, order=8, dt=1.0)
        lmu2 = LMULayer(feedback_size=3, order=8, dt=0.1)

        out1 = lmu1(x)
        out2 = lmu2(x)
        assert not torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# LMULayer: discretization parity
# ---------------------------------------------------------------------------


class TestLMULayerDiscretization:
    """ZOH vs Euler discretization comparison."""

    def test_euler_produces_different_output(self) -> None:
        """Euler and ZOH modes give different results."""
        x = torch.randn(2, 20, 3)
        lmu_zoh = LMULayer(feedback_size=3, order=8, discretization="zoh")
        lmu_euler = LMULayer(feedback_size=3, order=8, discretization="euler")
        assert not torch.allclose(lmu_zoh(x), lmu_euler(x))

    def test_euler_output_shape(self) -> None:
        """Euler mode still produces correct output shape."""
        lmu = LMULayer(feedback_size=3, order=8, discretization="euler")
        out = lmu(torch.randn(2, 10, 3))
        assert out.shape == (2, 10, 24)


# ---------------------------------------------------------------------------
# LMULayer: CPU device
# ---------------------------------------------------------------------------


class TestLMULayerDevice:
    """Test device handling."""

    def test_cpu_device(self) -> None:
        """Runs on CPU without errors."""
        lmu = LMULayer(feedback_size=3, order=4)
        out = lmu(torch.randn(2, 5, 3))
        assert out.device.type == "cpu"

    def test_to_cpu(self) -> None:
        """Explicit .to('cpu') call works."""
        lmu = LMULayer(feedback_size=3, order=4).to("cpu")
        buf = next(lmu.cell.buffers())
        assert buf.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_cuda(self) -> None:
        """Can be moved to CUDA and runs correctly."""
        lmu = LMULayer(feedback_size=3, order=4).to("cuda")
        x = torch.randn(2, 5, 3, device="cuda")
        out = lmu(x)
        assert out.device.type == "cuda"
        assert out.shape == (2, 5, 12)


# ---------------------------------------------------------------------------
# Integration: LMULayer → ESN infrastructure
# ---------------------------------------------------------------------------


class TestLMULayerIntegration:
    """Test that LMULayer integrates with the broader ESN infrastructure."""

    def test_importable_from_layers(self) -> None:
        """LMULayer and LMUCell importable from resdag.layers."""
        from resdag.layers import LMUCell, LMULayer  # noqa: F401

    def test_importable_from_resdag(self) -> None:
        """LMULayer and LMUCell importable from top-level resdag package."""
        from resdag import LMUCell, LMULayer  # noqa: F401

    def test_importable_from_cells_subpackage(self) -> None:
        from resdag.layers.cells import LMUCell  # noqa: F401

    def test_importable_from_reservoirs_subpackage(self) -> None:
        from resdag.layers.reservoirs import LMULayer  # noqa: F401

    def test_composition_with_pytorch_symbolic(self) -> None:
        """LMULayer composes in a pytorch_symbolic DAG."""
        import pytorch_symbolic as ps

        from resdag import CGReadoutLayer, ESNModel

        inp = ps.Input((50, 3))
        reservoir = LMULayer(feedback_size=3, order=8)(inp)
        readout = CGReadoutLayer(24, 3, name="output")(reservoir)
        model = ESNModel(inp, readout)

        assert model is not None
        x = torch.randn(2, 50, 3)
        model.reset_reservoirs()
        # ESNModel.warmup needs at least a forward pass to be exercised
        states = model(x)
        assert states is not None

    def test_driving_input_batch_mismatch_raises(self) -> None:
        """Driving input with different batch size raises ValueError."""
        lmu = LMULayer(feedback_size=3, input_size=2)
        fb = torch.randn(2, 10, 3)
        drv = torch.randn(3, 10, 2)  # Different batch!
        with pytest.raises(ValueError, match="match feedback dimensions"):
            lmu(fb, drv)

    def test_multiple_driving_inputs_raises(self) -> None:
        """More than one driving input raises ValueError."""
        lmu = LMULayer(feedback_size=3, input_size=4)
        fb = torch.randn(2, 10, 3)
        drv1 = torch.randn(2, 10, 2)
        drv2 = torch.randn(2, 10, 2)
        with pytest.raises(ValueError, match="Only one driving input"):
            lmu(fb, drv1, drv2)
