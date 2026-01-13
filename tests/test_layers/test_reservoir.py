"""Unit tests for ReservoirLayer."""

import pytest
import torch

from resdag.layers.reservoir import ReservoirLayer


class TestReservoirLayerInstantiation:
    """Test ReservoirLayer instantiation and configuration."""

    def test_instantiation_feedback_only(self) -> None:
        """Test creating reservoir with feedback only."""
        reservoir = ReservoirLayer(reservoir_size=100, feedback_size=10)

        assert reservoir.reservoir_size == 100
        assert reservoir.feedback_size == 10
        assert reservoir.input_size is None
        assert reservoir.spectral_radius is None  # default is None
        assert reservoir._initialized is True
        assert reservoir.weight_feedback.shape == (100, 10)
        assert reservoir.weight_hh.shape == (100, 100)
        assert reservoir.weight_input is None

    def test_instantiation_with_driving_inputs(self) -> None:
        """Test creating reservoir with feedback and driving inputs."""
        reservoir = ReservoirLayer(reservoir_size=100, feedback_size=10, input_size=5)

        assert reservoir.feedback_size == 10
        assert reservoir.input_size == 5
        assert reservoir.weight_feedback.shape == (100, 10)
        assert reservoir.weight_input.shape == (100, 5)
        assert reservoir.weight_hh.shape == (100, 100)

    def test_custom_parameters(self) -> None:
        """Test custom spectral radius, activation, leak rate, etc."""
        reservoir = ReservoirLayer(
            reservoir_size=50,
            feedback_size=5,
            input_size=3,
            spectral_radius=0.8,
            bias=False,
            activation="relu",
            leak_rate=0.5,
        )

        assert reservoir.spectral_radius == 0.8
        assert reservoir.leak_rate == 0.5
        assert reservoir.bias_h is None

    def test_invalid_activation_raises_error(self) -> None:
        """Test that invalid activation name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown activation"):
            ReservoirLayer(reservoir_size=100, feedback_size=10, activation="invalid")


class TestReservoirLayerForwardPass:
    """Test forward pass behavior."""

    def test_forward_feedback_only(self) -> None:
        """Test forward pass with feedback only."""
        reservoir = ReservoirLayer(reservoir_size=50, feedback_size=10)
        feedback = torch.randn(2, 20, 10)  # (batch=2, seq=20, feedback=10)

        output = reservoir(feedback)

        assert output.shape == (2, 20, 50)
        assert isinstance(output, torch.Tensor)

    def test_forward_with_driving_inputs(self) -> None:
        """Test forward pass with feedback and driving inputs."""
        reservoir = ReservoirLayer(reservoir_size=100, feedback_size=10, input_size=5)

        feedback = torch.randn(3, 15, 10)  # (batch=3, seq=15, feedback=10)
        driving = torch.randn(3, 15, 5)  # (batch=3, seq=15, input=5)

        output = reservoir(feedback, driving)

        assert output.shape == (3, 15, 100)

    def test_forward_with_multiple_driving_inputs(self) -> None:
        """Test forward with feedback and single driving input (multiple not supported)."""
        reservoir = ReservoirLayer(reservoir_size=100, feedback_size=10, input_size=8)

        feedback = torch.randn(2, 10, 10)
        driving = torch.randn(2, 10, 8)  # Single driving input matching input_size

        output = reservoir(feedback, driving)

        assert output.shape == (2, 10, 100)

    def test_forward_invalid_feedback_dimensions_raises_error(self) -> None:
        """Test that non-3D feedback raises error."""
        reservoir = ReservoirLayer(reservoir_size=50, feedback_size=10)

        with pytest.raises(ValueError, match="Feedback must be 3D"):
            reservoir(torch.randn(2, 10))  # Only 2D

    def test_forward_feedback_size_mismatch_raises_error(self) -> None:
        """Test that wrong feedback size raises error."""
        reservoir = ReservoirLayer(reservoir_size=50, feedback_size=10)
        feedback = torch.randn(2, 20, 8)  # Wrong size!

        with pytest.raises(ValueError, match="Feedback size mismatch"):
            reservoir(feedback)

    def test_forward_inconsistent_batch_sizes_raises_error(self) -> None:
        """Test that inconsistent batch sizes raise error."""
        reservoir = ReservoirLayer(reservoir_size=50, feedback_size=10, input_size=5)

        feedback = torch.randn(2, 10, 10)
        driving = torch.randn(3, 10, 5)  # Different batch size!

        with pytest.raises(ValueError, match="match feedback dimensions"):
            reservoir(feedback, driving)

    def test_forward_driving_without_input_size_raises_error(self) -> None:
        """Test that providing driving inputs without input_size raises error."""
        reservoir = ReservoirLayer(reservoir_size=50, feedback_size=10)  # No input_size

        feedback = torch.randn(2, 10, 10)
        driving = torch.randn(2, 10, 5)

        with pytest.raises(ValueError, match="without input_size"):
            reservoir(feedback, driving)

    def test_forward_driving_size_mismatch_raises_error(self) -> None:
        """Test that wrong driving input size raises error."""
        reservoir = ReservoirLayer(reservoir_size=50, feedback_size=10, input_size=5)

        feedback = torch.randn(2, 10, 10)
        driving = torch.randn(2, 10, 8)  # Wrong size!

        with pytest.raises(ValueError, match="Driving input size mismatch"):
            reservoir(feedback, driving)


class TestReservoirLayerStatefulBehavior:
    """Test stateful behavior (state persistence, reset, set)."""

    def test_state_persistence_across_forward_passes(self) -> None:
        """Test that state persists across multiple forward passes."""
        reservoir = ReservoirLayer(reservoir_size=50, feedback_size=10)

        fb1 = torch.randn(2, 5, 10)
        fb2 = torch.randn(2, 5, 10)

        # First forward pass
        out1 = reservoir(fb1)
        state_after_fb1 = reservoir.get_state()

        # Second forward pass (state should carry over)
        out2 = reservoir(fb2)
        state_after_fb2 = reservoir.get_state()

        # States should be different
        assert not torch.allclose(state_after_fb1, state_after_fb2)

        # Reset and run fb1 again - should get same initial evolution
        reservoir.reset_state(batch_size=2)
        out1_again = reservoir(fb1)

        # First output should be similar (small numerical differences OK)
        assert torch.allclose(out1, out1_again, rtol=1e-5, atol=1e-6)

    def test_reset_state_without_batch_size(self) -> None:
        """Test reset_state() with no arguments sets state to None."""
        reservoir = ReservoirLayer(reservoir_size=50, feedback_size=10)
        feedback = torch.randn(2, 5, 10)

        reservoir(feedback)
        assert reservoir.state is not None

        reservoir.reset_state()
        assert reservoir.state is None

    def test_reset_state_with_batch_size(self) -> None:
        """Test reset_state() with batch_size initializes zeros."""
        reservoir = ReservoirLayer(reservoir_size=50, feedback_size=10)

        reservoir.reset_state(batch_size=3)

        assert reservoir.state is not None
        assert reservoir.state.shape == (3, 50)
        assert torch.allclose(reservoir.state, torch.zeros(3, 50))

    def test_set_state(self) -> None:
        """Test set_state() sets internal state."""
        reservoir = ReservoirLayer(reservoir_size=50, feedback_size=10)

        custom_state = torch.randn(2, 50)
        reservoir.set_state(custom_state)

        assert reservoir.state is not None
        assert torch.allclose(reservoir.state, custom_state)

    def test_set_state_invalid_shape_raises_error(self) -> None:
        """Test set_state() with wrong shape raises error."""
        reservoir = ReservoirLayer(reservoir_size=50, feedback_size=10)

        wrong_state = torch.randn(2, 40)  # Wrong reservoir size!

        with pytest.raises(ValueError, match="State size mismatch"):
            reservoir.set_state(wrong_state)

    def test_get_state(self) -> None:
        """Test get_state() returns copy of state."""
        reservoir = ReservoirLayer(reservoir_size=50, feedback_size=10)
        feedback = torch.randn(2, 5, 10)

        reservoir(feedback)
        state1 = reservoir.get_state()
        state2 = reservoir.get_state()

        # Should return clones (not same object)
        assert state1 is not state2
        assert torch.allclose(state1, state2)

    def test_get_state_before_initialization_returns_none(self) -> None:
        """Test get_state() returns None if state not initialized."""
        reservoir = ReservoirLayer(reservoir_size=50, feedback_size=10)

        state = reservoir.get_state()
        assert state is None


class TestReservoirLayerActivations:
    """Test different activation functions."""

    @pytest.mark.parametrize("activation", ["tanh", "relu", "sigmoid", "identity"])
    def test_different_activations(self, activation: str) -> None:
        """Test reservoir works with different activations."""
        reservoir = ReservoirLayer(reservoir_size=50, feedback_size=10, activation=activation)
        feedback = torch.randn(2, 5, 10)

        output = reservoir(feedback)

        assert output.shape == (2, 5, 50)

        # Check activation is applied correctly
        if activation == "identity":
            # For identity, output can be any value
            pass
        elif activation == "relu":
            # All values should be non-negative
            assert torch.all(output >= 0)
        elif activation == "tanh":
            # All values should be in [-1, 1]
            assert torch.all(output >= -1.0) and torch.all(output <= 1.0)
        elif activation == "sigmoid":
            # All values should be in [0, 1]
            assert torch.all(output >= 0.0) and torch.all(output <= 1.0)


class TestReservoirLayerSpectralRadius:
    """Test spectral radius scaling."""

    def test_spectral_radius_scaling(self) -> None:
        """Test that recurrent weights are scaled to target spectral radius."""
        reservoir = ReservoirLayer(reservoir_size=50, feedback_size=10, spectral_radius=0.8)

        # Compute actual spectral radius
        eigenvalues = torch.linalg.eigvals(reservoir.weight_hh.data)
        actual_spectral_radius = torch.max(torch.abs(eigenvalues)).item()

        # Should be close to target (within numerical precision)
        assert abs(actual_spectral_radius - 0.8) < 0.01

    def test_different_spectral_radii(self) -> None:
        """Test different spectral radius values."""
        for target_sr in [0.5, 0.9, 1.2]:
            reservoir = ReservoirLayer(
                reservoir_size=50, feedback_size=10, spectral_radius=target_sr
            )

            eigenvalues = torch.linalg.eigvals(reservoir.weight_hh.data)
            actual_sr = torch.max(torch.abs(eigenvalues)).item()

            assert abs(actual_sr - target_sr) < 0.01


class TestReservoirLayerLeakRate:
    """Test leaky integration."""

    def test_leak_rate_affects_output(self) -> None:
        """Test that leak rate changes output."""
        torch.manual_seed(42)
        reservoir_no_leak = ReservoirLayer(reservoir_size=50, feedback_size=10, leak_rate=1.0)

        torch.manual_seed(42)
        reservoir_with_leak = ReservoirLayer(reservoir_size=50, feedback_size=10, leak_rate=0.5)

        feedback = torch.randn(2, 10, 10)

        out_no_leak = reservoir_no_leak(feedback)
        out_with_leak = reservoir_with_leak(feedback)

        # Outputs should be different
        assert not torch.allclose(out_no_leak, out_with_leak)


class TestReservoirLayerDevice:
    """Test device handling."""

    def test_cpu_device(self) -> None:
        """Test reservoir works on CPU."""
        reservoir = ReservoirLayer(reservoir_size=50, feedback_size=10)
        feedback = torch.randn(2, 5, 10)

        output = reservoir(feedback)

        assert output.device.type == "cpu"

    def test_to_device(self) -> None:
        """Test moving reservoir to different device."""
        reservoir = ReservoirLayer(reservoir_size=50, feedback_size=10)

        # Move to CPU explicitly
        reservoir_cpu = reservoir.to("cpu")
        assert next(reservoir_cpu.parameters()).device.type == "cpu"

        # GPU test (skip if not available)
        if torch.cuda.is_available():
            reservoir_gpu = reservoir.to("cuda")
            assert next(reservoir_gpu.parameters()).device.type == "cuda"

            feedback_gpu = torch.randn(2, 5, 10, device="cuda")
            output_gpu = reservoir_gpu(feedback_gpu)
            assert output_gpu.device.type == "cuda"


class TestReservoirLayerGradients:
    """Test gradient computation."""

    def test_gradients_flow_through_reservoir(self) -> None:
        """Test that gradients flow through the reservoir."""
        reservoir = ReservoirLayer(reservoir_size=50, feedback_size=10, trainable=True)
        feedback = torch.randn(2, 5, 10, requires_grad=True)

        output = reservoir(feedback)
        loss = output.sum()
        loss.backward()

        # Input should have gradients
        assert feedback.grad is not None

        # Weights should have gradients
        assert reservoir.weight_feedback.grad is not None
        assert reservoir.weight_hh.grad is not None


class TestReservoirLayerRepr:
    """Test string representation."""

    def test_repr_feedback_only(self) -> None:
        """Test __repr__ for feedback-only reservoir."""
        reservoir = ReservoirLayer(reservoir_size=100, feedback_size=20, spectral_radius=0.95)

        repr_str = repr(reservoir)

        assert "ReservoirLayer" in repr_str
        assert "reservoir_size=100" in repr_str
        assert "feedback_size=20" in repr_str
        assert "spectral_radius=0.95" in repr_str

    def test_repr_with_driving_inputs(self) -> None:
        """Test __repr__ for reservoir with driving inputs."""
        reservoir = ReservoirLayer(
            reservoir_size=100, feedback_size=20, input_size=5, spectral_radius=0.95
        )

        repr_str = repr(reservoir)

        assert "feedback_size=20" in repr_str
        assert "input_size=5" in repr_str
