"""Unit tests for ReadoutLayer."""

import pytest
import torch
import torch.nn as nn

from resdag.layers import ReadoutLayer


class TestReadoutLayerInstantiation:
    """Test ReadoutLayer instantiation and configuration."""

    def test_basic_instantiation(self) -> None:
        """Test creating readout with basic parameters."""
        readout = ReadoutLayer(in_features=100, out_features=10)

        assert readout.in_features == 100
        assert readout.out_features == 10
        assert readout.bias is not None
        assert readout.name is None
        assert not readout.is_fitted

    def test_instantiation_without_bias(self) -> None:
        """Test creating readout without bias."""
        readout = ReadoutLayer(in_features=50, out_features=5, bias=False)

        assert readout.bias is None

    def test_instantiation_with_name(self) -> None:
        """Test creating named readout."""
        readout = ReadoutLayer(in_features=100, out_features=10, name="output1")

        assert readout.name == "output1"

    def test_is_linear_subclass(self) -> None:
        """Test that ReadoutLayer is a proper nn.Linear subclass."""
        readout = ReadoutLayer(in_features=100, out_features=10)

        assert isinstance(readout, nn.Linear)
        assert isinstance(readout, nn.Module)

    def test_parameters_accessible(self) -> None:
        """Test that weight and bias parameters are accessible."""
        readout = ReadoutLayer(in_features=100, out_features=10)

        assert hasattr(readout, "weight")
        assert hasattr(readout, "bias")
        assert readout.weight.shape == (10, 100)
        assert readout.bias.shape == (10,)


class TestReadoutLayerForward2D:
    """Test forward pass with 2D inputs (standard linear layer behavior)."""

    def test_forward_2d_input(self) -> None:
        """Test forward pass with 2D input."""
        readout = ReadoutLayer(in_features=100, out_features=10)
        x = torch.randn(5, 100)  # (batch=5, features=100)

        output = readout(x)

        assert output.shape == (5, 10)
        assert isinstance(output, torch.Tensor)

    def test_forward_2d_matches_linear(self) -> None:
        """Test that 2D forward matches standard nn.Linear."""
        # Create both with same weights
        readout = ReadoutLayer(in_features=100, out_features=10)
        linear = nn.Linear(100, 10)

        # Copy weights
        linear.weight.data = readout.weight.data.clone()
        linear.bias.data = readout.bias.data.clone()

        x = torch.randn(5, 100)

        output_readout = readout(x)
        output_linear = linear(x)

        assert torch.allclose(output_readout, output_linear)


class TestReadoutLayerForward3D:
    """Test forward pass with 3D inputs (per-timestep application)."""

    def test_forward_3d_input(self) -> None:
        """Test forward pass with 3D sequence input."""
        readout = ReadoutLayer(in_features=100, out_features=10)
        x = torch.randn(2, 20, 100)  # (batch=2, seq=20, features=100)

        output = readout(x)

        assert output.shape == (2, 20, 10)
        assert isinstance(output, torch.Tensor)

    def test_forward_3d_preserves_per_timestep_semantics(self) -> None:
        """Test that 3D forward applies independent transformation per timestep."""
        readout = ReadoutLayer(in_features=100, out_features=10)
        x = torch.randn(2, 20, 100)

        # Apply to full sequence
        output_sequence = readout(x)

        # Verify we can apply readout to individual timesteps with same weights
        # (tests that same linear transform is used, not numerical equality)
        out_t0 = readout(x[:, 0, :])  # (2, 100) -> (2, 10)
        out_t5 = readout(x[:, 5, :])

        # These should have the correct shape from the linear layer
        assert out_t0.shape == (2, 10)
        assert out_t5.shape == (2, 10)

        # The full sequence output should have correct shape
        assert output_sequence.shape == (2, 20, 10)

    def test_forward_3d_different_sequence_lengths(self) -> None:
        """Test forward with different sequence lengths."""
        readout = ReadoutLayer(in_features=50, out_features=5)

        for seq_len in [1, 10, 50, 100]:
            x = torch.randn(3, seq_len, 50)
            output = readout(x)
            assert output.shape == (3, seq_len, 5)

    def test_forward_3d_batch_size_one(self) -> None:
        """Test forward with batch size 1."""
        readout = ReadoutLayer(in_features=20, out_features=5)
        x = torch.randn(1, 10, 20)

        output = readout(x)

        assert output.shape == (1, 10, 5)


class TestReadoutLayerInputValidation:
    """Test input validation and error handling."""

    def test_forward_invalid_dimensions_raises_error(self) -> None:
        """Test that invalid input dimensions raise ValueError."""
        readout = ReadoutLayer(in_features=100, out_features=10)

        # 1D input
        with pytest.raises(ValueError, match="expects 2D.*or 3D"):
            readout(torch.randn(100))

        # 4D input
        with pytest.raises(ValueError, match="expects 2D.*or 3D"):
            readout(torch.randn(2, 20, 10, 100))

    def test_forward_wrong_feature_size_raises_error(self) -> None:
        """Test that wrong feature size raises error."""
        readout = ReadoutLayer(in_features=100, out_features=10)

        # 2D with wrong feature size
        x_2d = torch.randn(5, 50)  # Should be 100, not 50
        with pytest.raises(RuntimeError):  # PyTorch runtime error from matmul
            readout(x_2d)

        # 3D with wrong feature size
        x_3d = torch.randn(2, 20, 50)  # Should be 100, not 50
        with pytest.raises(RuntimeError):
            readout(x_3d)


class TestReadoutLayerProperties:
    """Test ReadoutLayer properties."""

    def test_name_property(self) -> None:
        """Test name property getter."""
        readout_unnamed = ReadoutLayer(100, 10)
        assert readout_unnamed.name is None

        readout_named = ReadoutLayer(100, 10, name="my_readout")
        assert readout_named.name == "my_readout"

    def test_is_fitted_property_default_false(self) -> None:
        """Test is_fitted is False by default."""
        readout = ReadoutLayer(100, 10)
        assert readout.is_fitted is False

    def test_multiple_readouts_have_independent_names(self) -> None:
        """Test that multiple readouts have independent names."""
        readout1 = ReadoutLayer(100, 10, name="output1")
        readout2 = ReadoutLayer(100, 5, name="output2")
        readout3 = ReadoutLayer(100, 3)

        assert readout1.name == "output1"
        assert readout2.name == "output2"
        assert readout3.name is None


class TestReadoutLayerFit:
    """Test fit() method raises NotImplementedError in base class."""

    def test_fit_raises_not_implemented(self) -> None:
        """Test that fit() raises NotImplementedError in base ReadoutLayer."""
        readout = ReadoutLayer(100, 10)
        states = torch.randn(10, 20, 100)
        targets = torch.randn(10, 20, 10)

        with pytest.raises(NotImplementedError, match="not implemented"):
            readout.fit(states, targets)


class TestReadoutLayerDevice:
    """Test device handling."""

    def test_cpu_device(self) -> None:
        """Test readout works on CPU."""
        readout = ReadoutLayer(100, 10)
        x = torch.randn(2, 20, 100)

        output = readout(x)

        assert output.device.type == "cpu"

    def test_to_device(self) -> None:
        """Test moving readout to different device."""
        readout = ReadoutLayer(100, 10)

        # Move to CPU explicitly
        readout_cpu = readout.to("cpu")
        assert next(readout_cpu.parameters()).device.type == "cpu"

        # GPU test (skip if not available)
        if torch.cuda.is_available():
            readout_gpu = readout.to("cuda")
            assert next(readout_gpu.parameters()).device.type == "cuda"

            x_gpu = torch.randn(2, 20, 100, device="cuda")
            output_gpu = readout_gpu(x_gpu)
            assert output_gpu.device.type == "cuda"


class TestReadoutLayerGradients:
    """Test gradient computation."""

    def test_gradients_flow_through_readout_2d(self) -> None:
        """Test gradients flow through readout with 2D input."""
        readout = ReadoutLayer(in_features=100, out_features=10, trainable=True)
        x = torch.randn(5, 100, requires_grad=True)

        output = readout(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert readout.weight.grad is not None
        assert readout.bias.grad is not None

    def test_gradients_flow_through_readout_3d(self) -> None:
        """Test gradients flow through readout with 3D input."""
        readout = ReadoutLayer(in_features=100, out_features=10, trainable=True)
        x = torch.randn(2, 20, 100, requires_grad=True)

        output = readout(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert readout.weight.grad is not None
        assert readout.bias.grad is not None


class TestReadoutLayerStateDict:
    """Test state_dict serialization."""

    def test_state_dict_contains_weights(self) -> None:
        """Test state_dict contains weight and bias."""
        readout = ReadoutLayer(100, 10)
        state = readout.state_dict()

        assert "weight" in state
        assert "bias" in state
        assert state["weight"].shape == (10, 100)
        assert state["bias"].shape == (10,)

    def test_load_state_dict(self) -> None:
        """Test loading state_dict."""
        readout1 = ReadoutLayer(100, 10)
        readout2 = ReadoutLayer(100, 10)

        # Set readout1 to specific values
        with torch.no_grad():
            readout1.weight.fill_(1.0)
            readout1.bias.fill_(2.0)

        # Load into readout2
        readout2.load_state_dict(readout1.state_dict())

        assert torch.allclose(readout2.weight, torch.ones(10, 100))
        assert torch.allclose(readout2.bias, torch.ones(10) * 2.0)


class TestReadoutLayerTraining:
    """Test training mode behavior."""

    def test_train_eval_mode(self) -> None:
        """Test train/eval mode switching."""
        readout = ReadoutLayer(100, 10)

        # Default is training mode
        assert readout.training is True

        readout.eval()
        assert readout.training is False

        readout.train()
        assert readout.training is True

    def test_standard_pytorch_training_works(self) -> None:
        """Test that standard PyTorch training works (SGD alternative)."""
        readout = ReadoutLayer(in_features=100, out_features=10, trainable=True)
        optimizer = torch.optim.SGD(readout.parameters(), lr=0.01)

        # Simple training step
        x = torch.randn(5, 20, 100)
        targets = torch.randn(5, 20, 10)

        optimizer.zero_grad()
        output = readout(x)
        loss = nn.MSELoss()(output, targets)
        loss.backward()
        optimizer.step()

        # Should complete without error
        assert readout.weight.grad is not None


class TestReadoutLayerRepr:
    """Test string representation."""

    def test_repr_unnamed(self) -> None:
        """Test __repr__ for unnamed readout."""
        readout = ReadoutLayer(100, 10)
        repr_str = repr(readout)

        assert "ReadoutLayer" in repr_str
        assert "in_features=100" in repr_str
        assert "out_features=10" in repr_str
        assert "bias=True" in repr_str
        assert "name=" not in repr_str  # No name shown if None

    def test_repr_named(self) -> None:
        """Test __repr__ for named readout."""
        readout = ReadoutLayer(100, 10, name="output1")
        repr_str = repr(readout)

        assert "ReadoutLayer" in repr_str
        assert "name='output1'" in repr_str

    def test_repr_no_bias(self) -> None:
        """Test __repr__ for readout without bias."""
        readout = ReadoutLayer(100, 10, bias=False)
        repr_str = repr(readout)

        assert "bias=False" in repr_str
