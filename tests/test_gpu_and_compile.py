"""Tests for GPU support and torch.compile compatibility."""

import sys

import pytest
import torch

from resdag.init.input_feedback import ChebyshevInitializer, RandomInputInitializer
from resdag.layers import ReadoutLayer, ReservoirLayer

# torch.compile is not supported on Python 3.14+
COMPILE_SUPPORTED = torch.__version__ >= "2.0.0" and sys.version_info < (3, 14)


class TestGPUSupport:
    """Tests for GPU/CUDA support."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_reservoir_on_gpu(self):
        """Test ReservoirLayer works on GPU."""
        reservoir = ReservoirLayer(
            reservoir_size=100,
            feedback_size=10,
            topology="erdos_renyi",
            spectral_radius=0.9,
        )

        # Move to GPU
        reservoir = reservoir.cuda()

        # Verify all parameters are on GPU
        assert reservoir.weight_feedback.is_cuda
        assert reservoir.weight_hh.is_cuda
        if reservoir.bias_h is not None:
            assert reservoir.bias_h.is_cuda

        # Forward pass on GPU
        feedback = torch.randn(4, 20, 10, device="cuda")
        output = reservoir(feedback)

        assert output.is_cuda
        assert output.shape == (4, 20, 100)
        assert not torch.isnan(output).any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_reservoir_with_driving_inputs_on_gpu(self):
        """Test ReservoirLayer with driving inputs on GPU."""
        reservoir = ReservoirLayer(
            reservoir_size=100,
            feedback_size=10,
            input_size=5,
            topology="watts_strogatz",
        ).cuda()

        feedback = torch.randn(2, 15, 10, device="cuda")
        driving = torch.randn(2, 15, 5, device="cuda")

        output = reservoir(feedback, driving)

        assert output.is_cuda
        assert output.shape == (2, 15, 100)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_readout_on_gpu(self):
        """Test ReadoutLayer works on GPU."""
        readout = ReadoutLayer(in_features=100, out_features=10, name="output").cuda()

        # Verify parameters on GPU
        assert readout.weight.is_cuda
        assert readout.bias.is_cuda

        # Forward pass on GPU
        reservoir_output = torch.randn(4, 20, 100, device="cuda")
        output = readout(reservoir_output)

        assert output.is_cuda
        assert output.shape == (4, 20, 10)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_custom_initializers_on_gpu(self):
        """Test custom initializers work with GPU."""
        feedback_init = ChebyshevInitializer(p=0.3, k=3.5, input_scaling=0.8)
        input_init = RandomInputInitializer(input_scaling=1.0, seed=42)

        reservoir = ReservoirLayer(
            reservoir_size=100,
            feedback_size=10,
            input_size=5,
            feedback_initializer=feedback_init,
            input_initializer=input_init,
        ).cuda()

        # All parameters should be on GPU
        assert reservoir.weight_feedback.is_cuda
        assert reservoir.weight_input.is_cuda
        assert reservoir.weight_hh.is_cuda

        # Forward pass should work
        feedback = torch.randn(2, 10, 10, device="cuda")
        driving = torch.randn(2, 10, 5, device="cuda")
        output = reservoir(feedback, driving)

        assert output.is_cuda
        assert output.shape == (2, 10, 100)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_state_persistence_on_gpu(self):
        """Test reservoir state persists correctly on GPU."""
        reservoir = ReservoirLayer(
            reservoir_size=50,
            feedback_size=10,
        ).cuda()

        feedback1 = torch.randn(2, 10, 10, device="cuda")

        # First forward pass
        out1 = reservoir(feedback1)
        assert reservoir.state.is_cuda

        # Second forward pass (state carries over)
        assert reservoir.state.is_cuda

        # Reset and compare
        reservoir.reset_state(batch_size=2)
        assert reservoir.state.is_cuda
        out3 = reservoir(feedback1)

        # out3 should match out1 (same input, fresh state)
        assert torch.allclose(out1, out3, rtol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_transfer(self):
        """Test moving model between CPU and GPU."""
        reservoir = ReservoirLayer(
            reservoir_size=50,
            feedback_size=10,
            topology="erdos_renyi",
        )

        # Start on CPU
        feedback_cpu = torch.randn(2, 10, 10)
        out_cpu = reservoir(feedback_cpu)
        assert not out_cpu.is_cuda

        # Move to GPU
        reservoir = reservoir.cuda()
        feedback_gpu = torch.randn(2, 10, 10, device="cuda")
        out_gpu = reservoir(feedback_gpu)
        assert out_gpu.is_cuda

        # Move back to CPU
        reservoir = reservoir.cpu()
        out_cpu2 = reservoir(feedback_cpu)
        assert not out_cpu2.is_cuda


class TestTorchCompile:
    """Tests for torch.compile compatibility."""

    @pytest.mark.skipif(not COMPILE_SUPPORTED, reason="torch.compile not supported")
    def test_reservoir_compile(self):
        """Test ReservoirLayer can be compiled."""
        reservoir = ReservoirLayer(
            reservoir_size=100,
            feedback_size=10,
        )

        # Compile the model
        compiled_reservoir = torch.compile(reservoir)

        # Run forward pass
        feedback = torch.randn(4, 20, 10)
        output = compiled_reservoir(feedback)

        assert output.shape == (4, 20, 100)
        assert not torch.isnan(output).any()

    @pytest.mark.skipif(not COMPILE_SUPPORTED, reason="torch.compile not supported")
    def test_reservoir_compile_with_topology(self):
        """Test compiled ReservoirLayer with graph topology."""
        reservoir = ReservoirLayer(
            reservoir_size=100,
            feedback_size=10,
            topology="erdos_renyi",
            spectral_radius=0.9,
        )

        compiled_reservoir = torch.compile(reservoir)

        feedback = torch.randn(2, 15, 10)
        output = compiled_reservoir(feedback)

        assert output.shape == (2, 15, 100)

    @pytest.mark.skipif(not COMPILE_SUPPORTED, reason="torch.compile not supported")
    def test_reservoir_compile_with_driving_inputs(self):
        """Test compiled ReservoirLayer with driving inputs."""
        reservoir = ReservoirLayer(
            reservoir_size=100,
            feedback_size=10,
            input_size=5,
        )

        compiled_reservoir = torch.compile(reservoir)

        feedback = torch.randn(3, 12, 10)
        driving = torch.randn(3, 12, 5)
        output = compiled_reservoir(feedback, driving)

        assert output.shape == (3, 12, 100)

    @pytest.mark.skipif(not COMPILE_SUPPORTED, reason="torch.compile not supported")
    def test_readout_compile(self):
        """Test ReadoutLayer can be compiled."""
        readout = ReadoutLayer(in_features=100, out_features=10, name="output")

        compiled_readout = torch.compile(readout)

        reservoir_output = torch.randn(4, 20, 100)
        output = compiled_readout(reservoir_output)

        assert output.shape == (4, 20, 10)

    @pytest.mark.skipif(
        not COMPILE_SUPPORTED or not torch.cuda.is_available(),
        reason="torch.compile and CUDA required",
    )
    def test_reservoir_compile_on_gpu(self):
        """Test compiled ReservoirLayer on GPU."""
        reservoir = ReservoirLayer(
            reservoir_size=100,
            feedback_size=10,
            topology="watts_strogatz",
        ).cuda()

        compiled_reservoir = torch.compile(reservoir)

        feedback = torch.randn(4, 20, 10, device="cuda")
        output = compiled_reservoir(feedback)

        assert output.is_cuda
        assert output.shape == (4, 20, 100)


class TestMixedPrecision:
    """Tests for mixed precision (fp16/bf16) support."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_reservoir_fp16(self):
        """Test ReservoirLayer with FP16."""
        reservoir = (
            ReservoirLayer(
                reservoir_size=100,
                feedback_size=10,
            )
            .cuda()
            .half()
        )

        feedback = torch.randn(2, 10, 10, device="cuda", dtype=torch.float16)
        output = reservoir(feedback)

        assert output.dtype == torch.float16
        assert output.shape == (2, 10, 100)

    @pytest.mark.skipif(
        not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
        reason="BF16 not available",
    )
    def test_reservoir_bf16(self):
        """Test ReservoirLayer with BF16."""
        reservoir = (
            ReservoirLayer(
                reservoir_size=100,
                feedback_size=10,
            )
            .cuda()
            .to(torch.bfloat16)
        )

        feedback = torch.randn(2, 10, 10, device="cuda", dtype=torch.bfloat16)
        output = reservoir(feedback)

        assert output.dtype == torch.bfloat16
        assert output.shape == (2, 10, 100)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mixed_precision_autocast(self):
        """Test ReservoirLayer with autocast."""
        reservoir = ReservoirLayer(
            reservoir_size=100,
            feedback_size=10,
        ).cuda()

        feedback = torch.randn(2, 10, 10, device="cuda")

        with torch.amp.autocast("cuda"):
            output = reservoir(feedback)

        # Note: autocast may not always reduce precision for all operations
        # Just verify it runs without error
        assert output.shape == (2, 10, 100)


class TestBatchSizes:
    """Tests for different batch sizes and edge cases."""

    def test_batch_size_one(self):
        """Test with batch size of 1."""
        reservoir = ReservoirLayer(reservoir_size=50, feedback_size=10)
        feedback = torch.randn(1, 20, 10)
        output = reservoir(feedback)
        assert output.shape == (1, 20, 50)

    def test_large_batch_size(self):
        """Test with large batch size."""
        reservoir = ReservoirLayer(reservoir_size=100, feedback_size=10)
        feedback = torch.randn(128, 20, 10)
        output = reservoir(feedback)
        assert output.shape == (128, 20, 100)

    def test_varying_batch_sizes(self):
        """Test that state resets correctly with varying batch sizes."""
        reservoir = ReservoirLayer(reservoir_size=50, feedback_size=10)

        # Process batch of 4
        feedback1 = torch.randn(4, 10, 10)
        out1 = reservoir(feedback1)
        assert out1.shape == (4, 10, 50)

        # Process batch of 2 (should auto-reset state)
        feedback2 = torch.randn(2, 10, 10)
        out2 = reservoir(feedback2)
        assert out2.shape == (2, 10, 50)

        # State should be for batch of 2 now
        assert reservoir.state.shape[0] == 2


class TestSequenceLengths:
    """Tests for different sequence lengths."""

    def test_short_sequence(self):
        """Test with very short sequences."""
        reservoir = ReservoirLayer(reservoir_size=100, feedback_size=10)
        feedback = torch.randn(4, 1, 10)  # Single timestep
        output = reservoir(feedback)
        assert output.shape == (4, 1, 100)

    def test_long_sequence(self):
        """Test with long sequences."""
        reservoir = ReservoirLayer(reservoir_size=100, feedback_size=10)
        feedback = torch.randn(2, 500, 10)
        output = reservoir(feedback)
        assert output.shape == (2, 500, 100)

    def test_varying_sequence_lengths(self):
        """Test multiple forward passes with different sequence lengths."""
        reservoir = ReservoirLayer(reservoir_size=50, feedback_size=10)

        # Different sequence lengths
        for seq_len in [5, 10, 20, 50, 100]:
            reservoir.reset_state()
            feedback = torch.randn(2, seq_len, 10)
            output = reservoir(feedback)
            assert output.shape == (2, seq_len, 50)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
