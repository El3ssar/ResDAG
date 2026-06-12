"""torch.compile and mixed-precision (fp16/bf16/autocast) compatibility.

ESN components must survive ``torch.compile`` wrapping and reduced-precision
execution without shape or NaN regressions.  Compile tests run on every
available device; fp16/bf16 are CUDA-only (CPU support is incomplete).
"""

import sys

import pytest
import torch

from resdag.layers import ESNLayer, ReadoutLayer

COMPILE_SUPPORTED = torch.__version__ >= "2.0.0" and sys.version_info < (3, 15)

requires_compile = pytest.mark.skipif(not COMPILE_SUPPORTED, reason="torch.compile not supported")
cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@requires_compile
class TestTorchCompile:
    """torch.compile compatibility on every available device."""

    def test_reservoir_compile(self, device: torch.device) -> None:
        """A compiled ESNLayer runs and produces sane output."""
        reservoir = ESNLayer(
            reservoir_size=100,
            feedback_size=10,
        ).to(device)

        compiled_reservoir = torch.compile(reservoir)

        feedback = torch.randn(4, 20, 10, device=device)
        output = compiled_reservoir(feedback)

        assert output.device.type == device.type
        assert output.shape == (4, 20, 100)
        assert not torch.isnan(output).any()

    @pytest.mark.parametrize("topology", ["erdos_renyi", "watts_strogatz"])
    def test_reservoir_compile_with_topology(self, device: torch.device, topology: str) -> None:
        """A compiled ESNLayer with graph topology runs on every device."""
        reservoir = ESNLayer(
            reservoir_size=100,
            feedback_size=10,
            topology=topology,
            spectral_radius=0.9,
        ).to(device)

        compiled_reservoir = torch.compile(reservoir)

        feedback = torch.randn(2, 15, 10, device=device)
        output = compiled_reservoir(feedback)

        assert output.device.type == device.type
        assert output.shape == (2, 15, 100)

    def test_reservoir_compile_with_driving_inputs(self, device: torch.device) -> None:
        """A compiled ESNLayer accepts driving inputs."""
        reservoir = ESNLayer(
            reservoir_size=100,
            feedback_size=10,
            input_size=5,
        ).to(device)

        compiled_reservoir = torch.compile(reservoir)

        feedback = torch.randn(3, 12, 10, device=device)
        driving = torch.randn(3, 12, 5, device=device)
        output = compiled_reservoir(feedback, driving)

        assert output.shape == (3, 12, 100)

    def test_readout_compile(self, device: torch.device) -> None:
        """A compiled ReadoutLayer runs and preserves shapes."""
        readout = ReadoutLayer(in_features=100, out_features=10, name="output").to(device)

        compiled_readout = torch.compile(readout)

        reservoir_output = torch.randn(4, 20, 100, device=device)
        output = compiled_readout(reservoir_output)

        assert output.shape == (4, 20, 10)


class TestMixedPrecision:
    """Reduced-precision execution (fp16/bf16/autocast)."""

    @pytest.mark.gpu
    @cuda_required
    def test_reservoir_fp16(self) -> None:
        """ESNLayer runs in half precision on CUDA."""
        reservoir = (
            ESNLayer(
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

    @pytest.mark.gpu
    @pytest.mark.skipif(
        not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
        reason="BF16 not available",
    )
    def test_reservoir_bf16(self) -> None:
        """ESNLayer runs in bfloat16 on CUDA."""
        reservoir = (
            ESNLayer(
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

    def test_mixed_precision_autocast(self, device: torch.device) -> None:
        """ESNLayer runs under torch.amp.autocast on every device."""
        reservoir = ESNLayer(
            reservoir_size=100,
            feedback_size=10,
        ).to(device)

        feedback = torch.randn(2, 10, 10, device=device)

        with torch.amp.autocast(device.type):
            output = reservoir(feedback)

        # Note: autocast may not always reduce precision for all operations.
        # Just verify it runs without error.
        assert output.shape == (2, 10, 100)
