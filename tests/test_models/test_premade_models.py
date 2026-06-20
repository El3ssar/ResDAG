"""Premade model factory contracts (resdag.models).

Pins down the architecture of each factory — classic_esn (with input
concatenation), ott_esn (selective-exponentiation augmentation),
headless_esn (no readout), linear_esn (forced identity activation) —
their forward shapes, parameter plumbing, and GPU execution.
"""

import pytest
import torch

from resdag.layers.cells import ESNCell
from resdag.models import classic_esn, headless_esn, linear_esn, ott_esn


class TestClassicESN:
    """Tests for classic ESN architecture."""

    def test_basic_instantiation(self) -> None:
        """Test creating classic ESN with minimal parameters."""
        model = classic_esn(reservoir_size=50, feedback_size=2, output_size=3)

        assert model is not None
        # Check that model has expected layers
        layer_names = [name for name, _ in model.named_modules()]
        assert any("ESNLayer" in name for name in layer_names)
        assert any("CGReadoutLayer" in name for name in layer_names)

    def test_forward_pass(self) -> None:
        """Test forward pass through classic ESN."""
        model = classic_esn(reservoir_size=50, feedback_size=2, output_size=3)

        x = torch.randn(4, 20, 2)  # (batch, time, features)
        output = model(x)  # Direct call, no dict!

        assert output.shape == (4, 20, 3)

    def test_with_reservoir_params(self) -> None:
        """Test classic ESN with custom reservoir parameters."""
        model = classic_esn(
            reservoir_size=50,
            feedback_size=2,
            output_size=3,
            topology="erdos_renyi",
            spectral_radius=0.9,
            leak_rate=0.3,
        )

        x = torch.randn(4, 20, 2)
        output = model(x)

        assert output.shape == (4, 20, 3)

    def test_with_tuple_topology(self) -> None:
        """Test classic ESN with tuple topology spec."""
        model = classic_esn(
            reservoir_size=50,
            feedback_size=2,
            output_size=3,
            topology=("watts_strogatz", {"k": 4, "p": 0.1}),
            spectral_radius=0.9,
        )

        x = torch.randn(4, 20, 2)
        output = model(x)

        assert output.shape == (4, 20, 3)

    def test_with_readout_params(self) -> None:
        """Test classic ESN with custom readout parameters."""
        model = classic_esn(
            reservoir_size=50,
            feedback_size=2,
            output_size=3,
            readout_alpha=1e-4,
            readout_bias=False,
            readout_name="predictions",
        )

        x = torch.randn(4, 20, 2)
        output = model(x)

        assert output.shape == (4, 20, 3)

    def test_concatenation_dimension(self) -> None:
        """Test that input is concatenated with reservoir output."""
        model = classic_esn(reservoir_size=50, feedback_size=2, output_size=3)

        # Check for Concatenate layer
        has_concat = any("Concatenate" in str(type(m)) for m in model.modules())
        assert has_concat, "Classic ESN should have Concatenate layer"

    def test_noise_forwarded_to_reservoir(self) -> None:
        """The noise kwarg is forwarded to the reservoir cell."""
        model = classic_esn(reservoir_size=50, feedback_size=2, output_size=3, noise=0.05)

        cells = [m for m in model.modules() if isinstance(m, ESNCell)]
        assert cells, "classic_esn should contain an ESNCell"
        assert all(c.noise == 0.05 for c in cells)

    def test_noise_defaults_to_zero(self) -> None:
        """Without the noise kwarg the reservoir cell keeps noise=0.0."""
        model = classic_esn(reservoir_size=50, feedback_size=2, output_size=3)

        cells = [m for m in model.modules() if isinstance(m, ESNCell)]
        assert cells
        assert all(c.noise == 0.0 for c in cells)


class TestOttESN:
    """Tests for Ott's ESN architecture."""

    def test_basic_instantiation(self) -> None:
        """Test creating Ott ESN with minimal parameters."""
        model = ott_esn(reservoir_size=50, feedback_size=2, output_size=3)

        assert model is not None
        layer_names = [name for name, _ in model.named_modules()]
        assert any("ESNLayer" in name for name in layer_names)
        assert any("SelectiveExponentiation" in name for name in layer_names)

    def test_forward_pass(self) -> None:
        """Test forward pass through Ott ESN."""
        model = ott_esn(reservoir_size=50, feedback_size=2, output_size=3)

        x = torch.randn(4, 20, 2)
        output = model(x)

        assert output.shape == (4, 20, 3)

    def test_with_custom_params(self) -> None:
        """Test Ott ESN with custom parameters."""
        model = ott_esn(
            reservoir_size=50,
            feedback_size=2,
            output_size=3,
            spectral_radius=0.95,
            leak_rate=0.3,
            feedback_initializer="pseudo_diagonal",
        )

        x = torch.randn(4, 20, 2)
        output = model(x)

        assert output.shape == (4, 20, 3)

    def test_state_augmentation(self) -> None:
        """Test that Ott ESN includes state augmentation layer."""
        model = ott_esn(reservoir_size=50, feedback_size=2, output_size=3)

        # Check for SelectiveExponentiation layer
        has_augmentation = any("SelectiveExponentiation" in str(type(m)) for m in model.modules())
        assert has_augmentation, "Ott ESN should have SelectiveExponentiation layer"


class TestHeadlessESN:
    """Tests for headless ESN architecture."""

    def test_basic_instantiation(self) -> None:
        """Test creating headless ESN."""
        model = headless_esn(reservoir_size=50, feedback_size=2)

        assert model is not None
        layer_names = [name for name, _ in model.named_modules()]
        assert any("ESNLayer" in name for name in layer_names)

    def test_forward_pass(self) -> None:
        """Test forward pass returns reservoir states."""
        model = headless_esn(reservoir_size=50, feedback_size=2)

        x = torch.randn(4, 20, 2)
        output = model(x)

        # Output should be reservoir states
        assert output.shape == (4, 20, 50)

    def test_no_readout_layer(self) -> None:
        """Test that headless ESN has no readout layer."""
        model = headless_esn(reservoir_size=50, feedback_size=2)

        # Check that no readout layer exists
        from resdag.layers.readouts import ReadoutLayer

        has_readout = any(isinstance(m, ReadoutLayer) for m in model.modules())
        assert not has_readout, "Headless ESN should not have readout layer"

    def test_with_custom_params(self) -> None:
        """Test headless ESN with custom parameters."""
        model = headless_esn(
            reservoir_size=50,
            feedback_size=2,
            spectral_radius=0.8,
            leak_rate=0.2,
            topology=("ring_chord", {"L": 1}),
        )

        x = torch.randn(4, 20, 2)
        output = model(x)

        assert output.shape == (4, 20, 50)


class TestLinearESN:
    """Tests for linear ESN architecture."""

    def test_basic_instantiation(self) -> None:
        """Test creating linear ESN."""
        model = linear_esn(reservoir_size=50, feedback_size=2)

        assert model is not None
        layer_names = [name for name, _ in model.named_modules()]
        assert any("ESNLayer" in name for name in layer_names)

    def test_forward_pass(self) -> None:
        """Test forward pass returns reservoir states."""
        model = linear_esn(reservoir_size=50, feedback_size=2)

        x = torch.randn(4, 20, 2)
        output = model(x)

        assert output.shape == (4, 20, 50)

    def test_linear_activation_forced(self) -> None:
        """Test that linear ESN forces identity activation."""
        model = linear_esn(reservoir_size=50, feedback_size=2)

        # Find reservoir layer and check activation
        from resdag.layers import ESNLayer

        reservoir = None
        for module in model.modules():
            if isinstance(module, ESNLayer):
                reservoir = module
                break

        assert reservoir is not None
        # Test that activation behaves like identity
        test_input = torch.randn(1, 10)
        activated = reservoir.activation_fn(test_input)
        assert torch.allclose(activated, test_input), "Linear ESN should use identity activation"

    def test_activation_forced_even_with_kwargs(self) -> None:
        """Test that activation is forced even with extra kwargs."""
        # Even if user tries to pass activation in kwargs, linear_esn forces identity
        model = linear_esn(
            reservoir_size=50,
            feedback_size=2,
            spectral_radius=0.8,
        )

        # Verify activation is still identity
        from resdag.layers import ESNLayer

        reservoir = None
        for module in model.modules():
            if isinstance(module, ESNLayer):
                reservoir = module
                break

        test_input = torch.randn(1, 10)
        activated = reservoir.activation_fn(test_input)
        assert torch.allclose(activated, test_input)


class TestModelComparison:
    """Test differences between model architectures."""

    def test_classic_vs_ott_structure(self) -> None:
        """Test structural differences between classic and Ott ESN."""
        classic = classic_esn(50, 2, 3)
        ott = ott_esn(50, 2, 3)

        # Both should have reservoir and readout
        assert classic is not None and ott is not None

        # Ott should have SelectiveExponentiation
        ott_has_aug = any("SelectiveExponentiation" in str(type(m)) for m in ott.modules())
        assert ott_has_aug

    def test_headless_vs_linear_dynamics(self) -> None:
        """Test that headless and linear ESN produce different dynamics."""
        headless = headless_esn(50, 2)
        linear = linear_esn(50, 2)

        x = torch.randn(4, 20, 2)

        # Both should output reservoir states
        out_headless = headless(x)
        out_linear = linear(x)

        assert out_headless.shape == out_linear.shape == (4, 20, 50)

        # Outputs should differ due to different activations
        assert not torch.allclose(out_headless, out_linear, atol=1e-5)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestModelsGPU:
    """Premade models on GPU."""

    def test_classic_esn_on_gpu(self) -> None:
        """Test classic ESN on GPU."""
        model = classic_esn(50, 2, 3).cuda()
        x = torch.randn(4, 20, 2).cuda()

        output = model(x)

        assert output.is_cuda
        assert output.shape == (4, 20, 3)

    def test_ott_esn_on_gpu(self) -> None:
        """Test Ott ESN on GPU."""
        model = ott_esn(50, 2, 3).cuda()
        x = torch.randn(4, 20, 2).cuda()

        output = model(x)

        assert output.is_cuda
        assert output.shape == (4, 20, 3)

    def test_headless_esn_on_gpu(self) -> None:
        """Test headless ESN on GPU."""
        model = headless_esn(50, 2).cuda()
        x = torch.randn(4, 20, 2).cuda()

        output = model(x)

        assert output.is_cuda
        assert output.shape == (4, 20, 50)
