"""Premade model factory contracts (resdag.models).

Pins down the architecture of each factory — classic_esn (with input
concatenation), ott_esn (selective-exponentiation augmentation),
power_augmented (uniform Power augmentation, sign-preserving default),
headless_esn (no readout), linear_esn (forced identity activation) —
their forward shapes, parameter plumbing, and GPU execution.
"""

import pytest
import torch

from resdag.layers import Power
from resdag.layers.cells import ESNCell
from resdag.layers.readouts import CGReadoutLayer
from resdag.models import classic_esn, headless_esn, linear_esn, ott_esn, power_augmented
from resdag.training import ESNTrainer


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

    def test_activation_kwarg_forbidden(self) -> None:
        """Passing activation= raises a clear ValueError, not a cryptic TypeError."""
        # linear_esn forces identity activation by design; an explicit override
        # must fail loudly rather than collide with the hardcoded value and raise
        # "got multiple values for keyword argument 'activation'".
        with pytest.raises(ValueError, match="forces activation='identity'"):
            linear_esn(
                reservoir_size=50,
                feedback_size=2,
                spectral_radius=0.8,
                activation="relu",
            )


class TestPowerAugmented:
    """Tests for the power-augmented ESN architecture.

    Covers the contracts fixed in issue #234: the factory augments every
    reservoir unit with a configurable ``Power`` exponent (not Ott's even-index
    squaring), and the default exponent is sign-preserving rather than
    sign-destroying.
    """

    def _power_layer(self, model) -> Power:  # type: ignore[no-untyped-def]
        """Return the single Power augmentation node wired by the factory."""
        powers = [m for m in model.modules() if isinstance(m, Power)]
        assert len(powers) == 1, "power_augmented should wire exactly one Power node"
        return powers[0]

    def test_basic_instantiation(self) -> None:
        """The factory builds a model with reservoir, Power, and readout."""
        model = power_augmented(reservoir_size=50, feedback_size=2, output_size=3)

        assert model is not None
        layer_names = [name for name, _ in model.named_modules()]
        assert any("ESNLayer" in name for name in layer_names)
        assert any("CGReadoutLayer" in name for name in layer_names)
        # The augmentation node is a Power transform, not Ott's selective one.
        assert any(isinstance(m, Power) for m in model.modules())

    def test_default_exponent_is_sign_preserving(self) -> None:
        """The default exponent is the odd, sign-preserving 3.0.

        Acceptance criterion (#234): the default no longer silently destroys
        the sign of the reservoir states. The wired Power node carries 3.0 and,
        applied to a signed range, keeps negatives negative.
        """
        model = power_augmented(reservoir_size=50, feedback_size=2, output_size=3)
        power = self._power_layer(model)

        assert power.exponent == 3.0
        signed = power(torch.tensor([[-3.0, 3.0]]))
        # Odd exponent preserves sign: a negative base stays negative.
        assert signed[0, 0] < 0 < signed[0, 1]
        assert torch.allclose(signed, torch.tensor([[-27.0, 27.0]]))

    def test_exponent_two_destroys_sign(self) -> None:
        """Opting into exponent=2.0 is allowed and squares (drops) the sign."""
        model = power_augmented(reservoir_size=50, feedback_size=2, output_size=3, exponent=2.0)
        power = self._power_layer(model)

        assert power.exponent == 2.0
        squared = power(torch.tensor([[-3.0, 3.0]]))
        # Even exponent maps both signs to the same non-negative value.
        assert torch.allclose(squared, torch.tensor([[9.0, 9.0]]))

    @pytest.mark.parametrize("exponent", [2.0, 3.0, 1.5])
    def test_exponent_forwarded_to_power_node(self, exponent: float) -> None:
        """The requested exponent reaches the wired Power node verbatim."""
        model = power_augmented(
            reservoir_size=50, feedback_size=2, output_size=3, exponent=exponent
        )

        assert self._power_layer(model).exponent == exponent

    @pytest.mark.parametrize("exponent", [2.0, 3.0, 1.5])
    def test_forward_pass(self, exponent: float) -> None:
        """Forward pass yields the readout shape for integer and fractional exponents.

        Tanh states are non-negative often enough that a fractional exponent
        on a freshly built reservoir stays finite here; the NaN-on-negative
        contract of fractional ``torch.pow`` is pinned in the Power transform
        tests. This guards the factory wiring for exponents 2.0, 3.0 and 1.5.
        """
        model = power_augmented(
            reservoir_size=50, feedback_size=2, output_size=3, exponent=exponent
        )

        x = torch.randn(4, 20, 2)
        output = model(x)

        assert output.shape == (4, 20, 3)

    def test_uses_power_not_selective_exponentiation(self) -> None:
        """Augmentation is uniform Power, not Ott's even-index SelectiveExponentiation."""
        model = power_augmented(reservoir_size=50, feedback_size=2, output_size=3)

        type_names = {type(m).__name__ for m in model.modules()}
        assert "Power" in type_names
        assert "SelectiveExponentiation" not in type_names


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


DRIVEN_FACTORIES = [
    ("classic_esn", classic_esn),
    ("ott_esn", ott_esn),
    ("power_augmented", power_augmented),
]


class TestDrivenFactories:
    """Driving/exogenous-input support across the readout factories.

    Each of ``classic_esn``, ``ott_esn`` and ``power_augmented`` accepts an
    ``input_size`` that wires a second (driving) input into the reservoir while
    keeping the readout ``in_features`` at ``feedback_size + reservoir_size``.
    """

    @pytest.mark.parametrize("name, factory", DRIVEN_FACTORIES, ids=lambda v: v)
    def test_builds_two_input_model(self, name: str, factory) -> None:  # type: ignore[no-untyped-def]
        """With ``input_size`` set the model takes (feedback, driver)."""
        model = factory(reservoir_size=50, feedback_size=2, output_size=3, input_size=4)

        feedback = torch.randn(4, 20, 2)
        driver = torch.randn(4, 20, 4)
        output = model(feedback, driver)

        assert output.shape == (4, 20, 3)

    @pytest.mark.parametrize("name, factory", DRIVEN_FACTORIES, ids=lambda v: v)
    def test_readout_in_features_excludes_driver(self, name: str, factory) -> None:  # type: ignore[no-untyped-def]
        """Driver stays out of the concat: in_features == feedback + reservoir."""
        model = factory(reservoir_size=50, feedback_size=2, output_size=3, input_size=4)

        readouts = [m for m in model.modules() if isinstance(m, CGReadoutLayer)]
        assert readouts, f"{name} should contain a CGReadoutLayer"
        # feedback_size (2) + reservoir_size (50); the driver (4) is excluded.
        assert all(r.in_features == 52 for r in readouts)

    @pytest.mark.parametrize("name, factory", DRIVEN_FACTORIES, ids=lambda v: v)
    def test_input_size_forwarded_to_reservoir(self, name: str, factory) -> None:  # type: ignore[no-untyped-def]
        """The reservoir cell is built with the requested driving-input size."""
        model = factory(reservoir_size=50, feedback_size=2, output_size=3, input_size=4)

        cells = [m for m in model.modules() if isinstance(m, ESNCell)]
        assert cells, f"{name} should contain an ESNCell"
        assert all(c.input_size == 4 for c in cells)

    @pytest.mark.parametrize("name, factory", DRIVEN_FACTORIES, ids=lambda v: v)
    def test_fit_and_forecast_with_drivers(self, name: str, factory) -> None:  # type: ignore[no-untyped-def]
        """A driven model fits with driver tuples and forecasts with drivers."""
        torch.manual_seed(0)
        model = factory(reservoir_size=60, feedback_size=2, output_size=2, input_size=1)

        warmup_fb = torch.randn(1, 30, 2)
        warmup_dr = torch.randn(1, 30, 1)
        train_fb = torch.randn(1, 80, 2)
        train_dr = torch.randn(1, 80, 1)
        target = torch.randn(1, 80, 2)

        ESNTrainer(model).fit(
            warmup_inputs=(warmup_fb, warmup_dr),
            train_inputs=(train_fb, train_dr),
            targets={"output": target},
        )

        future_dr = torch.randn(1, 40, 1)
        preds = model.forecast(
            (warmup_fb, warmup_dr),
            forecast_inputs=(future_dr,),
            horizon=40,
        )

        assert preds.shape == (1, 40, 2)
        assert torch.isfinite(preds).all()

    @pytest.mark.parametrize("name, factory", DRIVEN_FACTORIES, ids=lambda v: v)
    def test_non_driven_path_unchanged(self, name: str, factory) -> None:  # type: ignore[no-untyped-def]
        """Default ``input_size=None`` keeps the single-input behavior."""
        model = factory(reservoir_size=50, feedback_size=2, output_size=3)

        x = torch.randn(4, 20, 2)
        output = model(x)
        assert output.shape == (4, 20, 3)

        cells = [m for m in model.modules() if isinstance(m, ESNCell)]
        assert cells
        # No driving input means a zero-size driving-weight placeholder.
        assert all((c.input_size or 0) == 0 for c in cells)

        readouts = [m for m in model.modules() if isinstance(m, CGReadoutLayer)]
        assert all(r.in_features == 52 for r in readouts)


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
