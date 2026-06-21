"""Behavioural contract of ESNLayer.

Pins down:

- construction and configuration (weight shapes, custom parameters, validation),
- forward-pass shape and error contracts (feedback + driving inputs),
- stateful behaviour (persistence, reset, get/set, detach-between-calls),
- weight construction semantics (spectral radius, bias, leak rate, activations),
- topology-spec resolution through the layer (string / tuple / object specs),
- batch-size and sequence-length envelopes,
- device placement (CPU/CUDA via the ``device`` fixture).
"""

import pytest
import torch

from resdag.init.graphs import erdos_renyi_graph
from resdag.init.topology import GraphTopology, get_topology
from resdag.layers import ESNLayer

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _spectral_radius(weight: torch.Tensor) -> float:
    """Largest absolute eigenvalue of a square weight matrix."""
    eigenvalues = torch.linalg.eigvals(weight)
    return torch.max(torch.abs(eigenvalues)).item()


class TestESNLayerInstantiation:
    """ESNLayer instantiation and configuration."""

    def test_instantiation_feedback_only(self) -> None:
        """Creating reservoir with feedback only."""
        reservoir = ESNLayer(reservoir_size=100, feedback_size=10)

        assert reservoir.reservoir_size == 100
        assert reservoir.feedback_size == 10
        assert reservoir.input_size is None
        assert reservoir.spectral_radius is None  # default is None
        assert reservoir._initialized is True
        assert reservoir.weight_feedback.shape == (100, 10)
        assert reservoir.weight_hh.shape == (100, 100)
        assert reservoir.weight_input is None

    def test_instantiation_with_driving_inputs(self) -> None:
        """Creating reservoir with feedback and driving inputs."""
        reservoir = ESNLayer(reservoir_size=100, feedback_size=10, input_size=5)

        assert reservoir.feedback_size == 10
        assert reservoir.input_size == 5
        assert reservoir.weight_feedback.shape == (100, 10)
        assert reservoir.weight_input.shape == (100, 5)
        assert reservoir.weight_hh.shape == (100, 100)

    def test_custom_parameters(self) -> None:
        """Custom spectral radius, activation, leak rate, etc."""
        reservoir = ESNLayer(
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
        """Invalid activation name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown activation"):
            ESNLayer(reservoir_size=100, feedback_size=10, activation="invalid")


class TestESNLayerForwardPass:
    """Forward pass shape and error contracts."""

    def test_forward_feedback_only(self) -> None:
        """Forward pass with feedback only."""
        reservoir = ESNLayer(reservoir_size=50, feedback_size=10)
        feedback = torch.randn(2, 20, 10)  # (batch=2, seq=20, feedback=10)

        output = reservoir(feedback)

        assert output.shape == (2, 20, 50)
        assert isinstance(output, torch.Tensor)

    def test_forward_with_driving_inputs(self) -> None:
        """Forward pass with feedback and driving inputs."""
        reservoir = ESNLayer(reservoir_size=100, feedback_size=10, input_size=5)

        feedback = torch.randn(3, 15, 10)  # (batch=3, seq=15, feedback=10)
        driving = torch.randn(3, 15, 5)  # (batch=3, seq=15, input=5)

        output = reservoir(feedback, driving)

        assert output.shape == (3, 15, 100)

    def test_forward_with_matching_driving_input(self) -> None:
        """Forward with feedback and single driving input matching input_size."""
        reservoir = ESNLayer(reservoir_size=100, feedback_size=10, input_size=8)

        feedback = torch.randn(2, 10, 10)
        driving = torch.randn(2, 10, 8)

        output = reservoir(feedback, driving)

        assert output.shape == (2, 10, 100)

    def test_forward_invalid_feedback_dimensions_raises_error(self) -> None:
        """Non-3D feedback raises error."""
        reservoir = ESNLayer(reservoir_size=50, feedback_size=10)

        with pytest.raises(ValueError, match="Feedback must be 3D"):
            reservoir(torch.randn(2, 10))  # Only 2D

    def test_forward_feedback_size_mismatch_raises_error(self) -> None:
        """Wrong feedback size raises error."""
        reservoir = ESNLayer(reservoir_size=50, feedback_size=10)
        feedback = torch.randn(2, 20, 8)  # Wrong size!

        with pytest.raises(ValueError, match="Feedback size mismatch"):
            reservoir(feedback)

    def test_forward_inconsistent_batch_sizes_raises_error(self) -> None:
        """Inconsistent batch sizes raise error."""
        reservoir = ESNLayer(reservoir_size=50, feedback_size=10, input_size=5)

        feedback = torch.randn(2, 10, 10)
        driving = torch.randn(3, 10, 5)  # Different batch size!

        with pytest.raises(ValueError, match="match feedback dimensions"):
            reservoir(feedback, driving)

    def test_forward_driving_without_input_size_raises_error(self) -> None:
        """Providing driving inputs without input_size raises error."""
        reservoir = ESNLayer(reservoir_size=50, feedback_size=10)  # No input_size

        feedback = torch.randn(2, 10, 10)
        driving = torch.randn(2, 10, 5)

        with pytest.raises(ValueError, match="without input_size"):
            reservoir(feedback, driving)

    def test_forward_driving_size_mismatch_raises_error(self) -> None:
        """Wrong driving input size raises error."""
        reservoir = ESNLayer(reservoir_size=50, feedback_size=10, input_size=5)

        feedback = torch.randn(2, 10, 10)
        driving = torch.randn(2, 10, 8)  # Wrong size!

        with pytest.raises(ValueError, match="Driving input size mismatch"):
            reservoir(feedback, driving)


class TestESNLayerStatefulBehavior:
    """Stateful behaviour: persistence, reset, get/set."""

    def test_state_persistence_across_forward_passes(self) -> None:
        """State persists across multiple forward passes."""
        reservoir = ESNLayer(reservoir_size=50, feedback_size=10)

        fb1 = torch.randn(2, 5, 10)
        fb2 = torch.randn(2, 5, 10)

        # First forward pass
        out1 = reservoir(fb1)
        state_after_fb1 = reservoir.get_state()

        # Second forward pass (state should carry over)
        reservoir(fb2)
        state_after_fb2 = reservoir.get_state()

        # States should be different
        assert not torch.allclose(state_after_fb1, state_after_fb2)

        # Reset and run fb1 again - should get same initial evolution
        reservoir.reset_state(batch_size=2)
        out1_again = reservoir(fb1)

        # First output should be similar (small numerical differences OK)
        assert torch.allclose(out1, out1_again, rtol=1e-5, atol=1e-6)

    def test_reset_state_without_batch_size(self) -> None:
        """reset_state() with no arguments sets state to None."""
        reservoir = ESNLayer(reservoir_size=50, feedback_size=10)
        feedback = torch.randn(2, 5, 10)

        reservoir(feedback)
        assert reservoir.state is not None

        reservoir.reset_state()
        assert reservoir.state is None

    def test_reset_state_with_batch_size(self) -> None:
        """reset_state() with batch_size initializes zeros."""
        reservoir = ESNLayer(reservoir_size=50, feedback_size=10)

        reservoir.reset_state(batch_size=3)

        assert reservoir.state is not None
        assert reservoir.state.shape == (3, 50)
        assert torch.allclose(reservoir.state, torch.zeros(3, 50))

    def test_set_state(self) -> None:
        """set_state() sets internal state."""
        reservoir = ESNLayer(reservoir_size=50, feedback_size=10)

        custom_state = torch.randn(2, 50)
        reservoir.set_state(custom_state)

        assert reservoir.state is not None
        assert torch.allclose(reservoir.state, custom_state)

    def test_set_state_invalid_shape_raises_error(self) -> None:
        """set_state() with wrong shape raises error."""
        reservoir = ESNLayer(reservoir_size=50, feedback_size=10)

        wrong_state = torch.randn(2, 40)  # Wrong reservoir size!

        with pytest.raises(ValueError, match="validate_state"):
            reservoir.set_state(wrong_state)

    def test_get_state(self) -> None:
        """get_state() returns copy of state."""
        reservoir = ESNLayer(reservoir_size=50, feedback_size=10)
        feedback = torch.randn(2, 5, 10)

        reservoir(feedback)
        state1 = reservoir.get_state()
        state2 = reservoir.get_state()

        # Should return clones (not same object)
        assert state1 is not state2
        assert torch.allclose(state1, state2)

    def test_get_state_before_initialization_returns_none(self) -> None:
        """get_state() returns None if state not initialized."""
        reservoir = ESNLayer(reservoir_size=50, feedback_size=10)

        state = reservoir.get_state()
        assert state is None


class TestESNLayerSetStateBatchContract:
    """set_state pins the next-forward batch size (#145).

    A state restored via :meth:`set_state` must not be silently discarded by a
    later forward at a mismatched batch size — the discard used to zero-reinit
    the reservoir and produce plausible-looking but wrong forecasts for the
    common train-with-``batch>1`` / forecast-with-``batch=1`` pattern.
    """

    def test_set_state_then_forward_mismatched_batch_raises(self) -> None:
        """set_state(batch=7) then forward(batch=4) raises, not silent zero-init."""
        reservoir = ESNLayer(reservoir_size=50, feedback_size=10)
        reservoir.reset_state(batch_size=7)
        saved_7 = reservoir.get_state()

        reservoir.set_state(saved_7)
        with pytest.raises(RuntimeError, match="set_state"):
            reservoir(torch.randn(4, 5, 10))

    def test_set_state_then_forward_matching_batch_preserved(self) -> None:
        """A restored state is honoured (not zeroed) when batch sizes match."""
        reservoir = ESNLayer(reservoir_size=50, feedback_size=10)
        reservoir(torch.randn(3, 8, 10))  # warm up so the state is non-trivial
        saved_3 = reservoir.get_state()
        same_input = torch.randn(3, 1, 10)

        # A one-step forward from the restored state must continue from it, not
        # from re-zeroed state: its output differs from the zero-started run.
        reservoir.set_state(saved_3)
        out_from_saved = reservoir(same_input)
        assert reservoir.state.shape[0] == 3

        reservoir.reset_state(batch_size=3)
        out_from_zero = reservoir(same_input)
        assert not torch.allclose(out_from_saved, out_from_zero)

    def test_lazy_zero_init_still_auto_resizes_without_error(self) -> None:
        """Lazy/None state path keeps auto-resizing on batch change, no error."""
        reservoir = ESNLayer(reservoir_size=50, feedback_size=10)
        assert reservoir.state is None  # lazy: nothing user-set

        reservoir(torch.randn(4, 6, 10))
        # An evolved (not user-set) state must still auto-resize silently.
        reservoir(torch.randn(2, 6, 10))
        assert reservoir.state.shape[0] == 2

    def test_evolved_state_after_set_state_auto_resizes(self) -> None:
        """Once a restored state evolves via forward, the pin is released."""
        reservoir = ESNLayer(reservoir_size=50, feedback_size=10)
        reservoir.reset_state(batch_size=5)
        reservoir.set_state(reservoir.get_state())

        reservoir(torch.randn(5, 4, 10))  # matching batch: state evolves, pin drops
        # Now a batch change must auto-resize (no longer the as-restored tensor).
        reservoir(torch.randn(2, 4, 10))
        assert reservoir.state.shape[0] == 2

    def test_reset_state_clears_user_set_pin(self) -> None:
        """reset_state() opts back into auto-resize after a set_state."""
        reservoir = ESNLayer(reservoir_size=50, feedback_size=10)
        reservoir.reset_state(batch_size=7)
        reservoir.set_state(reservoir.get_state())

        reservoir.reset_state()  # explicit opt back into auto-resize
        reservoir(torch.randn(4, 5, 10))  # must not raise
        assert reservoir.state.shape[0] == 4

    def test_set_state_docstring_states_batch_contract(self) -> None:
        """set_state docstring documents the batch-size contract explicitly."""
        doc = ESNLayer.set_state.__doc__
        assert doc is not None
        assert "batch" in doc.lower()
        assert "next forward" in doc.lower()


class TestESNLayerActivations:
    """Activation functions and their codomain."""

    @pytest.mark.parametrize("activation", ["tanh", "relu", "sigmoid", "identity"])
    def test_different_activations(self, activation: str) -> None:
        """Reservoir works with different activations."""
        reservoir = ESNLayer(reservoir_size=50, feedback_size=10, activation=activation)
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


class TestESNLayerSpectralRadius:
    """Spectral radius scaling of the recurrent weights."""

    @pytest.mark.parametrize("target_sr", [0.5, 0.8, 0.9, 1.2])
    def test_spectral_radius_scaling(self, target_sr: float) -> None:
        """Recurrent weights are scaled to the target spectral radius."""
        reservoir = ESNLayer(reservoir_size=50, feedback_size=10, spectral_radius=target_sr)

        actual_sr = _spectral_radius(reservoir.weight_hh.data)

        # Should be close to target (within numerical precision)
        assert abs(actual_sr - target_sr) < 0.01


class TestESNLayerLeakRate:
    """Leaky integration."""

    def test_leak_rate_affects_output(self) -> None:
        """Leak rate changes output."""
        torch.manual_seed(42)
        reservoir_no_leak = ESNLayer(reservoir_size=50, feedback_size=10, leak_rate=1.0)

        torch.manual_seed(42)
        reservoir_with_leak = ESNLayer(reservoir_size=50, feedback_size=10, leak_rate=0.5)

        feedback = torch.randn(2, 10, 10)

        out_no_leak = reservoir_no_leak(feedback)
        out_with_leak = reservoir_with_leak(feedback)

        # Outputs should be different
        assert not torch.allclose(out_no_leak, out_with_leak)


class TestESNLayerNoise:
    """Train-mode state-noise injection plumbed through the sequence loop."""

    def test_noise_forwarded_to_cell(self) -> None:
        """The noise kwarg reaches the inner ESNCell."""
        reservoir = ESNLayer(reservoir_size=50, feedback_size=10, noise=0.1)

        assert reservoir.noise == 0.1
        assert reservoir.cell.noise == 0.1

    def test_noise_perturbs_sequence_in_train_mode(self) -> None:
        """noise > 0 changes the output sequence in train() mode."""
        noisy = ESNLayer(reservoir_size=50, feedback_size=10, noise=0.1, seed=0)
        clean = ESNLayer(reservoir_size=50, feedback_size=10, noise=0.0, seed=0)
        noisy.train()
        clean.train()
        feedback = torch.randn(2, 15, 10)

        out_noisy = noisy(feedback)
        out_clean = clean(feedback)

        assert not torch.allclose(out_noisy, out_clean)

    def test_noise_noop_in_eval_mode(self) -> None:
        """noise > 0 is a no-op under eval(), matching the noiseless layer."""
        noisy = ESNLayer(reservoir_size=50, feedback_size=10, noise=0.5, seed=0)
        clean = ESNLayer(reservoir_size=50, feedback_size=10, noise=0.0, seed=0)
        noisy.eval()
        clean.eval()
        feedback = torch.randn(2, 15, 10)

        out_noisy = noisy(feedback)
        out_clean = clean(feedback)

        assert torch.equal(out_noisy, out_clean)

    def test_noise_reproducible_under_seed(self) -> None:
        """Two seeded layers produce identical noisy sequences."""
        a = ESNLayer(reservoir_size=50, feedback_size=10, noise=0.2, seed=99)
        b = ESNLayer(reservoir_size=50, feedback_size=10, noise=0.2, seed=99)
        a.train()
        b.train()
        feedback = torch.randn(2, 15, 10)

        assert torch.equal(a(feedback), b(feedback))


class TestESNLayerGradients:
    """Gradient flow through trainable reservoirs."""

    def test_gradients_flow_through_reservoir(self) -> None:
        """Gradients flow through the reservoir."""
        reservoir = ESNLayer(reservoir_size=50, feedback_size=10, trainable=True)
        feedback = torch.randn(2, 5, 10, requires_grad=True)

        output = reservoir(feedback)
        loss = output.sum()
        loss.backward()

        # Input should have gradients
        assert feedback.grad is not None

        # Weights should have gradients
        assert reservoir.weight_feedback.grad is not None
        assert reservoir.weight_hh.grad is not None


class TestESNLayerRepr:
    """String representation (delegates to the inner ESNCell)."""

    def test_repr_feedback_only(self) -> None:
        """__repr__ for feedback-only reservoir."""
        reservoir = ESNLayer(reservoir_size=100, feedback_size=20, spectral_radius=0.95)

        repr_str = repr(reservoir)

        assert "ESNCell" in repr_str
        assert "reservoir_size=100" in repr_str
        assert "feedback_size=20" in repr_str
        assert "spectral_radius=0.95" in repr_str

    def test_repr_with_driving_inputs(self) -> None:
        """__repr__ for reservoir with driving inputs."""
        reservoir = ESNLayer(
            reservoir_size=100, feedback_size=20, input_size=5, spectral_radius=0.95
        )

        repr_str = repr(reservoir)

        assert "feedback_size=20" in repr_str
        assert "input_size=5" in repr_str


class TestESNLayerBias:
    """Reservoir bias initialization and its effect on dynamics."""

    def test_bias_is_random_by_default(self) -> None:
        """Default bias must be nonzero — a zero frozen bias is a no-op."""
        torch.manual_seed(0)
        reservoir = ESNLayer(reservoir_size=100, feedback_size=3)

        assert reservoir.bias_h is not None
        assert not torch.allclose(reservoir.bias_h, torch.zeros(100))
        assert reservoir.bias_h.abs().max() <= 1.0

    def test_bias_scaling_bounds(self) -> None:
        """bias_scaling controls the uniform range of the bias."""
        torch.manual_seed(0)
        reservoir = ESNLayer(reservoir_size=200, feedback_size=3, bias_scaling=0.3)

        assert reservoir.bias_h.abs().max() <= 0.3
        assert not torch.allclose(reservoir.bias_h, torch.zeros(200))

    def test_bias_scaling_zero_gives_zero_bias(self) -> None:
        """bias_scaling=0.0 reproduces the legacy zero-bias behaviour."""
        reservoir = ESNLayer(reservoir_size=50, feedback_size=3, bias_scaling=0.0)

        assert reservoir.bias_h is not None
        assert torch.allclose(reservoir.bias_h, torch.zeros(50))

    def test_bias_false_still_none(self) -> None:
        """bias=False must not create a bias parameter regardless of scaling."""
        reservoir = ESNLayer(reservoir_size=50, feedback_size=3, bias=False, bias_scaling=0.5)

        assert reservoir.bias_h is None

    def test_bias_breaks_odd_symmetry(self) -> None:
        """Without bias, tanh dynamics are odd (f(-x) = -f(x)); the random
        bias must break that symmetry."""
        torch.manual_seed(7)
        x = torch.randn(1, 30, 3)

        sym = ESNLayer(reservoir_size=64, feedback_size=3, bias=False, spectral_radius=0.9)
        sym.reset_state()
        out_pos = sym(x)
        sym.reset_state()
        out_neg = sym(-x)
        assert torch.allclose(out_neg, -out_pos, atol=1e-6)

        torch.manual_seed(7)
        biased = ESNLayer(reservoir_size=64, feedback_size=3, bias=True, spectral_radius=0.9)
        biased.reset_state()
        out_pos = biased(x)
        biased.reset_state()
        out_neg = biased(-x)
        assert not torch.allclose(out_neg, -out_pos, atol=1e-3)

    def test_bias_frozen_when_not_trainable(self) -> None:
        """The random bias is a fixed parameter on the frozen path."""
        reservoir = ESNLayer(reservoir_size=50, feedback_size=3, trainable=False)

        assert reservoir.bias_h.requires_grad is False


class TestStateDetachBetweenCalls:
    """Truncated-BPTT contract: stored state never carries the call graph."""

    def test_state_detached_after_forward(self) -> None:
        """Stored state has no grad_fn after a grad-enabled forward."""
        reservoir = ESNLayer(reservoir_size=32, feedback_size=3, trainable=True)
        x = torch.randn(2, 10, 3)

        out = reservoir(x)

        assert out.requires_grad
        assert reservoir.state is not None
        assert reservoir.state.grad_fn is None

    def test_opt_out_keeps_graph(self) -> None:
        """detach_state_between_calls=False preserves cross-call graphs."""
        reservoir = ESNLayer(reservoir_size=32, feedback_size=3, trainable=True)
        reservoir.detach_state_between_calls = False
        x = torch.randn(2, 10, 3)

        reservoir(x)

        assert reservoir.state.grad_fn is not None

    def test_consecutive_backward_without_reset(self) -> None:
        """Two forward+backward cycles without reset must not raise."""
        reservoir = ESNLayer(reservoir_size=32, feedback_size=3, trainable=True)
        x = torch.randn(2, 10, 3)

        for _ in range(2):
            out = reservoir(x)
            out.sum().backward()


class TestESNLayerStateBuffer:
    """The reservoir state is a non-persistent buffer (issue #132).

    The 2-D ESN state must move with the module under ``.to()`` / ``.double()``
    (so a warmed-up trajectory survives a device/dtype change instead of being
    silently zero-reinitialised), yet must NOT leak into ``state_dict()`` — the
    ``save`` / ``include_states`` split depends on it staying out.
    """

    def test_state_is_registered_buffer(self) -> None:
        """``state`` lives in ``named_buffers()`` but not ``state_dict()``."""
        reservoir = ESNLayer(reservoir_size=40, feedback_size=3)
        reservoir(torch.randn(2, 8, 3))  # warm up so the buffer is a real tensor

        assert "state" in dict(reservoir.named_buffers())
        assert "state" not in reservoir.state_dict()

    def test_state_not_in_state_dict_before_warmup(self) -> None:
        """A ``None`` state never appears in ``state_dict()``."""
        reservoir = ESNLayer(reservoir_size=40, feedback_size=3)
        assert reservoir.state is None
        assert "state" not in reservoir.state_dict()

    def test_state_does_not_require_grad(self) -> None:
        """A buffer state must not carry gradients by default."""
        reservoir = ESNLayer(reservoir_size=40, feedback_size=3)
        reservoir(torch.randn(2, 8, 3))
        assert reservoir.state.requires_grad is False

    def test_double_preserves_warmed_state(self) -> None:
        """``.double()`` moves and preserves the warmed state (no zero-reinit)."""
        reservoir = ESNLayer(reservoir_size=40, feedback_size=3)
        reservoir(torch.randn(2, 8, 3))  # warm up on CPU/float32
        warmed = reservoir.get_state().clone()

        assert reservoir.state.dtype == torch.float32
        reservoir = reservoir.double()

        # State moved to float64 and values are preserved (cast, not re-zeroed).
        assert reservoir.state.dtype == torch.float64
        assert (reservoir.state != 0).any()
        assert torch.allclose(reservoir.state, warmed.double())

    def test_double_then_forward_keeps_continuity(self) -> None:
        """Warm → ``.double()`` → forward must not silently zero the state."""
        reservoir = ESNLayer(reservoir_size=40, feedback_size=3)
        feedback = torch.randn(2, 8, 3)
        reservoir(feedback)
        warmed = reservoir.get_state().clone()

        reservoir = reservoir.double()
        # The next forward must continue from the (moved) warmed state, i.e.
        # _maybe_init_state must NOT re-init purely due to the dtype change.
        out = reservoir(feedback.double())
        assert out.dtype == torch.float64
        assert not torch.allclose(reservoir.state, warmed.double())  # state evolved
        assert (reservoir.state != 0).any()

    def test_maybe_init_state_no_reinit_on_dtype_change(self) -> None:
        """A dtype-only change must reuse the moved state, not allocate zeros."""
        reservoir = ESNLayer(reservoir_size=40, feedback_size=3).double()
        reservoir(torch.randn(2, 8, 3, dtype=torch.float64))
        before = reservoir.state

        # Same batch, dtype already matches the buffer: identity must be kept.
        reservoir._maybe_init_state(2, before.device, torch.float64)
        assert reservoir.state is before

    def test_maybe_init_state_reinits_on_batch_change(self) -> None:
        """A genuine batch-size change still triggers a fresh zero state."""
        reservoir = ESNLayer(reservoir_size=40, feedback_size=3)
        reservoir(torch.randn(2, 8, 3))
        assert reservoir.state.shape[0] == 2

        reservoir(torch.randn(5, 8, 3))  # different batch
        assert reservoir.state.shape[0] == 5

    def test_reset_to_none_then_reinit_still_a_buffer(self) -> None:
        """reset_state() → None → forward keeps ``state`` a buffer (moves on .to)."""
        reservoir = ESNLayer(reservoir_size=40, feedback_size=3)
        reservoir(torch.randn(2, 8, 3))
        reservoir.reset_state()
        assert reservoir.state is None

        reservoir(torch.randn(2, 8, 3))  # lazy re-init
        assert "state" in dict(reservoir.named_buffers())
        reservoir = reservoir.double()  # buffer still moves
        assert reservoir.state.dtype == torch.float64

    @pytest.mark.gpu
    @cuda_required
    def test_to_cuda_preserves_warmed_state(self) -> None:
        """Warm on CPU, ``.to('cuda')``: state moves to GPU with values intact."""
        reservoir = ESNLayer(reservoir_size=40, feedback_size=3)
        reservoir(torch.randn(2, 8, 3))
        warmed = reservoir.get_state().clone()

        reservoir = reservoir.to("cuda")
        assert reservoir.state.device.type == "cuda"
        assert torch.allclose(reservoir.state.cpu(), warmed)


class TestESNLayerTopology:
    """Topology-spec resolution through the layer (string/tuple/object)."""

    def test_reservoir_with_string_topology(self) -> None:
        """Reservoir initialization with string topology name."""
        reservoir = ESNLayer(
            reservoir_size=50,
            feedback_size=10,
            topology="erdos_renyi",
            spectral_radius=0.9,
        )

        # Should have initialized weight matrices
        assert reservoir.weight_feedback.shape == (50, 10)
        assert reservoir.weight_hh.shape == (50, 50)
        assert reservoir.weight_input is None  # No driving inputs

        # Check spectral radius is approximately correct
        assert abs(_spectral_radius(reservoir.weight_hh.data) - 0.9) < 0.05

    def test_reservoir_with_topology_object(self) -> None:
        """Reservoir with TopologyInitializer object."""
        topology = get_topology("watts_strogatz", k=4, p=0.1, seed=42)

        reservoir = ESNLayer(
            reservoir_size=40,
            feedback_size=5,
            topology=topology,
            spectral_radius=0.95,
        )

        assert reservoir.weight_hh.shape == (40, 40)
        assert abs(_spectral_radius(reservoir.weight_hh.data) - 0.95) < 0.05

    def test_reservoir_with_custom_graph_topology(self) -> None:
        """Reservoir with custom GraphTopology."""
        topology = GraphTopology(erdos_renyi_graph, {"p": 0.15, "directed": True, "seed": 42})

        reservoir = ESNLayer(
            reservoir_size=30,
            feedback_size=8,
            topology=topology,
        )

        assert reservoir.weight_hh.shape == (30, 30)
        assert not torch.all(reservoir.weight_hh == 0)

    def test_reservoir_topology_forward_pass(self) -> None:
        """Reservoir with topology works in forward pass."""
        reservoir = ESNLayer(
            reservoir_size=50,
            feedback_size=10,
            topology="erdos_renyi",
            spectral_radius=0.9,
        )

        feedback = torch.randn(4, 20, 10)  # (batch=4, time=20, features=10)

        output = reservoir(feedback)

        assert output.shape == (4, 20, 50)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_reservoir_topology_with_driving_inputs(self) -> None:
        """Reservoir with topology and driving inputs."""
        reservoir = ESNLayer(
            reservoir_size=60,
            feedback_size=5,
            input_size=3,
            topology="ring_chord",
            spectral_radius=0.85,
        )

        feedback = torch.randn(2, 15, 5)
        driving = torch.randn(2, 15, 3)

        output = reservoir(feedback, driving)

        assert output.shape == (2, 15, 60)
        assert reservoir.weight_input.shape == (60, 3)

    def test_different_topologies_produce_different_weights(self) -> None:
        """Different topologies produce different weight matrices."""
        reservoir1 = ESNLayer(
            reservoir_size=30,
            feedback_size=5,
            topology="erdos_renyi",
        )

        reservoir2 = ESNLayer(
            reservoir_size=30,
            feedback_size=5,
            topology="watts_strogatz",
        )

        # Weights should be different (very unlikely to be identical by chance)
        assert not torch.allclose(reservoir1.weight_hh, reservoir2.weight_hh)

    def test_reservoir_topology_invalid_type(self) -> None:
        """Invalid topology type raises error."""
        with pytest.raises(TypeError, match="Invalid topology spec type"):
            ESNLayer(
                reservoir_size=30,
                feedback_size=5,
                topology=123,  # Invalid type
            )

    def test_reservoir_with_tuple_topology_spec(self) -> None:
        """Reservoir with tuple (name, params) topology specification."""
        reservoir = ESNLayer(
            reservoir_size=50,
            feedback_size=10,
            topology=("watts_strogatz", {"k": 4, "p": 0.3, "seed": 42}),
            spectral_radius=0.9,
        )

        assert reservoir.weight_hh.shape == (50, 50)
        assert abs(_spectral_radius(reservoir.weight_hh.data) - 0.9) < 0.05

    def test_reservoir_with_tuple_initializer_spec(self) -> None:
        """Reservoir with tuple (name, params) initializer specification."""
        reservoir = ESNLayer(
            reservoir_size=50,
            feedback_size=10,
            feedback_initializer=("pseudo_diagonal", {"input_scaling": 0.5}),
            spectral_radius=0.9,
        )

        assert reservoir.weight_feedback.shape == (50, 10)

    def test_reservoir_topology_state_persistence(self) -> None:
        """Topology-based reservoir maintains state correctly."""
        reservoir = ESNLayer(
            reservoir_size=40,
            feedback_size=10,
            topology="erdos_renyi",
        )

        feedback1 = torch.randn(2, 10, 10)
        feedback2 = torch.randn(2, 10, 10)

        # First forward pass
        out1 = reservoir(feedback1)

        # Second forward pass (state should carry over)
        out2 = reservoir(feedback2)

        # Reset and run again
        reservoir.reset_state()
        out3 = reservoir(feedback1)

        # out3 should match out1 (same input, fresh state)
        # but out2 should be different (carried state)
        assert torch.allclose(out1, out3, rtol=1e-5)
        assert not torch.allclose(out2, out3, rtol=1e-5)

    def test_reservoir_topology_reproducibility(self) -> None:
        """Topology with seed produces reproducible results."""
        topology1 = GraphTopology(erdos_renyi_graph, {"p": 0.1, "directed": True, "seed": 42})
        topology2 = GraphTopology(erdos_renyi_graph, {"p": 0.1, "directed": True, "seed": 42})

        reservoir1 = ESNLayer(
            reservoir_size=30,
            feedback_size=5,
            topology=topology1,
        )

        reservoir2 = ESNLayer(
            reservoir_size=30,
            feedback_size=5,
            topology=topology2,
        )

        # Same seed should produce identical weights
        assert torch.allclose(reservoir1.weight_hh, reservoir2.weight_hh)

    @pytest.mark.parametrize("size", [50, 100, 200])
    def test_reservoir_topology_various_sizes(self, size: int) -> None:
        """Topology initialization with various reservoir sizes."""
        reservoir = ESNLayer(
            reservoir_size=size,
            feedback_size=5,
            topology="erdos_renyi",
            spectral_radius=0.9,
        )

        assert reservoir.weight_hh.shape == (size, size)
        assert abs(_spectral_radius(reservoir.weight_hh.data) - 0.9) < 0.1


class TestESNLayerBatchAndSequence:
    """Shape contract across batch sizes and sequence lengths."""

    @pytest.mark.parametrize("batch", [1, 4, 128])
    def test_batch_sizes(self, batch: int) -> None:
        """Output preserves arbitrary batch sizes, including batch=1."""
        reservoir = ESNLayer(reservoir_size=50, feedback_size=10)
        feedback = torch.randn(batch, 20, 10)

        output = reservoir(feedback)

        assert output.shape == (batch, 20, 50)

    def test_varying_batch_sizes_auto_reset_state(self) -> None:
        """State resets automatically when the batch size changes."""
        reservoir = ESNLayer(reservoir_size=50, feedback_size=10)

        # Process batch of 4
        out1 = reservoir(torch.randn(4, 10, 10))
        assert out1.shape == (4, 10, 50)

        # Process batch of 2 (should auto-reset state)
        out2 = reservoir(torch.randn(2, 10, 10))
        assert out2.shape == (2, 10, 50)

        # State should be for batch of 2 now
        assert reservoir.state.shape[0] == 2

    @pytest.mark.parametrize("seq_len", [1, 5, 100, 500])
    def test_sequence_lengths(self, seq_len: int) -> None:
        """Output preserves any sequence length, including a single timestep."""
        reservoir = ESNLayer(reservoir_size=100, feedback_size=10)
        feedback = torch.randn(2, seq_len, 10)

        output = reservoir(feedback)

        assert output.shape == (2, seq_len, 100)

    def test_varying_sequence_lengths_across_calls(self) -> None:
        """Multiple forward passes with different sequence lengths."""
        reservoir = ESNLayer(reservoir_size=50, feedback_size=10)

        for seq_len in [5, 10, 20, 50, 100]:
            reservoir.reset_state()
            feedback = torch.randn(2, seq_len, 10)
            output = reservoir(feedback)
            assert output.shape == (2, seq_len, 50)


class TestESNLayerDevice:
    """Device placement: parameters, outputs, and state must follow the layer."""

    def test_forward_on_device(self, device: torch.device) -> None:
        """Parameters and outputs land on the target device."""
        reservoir = ESNLayer(
            reservoir_size=100,
            feedback_size=10,
            topology="erdos_renyi",
            spectral_radius=0.9,
        ).to(device)

        # All parameters should be on the device
        assert reservoir.weight_feedback.device.type == device.type
        assert reservoir.weight_hh.device.type == device.type
        if reservoir.bias_h is not None:
            assert reservoir.bias_h.device.type == device.type

        feedback = torch.randn(4, 20, 10, device=device)
        output = reservoir(feedback)

        assert output.device.type == device.type
        assert output.shape == (4, 20, 100)
        assert not torch.isnan(output).any()

    def test_forward_with_driving_inputs_on_device(self, device: torch.device) -> None:
        """Driving-input path works on every device."""
        reservoir = ESNLayer(
            reservoir_size=100,
            feedback_size=10,
            input_size=5,
            topology="watts_strogatz",
        ).to(device)

        feedback = torch.randn(2, 15, 10, device=device)
        driving = torch.randn(2, 15, 5, device=device)

        output = reservoir(feedback, driving)

        assert output.device.type == device.type
        assert output.shape == (2, 15, 100)

    def test_state_persistence_on_device(self, device: torch.device) -> None:
        """Reservoir state lives on the device and reset preserves placement."""
        reservoir = ESNLayer(reservoir_size=50, feedback_size=10).to(device)

        feedback = torch.randn(2, 10, 10, device=device)

        # First forward pass
        out1 = reservoir(feedback)
        assert reservoir.state.device.type == device.type

        # Reset (with explicit batch size) and compare a fresh run
        reservoir.reset_state(batch_size=2)
        assert reservoir.state.device.type == device.type
        out3 = reservoir(feedback)

        # out3 should match out1 (same input, fresh state)
        assert torch.allclose(out1, out3, rtol=1e-5)

    @pytest.mark.gpu
    @cuda_required
    def test_device_round_trip(self) -> None:
        """Moving the layer CPU -> GPU -> CPU keeps it functional."""
        reservoir = ESNLayer(
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
        assert next(reservoir.parameters()).device.type == "cuda"
        feedback_gpu = torch.randn(2, 10, 10, device="cuda")
        out_gpu = reservoir(feedback_gpu)
        assert out_gpu.is_cuda

        # Move back to CPU
        reservoir = reservoir.cpu()
        out_cpu2 = reservoir(feedback_cpu)
        assert not out_cpu2.is_cuda
