"""Single-step contract of ESNCell.

Pins down the cell-level interface that BaseReservoirLayer relies on:
``state_size`` / ``output_size`` properties, ``init_state`` semantics,
the leaky-integrator update equation of ``forward``, input validation in
``project_inputs``, parameter freezing, and ``__repr__``.

Sequence-level behaviour (state management, fast path) is covered in
``test_esn_layer.py`` and ``test_fast_path.py``.
"""

import pytest
import torch

from resdag.layers.cells import ESNCell


class TestESNCellConstruction:
    """Construction: weight shapes, properties, and parameter freezing."""

    def test_basic_construction(self) -> None:
        """Cell owns all weight matrices with documented shapes."""
        cell = ESNCell(reservoir_size=64, feedback_size=5)

        assert cell.reservoir_size == 64
        assert cell.feedback_size == 5
        assert cell.weight_feedback.shape == (64, 5)
        assert cell.weight_hh.shape == (64, 64)
        assert cell.weight_input is None
        assert cell.bias_h is not None
        assert cell.bias_h.shape == (64,)

    def test_state_and_output_size_equal_reservoir_size(self) -> None:
        """For the ESN cell, state_size == output_size == reservoir_size."""
        cell = ESNCell(reservoir_size=32, feedback_size=3)

        assert cell.state_size == 32
        assert cell.output_size == 32

    def test_input_size_creates_input_weights(self) -> None:
        """input_size creates the driving-input weight matrix."""
        cell = ESNCell(reservoir_size=16, feedback_size=3, input_size=7)

        assert cell.input_size == 7
        assert cell.weight_input.shape == (16, 7)

    def test_input_size_zero_treated_as_none(self) -> None:
        """input_size=0 must not create a zero-column weight matrix."""
        cell = ESNCell(reservoir_size=16, feedback_size=3, input_size=0)

        assert cell.input_size is None
        assert cell.weight_input is None

    def test_frozen_by_default(self) -> None:
        """trainable=False (default) freezes every parameter."""
        cell = ESNCell(reservoir_size=16, feedback_size=3, input_size=2)

        assert all(not p.requires_grad for p in cell.parameters())

    def test_trainable_exposes_grads(self) -> None:
        """trainable=True leaves all parameters tracked by autograd."""
        cell = ESNCell(reservoir_size=16, feedback_size=3, trainable=True)

        params = list(cell.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)

    def test_activation_property_returns_name(self) -> None:
        """The activation property reports the configured name."""
        cell = ESNCell(reservoir_size=8, feedback_size=2, activation="relu")

        assert cell.activation == "relu"

    def test_invalid_activation_raises(self) -> None:
        """Unknown activation names raise ValueError at construction."""
        with pytest.raises(ValueError, match="Unknown activation"):
            ESNCell(reservoir_size=8, feedback_size=2, activation="bogus")

    def test_repr(self) -> None:
        """__repr__ reports size, feedback, input and spectral radius."""
        cell = ESNCell(reservoir_size=8, feedback_size=2, input_size=3, spectral_radius=0.9)

        r = repr(cell)
        assert "ESNCell" in r
        assert "reservoir_size=8" in r
        assert "feedback_size=2" in r
        assert "input_size=3" in r
        assert "spectral_radius=0.9" in r


class TestESNCellInitState:
    """init_state must return zeros with the requested placement."""

    def test_shape_and_zeros(self) -> None:
        """Zero state of shape (batch, reservoir_size)."""
        cell = ESNCell(reservoir_size=32, feedback_size=3)

        state = cell.init_state(4, torch.device("cpu"), torch.float32)

        assert state.shape == (4, 32)
        assert torch.all(state == 0)

    def test_dtype_respected(self) -> None:
        """Requested dtype is honoured."""
        cell = ESNCell(reservoir_size=8, feedback_size=2)

        state = cell.init_state(1, torch.device("cpu"), torch.float64)

        assert state.dtype == torch.float64

    def test_device_respected(self, device: torch.device) -> None:
        """Requested device is honoured."""
        cell = ESNCell(reservoir_size=8, feedback_size=2).to(device)

        state = cell.init_state(2, device, torch.float32)

        assert state.device.type == device.type


class TestESNCellForwardStep:
    """Single-step update semantics."""

    def test_output_is_new_state(self) -> None:
        """forward returns (output, new_state) with output is new_state."""
        cell = ESNCell(reservoir_size=16, feedback_size=3)
        fb = torch.randn(2, 3)
        state = cell.init_state(2, fb.device, fb.dtype)

        output, new_state = cell([fb], state)

        assert output is new_state
        assert output.shape == (2, 16)

    def test_matches_leaky_update_equation(self) -> None:
        """h_t = (1-a) h_{t-1} + a tanh(W_fb x + W_rec h_{t-1} + b)."""
        torch.manual_seed(3)
        cell = ESNCell(reservoir_size=12, feedback_size=4, leak_rate=0.3)
        fb = torch.randn(5, 4)
        state = torch.randn(5, 12)

        output, _ = cell([fb], state)

        pre = fb @ cell.weight_feedback.T + state @ cell.weight_hh.T + cell.bias_h
        expected = torch.lerp(state, torch.tanh(pre), 0.3)
        assert torch.allclose(output, expected, atol=1e-6)

    def test_driving_input_contributes(self) -> None:
        """The driving-input projection enters the pre-activation."""
        torch.manual_seed(4)
        cell = ESNCell(reservoir_size=12, feedback_size=3, input_size=2)
        fb = torch.randn(2, 3)
        driver = torch.randn(2, 2)
        state = cell.init_state(2, fb.device, fb.dtype)

        out_with, _ = cell([fb, driver], state)
        out_without, _ = cell([fb], state)

        assert not torch.allclose(out_with, out_without)

    def test_feedback_size_mismatch_raises(self) -> None:
        """Wrong feedback feature dimension raises at the cell level."""
        cell = ESNCell(reservoir_size=8, feedback_size=3)
        state = cell.init_state(1, torch.device("cpu"), torch.float32)

        with pytest.raises(ValueError, match="Feedback size mismatch"):
            cell([torch.randn(1, 4)], state)

    def test_driver_without_input_size_raises(self) -> None:
        """Driving input on a cell built without input_size raises."""
        cell = ESNCell(reservoir_size=8, feedback_size=3)
        state = cell.init_state(1, torch.device("cpu"), torch.float32)

        with pytest.raises(ValueError, match="without input_size"):
            cell([torch.randn(1, 3), torch.randn(1, 2)], state)

    def test_driver_size_mismatch_raises(self) -> None:
        """Wrong driving-input feature dimension raises."""
        cell = ESNCell(reservoir_size=8, feedback_size=3, input_size=2)
        state = cell.init_state(1, torch.device("cpu"), torch.float32)

        with pytest.raises(ValueError, match="Driving input size mismatch"):
            cell([torch.randn(1, 3), torch.randn(1, 5)], state)


class TestESNCellNoise:
    """Train-mode additive Gaussian state-noise regularizer."""

    def test_default_noise_is_zero(self) -> None:
        """noise defaults to 0.0 (no behaviour change)."""
        cell = ESNCell(reservoir_size=8, feedback_size=2)

        assert cell.noise == 0.0

    def test_negative_noise_raises(self) -> None:
        """A negative noise stddev is rejected at construction."""
        with pytest.raises(ValueError, match="noise must be non-negative"):
            ESNCell(reservoir_size=8, feedback_size=2, noise=-0.1)

    def test_default_noise_is_bit_identical(self) -> None:
        """noise=0.0 leaves the train-mode output bit-identical to no noise."""
        cell = ESNCell(reservoir_size=16, feedback_size=3, seed=0)
        cell.train()
        fb = torch.randn(4, 3)
        state = cell.init_state(4, fb.device, fb.dtype)

        out_a, _ = cell([fb], state)
        out_b, _ = cell([fb], state)

        # No noise => deterministic and identical across calls in train mode.
        assert torch.equal(out_a, out_b)

    def test_noise_perturbs_state_in_train_mode(self) -> None:
        """noise > 0 in train() mode perturbs the output state."""
        noisy = ESNCell(reservoir_size=32, feedback_size=3, noise=0.1, seed=0)
        clean = ESNCell(reservoir_size=32, feedback_size=3, noise=0.0, seed=0)
        noisy.train()
        clean.train()
        fb = torch.randn(4, 3)
        state = clean.init_state(4, fb.device, fb.dtype)

        out_noisy, _ = noisy([fb], state)
        out_clean, _ = clean([fb], state)

        # Same seed => identical weights, so any difference is the noise.
        assert torch.equal(noisy.weight_hh, clean.weight_hh)
        assert not torch.allclose(out_noisy, out_clean)

    def test_noise_is_noop_in_eval_mode(self) -> None:
        """noise > 0 is a no-op under eval(), matching the noiseless cell."""
        noisy = ESNCell(reservoir_size=32, feedback_size=3, noise=0.5, seed=0)
        clean = ESNCell(reservoir_size=32, feedback_size=3, noise=0.0, seed=0)
        noisy.eval()
        clean.eval()
        fb = torch.randn(4, 3)
        state = clean.init_state(4, fb.device, fb.dtype)

        out_noisy, _ = noisy([fb], state)
        out_clean, _ = clean([fb], state)

        assert torch.equal(out_noisy, out_clean)

    def test_noise_output_shape(self) -> None:
        """Noisy output keeps the (batch, reservoir_size) shape."""
        cell = ESNCell(reservoir_size=24, feedback_size=3, noise=0.2, seed=0)
        cell.train()
        fb = torch.randn(5, 3)
        state = cell.init_state(5, fb.device, fb.dtype)

        output, new_state = cell([fb], state)

        assert output.shape == (5, 24)
        assert new_state.shape == (5, 24)

    def test_noise_is_reproducible_under_seed(self) -> None:
        """Two seeded cells produce identical noise streams in train mode."""
        a = ESNCell(reservoir_size=32, feedback_size=3, noise=0.3, seed=123)
        b = ESNCell(reservoir_size=32, feedback_size=3, noise=0.3, seed=123)
        a.train()
        b.train()
        fb = torch.randn(4, 3)
        state = a.init_state(4, fb.device, fb.dtype)

        # Step both through several timesteps; the noise streams must agree.
        for _ in range(5):
            out_a, state_a = a([fb], state)
            out_b, state_b = b([fb], state)
            assert torch.equal(out_a, out_b)
            state = state_a

    def test_forward_and_step_apply_noise_identically(self) -> None:
        """forward() and the step() fast path inject the same noise."""
        cell_fwd = ESNCell(reservoir_size=20, feedback_size=3, noise=0.25, seed=7)
        cell_step = ESNCell(reservoir_size=20, feedback_size=3, noise=0.25, seed=7)
        cell_fwd.train()
        cell_step.train()
        fb = torch.randn(4, 3)
        state = cell_fwd.init_state(4, fb.device, fb.dtype)

        out_fwd, _ = cell_fwd([fb], state)

        # step() consumes a precomputed projection; build it the same way the
        # layer fast path does, then compare the perturbed states.  The two
        # paths differ only by the fused-vs-unfused recurrent matmul (~1e-6),
        # so an identical noise stream keeps them within that same band.
        projected = cell_step.project_inputs([fb])
        out_step, _ = cell_step.step(projected, state)

        assert torch.allclose(out_fwd, out_step, atol=1e-6)

    def test_forward_step_noise_matches_added_epsilon(self) -> None:
        """The noise added by forward()/step() is the *same* perturbation.

        Subtracting the noiseless output isolates the injected epsilon; the
        two paths must inject the identical noise tensor (same seeded stream),
        not merely a sample of the same magnitude.
        """
        noisy = ESNCell(reservoir_size=20, feedback_size=3, noise=0.4, seed=11)
        clean = ESNCell(reservoir_size=20, feedback_size=3, noise=0.0, seed=11)
        for c in (noisy, clean):
            c.train()
        fb = torch.randn(4, 3)
        state = clean.init_state(4, fb.device, fb.dtype)

        # Noise injected on the forward() path.
        eps_fwd = noisy([fb], state)[0] - clean([fb], state)[0]

        # Re-seed the noise stream by clearing the per-device cache, then take
        # the step() path on a fresh cell so its first draw matches forward's.
        noisy_step = ESNCell(reservoir_size=20, feedback_size=3, noise=0.4, seed=11)
        clean_step = ESNCell(reservoir_size=20, feedback_size=3, noise=0.0, seed=11)
        for c in (noisy_step, clean_step):
            c.train()
        proj_noisy = noisy_step.project_inputs([fb])
        proj_clean = clean_step.project_inputs([fb])
        eps_step = noisy_step.step(proj_noisy, state)[0] - clean_step.step(proj_clean, state)[0]

        assert torch.allclose(eps_fwd, eps_step, atol=1e-6)
