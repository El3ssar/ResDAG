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
