"""ESNModel forward-pass and reservoir-state management contracts.

Pins down the model-level API layered on top of pytorch_symbolic:
forward shape/determinism, ``reset_reservoirs``, ``get/set_reservoir_states``
(clone semantics, strict key checking), ``set_random_reservoir_states``,
and teacher-forced ``warmup``.

Persistence is covered in ``test_save_load.py``; autoregressive generation
in ``test_forecast.py``.
"""

import pytest
import torch


class TestESNModelForward:
    """Forward pass through a reservoir -> readout model."""

    def test_forward_shape_on_device(self, make_tiny_model, device: torch.device) -> None:
        """Output is (batch, time, output_size) on the model's device."""
        model = make_tiny_model(device=device)
        x = torch.randn(2, 30, 3, device=device)

        output = model(x)

        assert output.shape == (2, 30, 3)
        assert output.device.type == device.type

    def test_forward_reproducible_after_reset(self, make_tiny_model, seeded: None) -> None:
        """Same input from a reset state reproduces the same output."""
        model = make_tiny_model()
        x = torch.randn(1, 20, 3)

        model.reset_reservoirs()
        out1 = model(x)
        model.reset_reservoirs()
        out2 = model(x)

        assert torch.allclose(out1, out2, rtol=1e-5, atol=1e-6)

    def test_forward_is_stateful_without_reset(self, make_tiny_model, seeded: None) -> None:
        """Without a reset, the carried state changes the output."""
        model = make_tiny_model()
        x = torch.randn(1, 20, 3)

        out1 = model(x)
        out2 = model(x)  # continues from the state left by out1

        assert not torch.allclose(out1, out2)


class TestESNModelStateManagement:
    """reset / get / set / randomize reservoir states."""

    def test_construction_leaves_trace_state_reset_clears_it(self, make_tiny_model) -> None:
        """Symbolic tracing at build time leaves a state; reset clears it.

        pytorch_symbolic runs the layers on placeholder data while the graph
        is being defined, so a freshly built model already carries a batch-1
        reservoir state.  ``reset_reservoirs()`` must clear it.
        """
        model = make_tiny_model()

        assert len(model.get_reservoir_states()) > 0  # trace residue

        model.reset_reservoirs()

        assert model.get_reservoir_states() == {}

    def test_reset_reservoirs_clears_states(self, make_tiny_model, seeded: None) -> None:
        """reset_reservoirs drops every reservoir state."""
        model = make_tiny_model()
        model(torch.randn(2, 10, 3))
        assert len(model.get_reservoir_states()) > 0

        model.reset_reservoirs()

        assert model.get_reservoir_states() == {}

    def test_get_reservoir_states_returns_clones(self, make_tiny_model, seeded: None) -> None:
        """Mutating a returned state tensor must not touch the model."""
        model = make_tiny_model()
        model(torch.randn(2, 10, 3))

        states = model.get_reservoir_states()
        snapshot = {name: state.clone() for name, state in states.items()}
        for state in states.values():
            state.add_(123.0)

        unchanged = model.get_reservoir_states()
        for name, state in unchanged.items():
            assert torch.equal(state, snapshot[name])

    def test_set_reservoir_states_roundtrip(self, make_tiny_model, seeded: None) -> None:
        """Restoring saved states reproduces the continued trajectory."""
        model = make_tiny_model()
        x1 = torch.randn(1, 20, 3)
        x2 = torch.randn(1, 10, 3)

        model(x1)
        saved = model.get_reservoir_states()

        out_a = model(x2)  # consumes the saved state
        model.set_reservoir_states(saved)
        out_b = model(x2)  # restored: must reproduce out_a

        assert torch.allclose(out_a, out_b, rtol=1e-5, atol=1e-6)

    def test_set_reservoir_states_strict_mismatch_raises(
        self, make_tiny_model, seeded: None
    ) -> None:
        """strict=True rejects state dicts with missing/unknown keys."""
        model = make_tiny_model()
        model(torch.randn(1, 5, 3))

        with pytest.raises(KeyError, match="strict"):
            model.set_reservoir_states({"not_a_reservoir": torch.zeros(1, 32)})

    def test_set_random_reservoir_states_lazy(self, make_tiny_model) -> None:
        """set_random_reservoir_states(batch_size=N) initialises and randomizes."""
        model = make_tiny_model()
        model.reset_reservoirs()  # drop the build-time trace state

        model.set_random_reservoir_states(batch_size=4)

        states = model.get_reservoir_states()
        assert len(states) > 0
        for state in states.values():
            assert state.shape[0] == 4
            assert not torch.all(state == 0)


class TestESNModelWarmup:
    """Teacher-forced warmup synchronizes reservoir states."""

    def test_warmup_populates_states(self, make_tiny_model, seeded: None) -> None:
        """warmup returns None by default and leaves states behind."""
        model = make_tiny_model()
        x = torch.randn(1, 25, 3)

        result = model.warmup(x)

        assert result is None
        assert len(model.get_reservoir_states()) > 0

    def test_warmup_return_outputs(self, make_tiny_model, seeded: None) -> None:
        """return_outputs=True yields the teacher-forced outputs."""
        model = make_tiny_model()
        x = torch.randn(2, 25, 3)

        outputs = model.warmup(x, return_outputs=True)

        assert outputs is not None
        assert outputs.shape == (2, 25, 3)

    def test_warmup_resets_by_default(self, make_tiny_model, seeded: None) -> None:
        """warmup(reset=True) starts from a fresh state."""
        model = make_tiny_model()
        x = torch.randn(1, 15, 3)

        fresh = model.warmup(x, return_outputs=True)

        # Pollute the state, then warmup again: must match the fresh run.
        model(torch.randn(1, 7, 3))
        again = model.warmup(x, return_outputs=True)

        assert torch.allclose(fresh, again, rtol=1e-5, atol=1e-6)

    def test_warmup_reset_false_continues_from_state(self, make_tiny_model, seeded: None) -> None:
        """warmup(reset=False) continues from the current state."""
        model = make_tiny_model()
        x1 = torch.randn(1, 10, 3)
        x2 = torch.randn(1, 10, 3)

        model.warmup(x1)
        continued = model.warmup(x2, return_outputs=True, reset=False)
        fresh = model.warmup(x2, return_outputs=True)  # reset=True default

        assert not torch.allclose(continued, fresh)

    def test_warmup_without_inputs_raises(self, make_tiny_model) -> None:
        """warmup() with no inputs raises ValueError."""
        model = make_tiny_model()

        with pytest.raises(ValueError, match="At least one input"):
            model.warmup()
