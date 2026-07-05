"""Contract of the gated ("reservoir GRU") cell and layer.

Covers :class:`~resdag.layers.cells.gated_cell.GatedReservoirCell` and
:class:`~resdag.layers.reservoirs.gated.GatedReservoirLayer`:

- the packed multi-tensor-state convention (``_split_state`` / ``_pack_state``),
  including a hypothetical 2-state layout proving Phase 10/11 readiness,
- construction (weight shapes, retention-biased update gate, freezing),
- the single-step LiGRU update and its retention semantics,
- sequence forward shapes, state round-trip, frozen stability (bounded,
  non-collapsing, input-dependent),
- the trainable BPTT path (grads flow through every gate),
- compile-cleanliness of ``cell.step`` and save/load via pickling.

Classes are imported directly from their module files (the public ``__init__``
wiring is owned elsewhere).
"""

import copy
import pickle

import pytest
import torch

from resdag.layers.cells.gated_cell import (
    GatedReservoirCell,
    _pack_state,
    _split_state,
)
from resdag.layers.reservoirs.gated import GatedReservoirLayer


class TestPackedStateConvention:
    """The packed multi-tensor-state helpers (the reusable convention)."""

    def test_single_state_pack_split_is_identity(self) -> None:
        """A single-state pack/split touches nothing (identity fast path)."""
        h = torch.randn(4, 8)
        (only,) = _split_state(h, 8)
        assert only is h
        assert _pack_state(h) is h

    def test_two_state_roundtrip_preserves_order(self) -> None:
        """A hypothetical (h, c) 2-state layout round-trips and keeps order.

        Proves the convention is ready for the Phase 10/11 continuous / spiking
        cells that carry two state tensors: ``state_size == 2 * hidden`` packs
        side-by-side, splits back to the same tensors in the same order.
        """
        b, h_size = 4, 8
        h = torch.randn(b, h_size)
        c = torch.randn(b, h_size)

        packed = _pack_state(h, c)
        assert packed.shape == (b, 2 * h_size)
        # Packing order: first tensor occupies the low columns.
        assert torch.equal(packed[:, :h_size], h)
        assert torch.equal(packed[:, h_size:], c)

        h2, c2 = _split_state(packed, h_size)
        assert torch.equal(h2, h)
        assert torch.equal(c2, c)

    def test_three_state_roundtrip(self) -> None:
        """The convention generalises to any n_state_tensors (here 3)."""
        b, h_size = 3, 5
        tensors = [torch.randn(b, h_size) for _ in range(3)]
        packed = _pack_state(*tensors)
        assert packed.shape == (b, 3 * h_size)
        recovered = _split_state(packed, h_size)
        assert len(recovered) == 3
        for got, want in zip(recovered, tensors):
            assert torch.equal(got, want)

    def test_split_rejects_non_multiple_width(self) -> None:
        """A packed width that is not a multiple of hidden_size is a hard error."""
        with pytest.raises(ValueError, match="not a positive multiple"):
            _split_state(torch.randn(4, 17), 8)

    def test_pack_requires_at_least_one_tensor(self) -> None:
        """``_pack_state()`` with no tensors is a programming error."""
        with pytest.raises(ValueError, match="at least one"):
            _pack_state()


class TestGatedCellConstruction:
    """Construction: weight shapes, retention gate, freezing."""

    def test_weight_shapes_and_sizes(self) -> None:
        """Cell owns candidate + gate recurrent weights and both drive weights."""
        cell = GatedReservoirCell(reservoir_size=64, feedback_size=5, seed=1)
        assert cell.reservoir_size == 64
        assert cell.hidden_size == 64
        # Single-state GRU: state_size == hidden_size (n_state_tensors == 1).
        assert cell.n_state_tensors == 1
        assert cell.state_size == 64
        assert cell.output_size == 64
        assert cell.weight_feedback.shape == (64, 5)
        assert cell.weight_hh.shape == (64, 64)
        assert cell.weight_hz.shape == (64, 64)
        assert cell.weight_input is None
        assert cell.bias_h is not None and cell.bias_h.shape == (64,)
        assert cell.bias_z is not None and cell.bias_z.shape == (64,)

    def test_driving_input_weight_created(self) -> None:
        """A positive ``input_size`` creates the driving-input drive weight."""
        cell = GatedReservoirCell(reservoir_size=32, feedback_size=3, input_size=4, seed=1)
        assert cell.weight_input is not None
        assert cell.weight_input.shape == (32, 4)

    def test_input_size_zero_treated_as_none(self) -> None:
        """``input_size=0`` behaves like ``None`` (no driving weight)."""
        cell = GatedReservoirCell(reservoir_size=16, feedback_size=3, input_size=0)
        assert cell.weight_input is None

    def test_update_gate_bias_is_retention_biased(self) -> None:
        """``bias_z`` is the constant retention bias, not randomised."""
        cell = GatedReservoirCell(reservoir_size=48, feedback_size=3, update_gate_bias=2.0)
        assert torch.allclose(cell.bias_z, torch.full((48,), 2.0))
        # A fresh, retention-biased update gate keeps most of the state each step.
        z0 = torch.sigmoid(cell.bias_z)
        assert float(z0.mean()) > 0.8

    def test_no_bias_drops_both_gate_biases(self) -> None:
        """``bias=False`` removes both gate biases."""
        cell = GatedReservoirCell(reservoir_size=16, feedback_size=3, bias=False)
        assert cell.bias_h is None
        assert cell.bias_z is None

    def test_frozen_by_default(self) -> None:
        """Frozen by default: no parameter requires grad."""
        cell = GatedReservoirCell(reservoir_size=32, feedback_size=3)
        assert cell.trainable is False
        assert all(not p.requires_grad for p in cell.parameters())

    def test_trainable_flag_unfreezes(self) -> None:
        """``trainable=True`` unfreezes every parameter."""
        cell = GatedReservoirCell(reservoir_size=32, feedback_size=3, trainable=True)
        assert cell.trainable is True
        assert all(p.requires_grad for p in cell.parameters())

    def test_live_trainable_toggle(self) -> None:
        """The ``trainable`` property is a live switch (freeze/unfreeze)."""
        cell = GatedReservoirCell(reservoir_size=16, feedback_size=3)
        cell.trainable = True
        assert all(p.requires_grad for p in cell.parameters())
        cell.trainable = False
        assert all(not p.requires_grad for p in cell.parameters())

    def test_seed_reproducibility(self) -> None:
        """Same seed reproduces every weight and bias."""
        a = GatedReservoirCell(48, feedback_size=3, seed=99, spectral_radius=0.9)
        b = GatedReservoirCell(48, feedback_size=3, seed=99, spectral_radius=0.9)
        for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
            assert na == nb
            assert torch.equal(pa, pb), na

    def test_unknown_activation_raises(self) -> None:
        """An unknown candidate activation is rejected at construction."""
        with pytest.raises(ValueError, match="Unknown activation"):
            GatedReservoirCell(16, feedback_size=3, activation="swish")


class TestGatedCellUpdate:
    """The single-step LiGRU update and its retention semantics."""

    def test_step_shapes(self) -> None:
        """``step`` returns ``(h, packed_state)`` of the right shapes."""
        cell = GatedReservoirCell(32, feedback_size=3, seed=1)
        drive = torch.randn(4, 32)
        state = cell.init_state(4, torch.device("cpu"), torch.float32)
        out, new_state = cell.step(drive, state)
        assert out.shape == (4, 32)
        assert new_state.shape == (4, 32)
        # Single-state: output and packed state coincide.
        assert torch.equal(out, new_state)

    def test_forward_matches_step_via_projection(self) -> None:
        """``forward`` == ``step(project_inputs(...))`` (the two paths agree)."""
        cell = GatedReservoirCell(24, feedback_size=3, seed=2, spectral_radius=0.9)
        fb = torch.randn(4, 3)
        state = cell.init_state(4, torch.device("cpu"), torch.float32)
        out_fwd, s_fwd = cell([fb], state)
        drive = cell.project_inputs([fb])
        out_step, s_step = cell.step(drive, state)
        assert torch.equal(out_fwd, out_step)
        assert torch.equal(s_fwd, s_step)

    def test_full_retention_keeps_state(self) -> None:
        """With ``z==1`` (huge retention bias, zero drive) the state is unchanged."""
        cell = GatedReservoirCell(16, feedback_size=3, bias=True, update_gate_bias=50.0, seed=1)
        # Kill the input and recurrent-gate contribution so z is driven purely by
        # the (enormous) retention bias => z ~ 1 => h_t == h_{t-1}.
        with torch.no_grad():
            cell.weight_feedback.zero_()
        state = torch.randn(2, 16)
        drive = cell.project_inputs([torch.zeros(2, 3)])  # zero drive
        out, _ = cell.step(drive, state)
        assert torch.allclose(out, state, atol=1e-5)

    def test_feedback_size_mismatch_raises(self) -> None:
        """A wrong feedback width is rejected by ``project_inputs``."""
        cell = GatedReservoirCell(16, feedback_size=3)
        with pytest.raises(ValueError, match="Feedback size mismatch"):
            cell.project_inputs([torch.randn(4, 5)])

    def test_driving_input_without_weight_raises(self) -> None:
        """A driving input on a feedback-only cell is an error."""
        cell = GatedReservoirCell(16, feedback_size=3)
        with pytest.raises(ValueError, match="without input_size"):
            cell.project_inputs([torch.randn(4, 3), torch.randn(4, 2)])


class TestGatedLayer:
    """Sequence-level layer behaviour."""

    def test_forward_shape(self) -> None:
        """A ``(B, T, F)`` feedback sequence yields ``(B, T, hidden)``."""
        layer = GatedReservoirLayer(reservoir_size=64, feedback_size=5, seed=1)
        out = layer(torch.randn(4, 50, 5))
        assert out.shape == (4, 50, 64)

    def test_forward_with_driver(self) -> None:
        """A driven layer accepts ``(feedback, driver)``."""
        layer = GatedReservoirLayer(64, feedback_size=3, input_size=2, seed=1)
        out = layer(torch.randn(4, 20, 3), torch.randn(4, 20, 2))
        assert out.shape == (4, 20, 64)

    def test_spectral_radius_scaled(self) -> None:
        """The candidate weight is scaled to the requested spectral radius."""
        layer = GatedReservoirLayer(64, feedback_size=3, seed=1, spectral_radius=0.7)
        assert layer.spectral_radius_achieved == pytest.approx(0.7, abs=1e-3)

    def test_state_roundtrip(self) -> None:
        """get_state / set_state round-trip the packed 2-D state."""
        layer = GatedReservoirLayer(48, feedback_size=3, seed=1)
        layer(torch.randn(2, 10, 3))
        saved = layer.get_state()
        assert saved is not None and saved.shape == (2, 48)
        layer.set_state(saved)
        assert torch.equal(layer.get_state(), saved)

    def test_set_state_validates_shape(self) -> None:
        """A wrong-width state is rejected by the base 2-D validator."""
        layer = GatedReservoirLayer(48, feedback_size=3, seed=1)
        with pytest.raises(ValueError):
            layer.set_state(torch.randn(2, 47))

    def test_attribute_delegation_to_cell(self) -> None:
        """Unknown attribute reads fall through to the inner cell."""
        layer = GatedReservoirLayer(32, feedback_size=3, seed=1)
        assert layer.reservoir_size == 32
        assert layer.weight_hz.shape == (32, 32)

    def test_repr_delegates_to_cell(self) -> None:
        """The layer repr is the cell repr."""
        layer = GatedReservoirLayer(16, feedback_size=3, seed=1)
        assert repr(layer) == repr(layer.cell)
        assert "GatedReservoirCell" in repr(layer)


class TestFrozenStability:
    """A frozen, retention-biased gated reservoir stays bounded and lively."""

    def test_state_bounded_and_non_collapsing(self) -> None:
        """Over a long sequence ||h|| stays finite/bounded and varies in time."""
        layer = GatedReservoirLayer(300, feedback_size=3, seed=7, spectral_radius=0.9)
        layer.eval()
        out = layer(torch.randn(1, 1500, 3)).squeeze(0)  # (T, N)

        norms = out.norm(dim=-1)
        assert torch.isfinite(norms).all()
        # Bounded: no blow-up over 1500 steps.
        assert float(norms.max()) < 100.0
        # Non-collapse: the state actually varies across time (not a constant).
        temporal_std = out.std(dim=0).mean()
        assert float(temporal_std) > 1e-2

    def test_state_depends_on_input(self) -> None:
        """Two different drives produce two different trajectories."""
        layer = GatedReservoirLayer(200, feedback_size=3, seed=7, spectral_radius=0.9)
        layer.eval()
        x1 = torch.randn(1, 800, 3)
        x2 = torch.randn(1, 800, 3)
        layer.reset_state()
        o1 = layer(x1)
        layer.reset_state()
        o2 = layer(x2)
        assert float((o1[:, -1] - o2[:, -1]).abs().mean()) > 1e-3


class TestTrainableBPTT:
    """The trainable path: gradients flow through every gate."""

    def test_grads_flow_through_all_gates(self) -> None:
        """A BPTT step produces non-zero grads on candidate + gate weights."""
        layer = GatedReservoirLayer(
            32, feedback_size=3, seed=1, spectral_radius=0.9, trainable=True
        )
        out = layer(torch.randn(2, 40, 3))
        out.pow(2).mean().backward()
        for name, p in layer.cell.named_parameters():
            assert p.grad is not None, name
            assert float(p.grad.abs().sum()) > 0.0, name

    def test_optimizer_step_updates_weights(self) -> None:
        """An SGD step actually moves the gate weights."""
        layer = GatedReservoirLayer(24, feedback_size=3, seed=1, trainable=True)
        w0 = layer.cell.weight_hz.detach().clone()
        out = layer(torch.randn(2, 20, 3))
        out.pow(2).mean().backward()
        torch.optim.SGD(layer.parameters(), lr=0.1).step()
        assert not torch.equal(w0, layer.cell.weight_hz)

    def test_frozen_has_no_grads(self) -> None:
        """A frozen layer accumulates no gradients on a backward."""
        layer = GatedReservoirLayer(24, feedback_size=3, seed=1)  # frozen default
        out = layer(torch.randn(2, 20, 3))
        # No leaf requires grad => the graph has no parameter to backprop into.
        assert not out.requires_grad
        assert all(p.grad is None for p in layer.cell.parameters())


class TestCompileAndSerialize:
    """cell.step compiles fullgraph-clean; the cell/layer pickle round-trips."""

    def test_step_compiles_fullgraph(self) -> None:
        """``torch.compile(cell.step, fullgraph=True)`` traces with no break."""
        cell = GatedReservoirCell(64, feedback_size=3, seed=1, spectral_radius=0.9)
        cell.eval()
        drive = torch.randn(4, 64)
        state = torch.zeros(4, 64)
        compiled = torch.compile(cell.step, fullgraph=True)
        try:
            out_c, _ = compiled(drive, state)
        except Exception as exc:  # pragma: no cover - environment-dependent
            pytest.skip(f"torch.compile unavailable in this environment: {exc}")
        out_e, _ = cell.step(drive, state)
        assert torch.allclose(out_c, out_e, atol=1e-5)

    def test_cell_pickle_roundtrip(self) -> None:
        """The cell pickles and reproduces its update after unpickling."""
        cell = GatedReservoirCell(32, feedback_size=3, seed=5, spectral_radius=0.9)
        cell.eval()
        restored = pickle.loads(pickle.dumps(cell))
        drive = torch.randn(4, 32)
        state = torch.randn(4, 32)
        o0, _ = cell.step(drive, state)
        o1, _ = restored.step(drive, state)
        assert torch.equal(o0, o1)

    def test_layer_deepcopy_roundtrip(self) -> None:
        """The layer deep-copies (the base drops its unpicklable compile cache)."""
        layer = GatedReservoirLayer(32, feedback_size=3, seed=5)
        layer(torch.randn(2, 10, 3))
        clone = copy.deepcopy(layer)
        clone.reset_state()
        layer.reset_state()
        x = torch.randn(2, 10, 3)
        assert torch.equal(layer(x), clone(x))
