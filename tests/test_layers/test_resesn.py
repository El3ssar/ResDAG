"""Behavioural contract of the ResESN cell and layer.

Pins down the Residual Echo State Network (Ceni & Gallicchio, 2024; shallow
equation reviewed in arXiv:2508.21172):

- construction: weight shapes, frozen orthogonal skip, alpha/beta validation,
- the decoupled two-branch update (``alpha * O x + beta * phi(...)``),
- the ES2N special case (``alpha = 1 - beta``) and the leaky-ESN special case,
- ``forward`` / ``step`` consistency (the compiled-forecast fast path),
- live alpha/beta writes through both cell and layer,
- seed reproducibility and basic state management.

Imports are from the concrete module files: the public ``__init__`` exports are
wired separately.
"""

import pytest
import torch

from resdag.layers.cells.es2n_cell import ES2NCell
from resdag.layers.cells.resesn_cell import ResESNCell
from resdag.layers.reservoirs.resesn import ResESNLayer


class TestResESNCellConstruction:
    """Construction: weight shapes, the orthogonal skip, and freezing."""

    def test_basic_construction(self) -> None:
        """Cell owns ESN weights plus a square orthogonal skip buffer."""
        cell = ResESNCell(reservoir_size=64, feedback_size=5, alpha=0.6, beta=0.4)

        assert cell.reservoir_size == 64
        assert cell.alpha == 0.6
        assert cell.beta == 0.4
        assert cell.weight_hh.shape == (64, 64)
        assert cell.skip_matrix.shape == (64, 64)

    def test_state_and_output_size_equal_reservoir_size(self) -> None:
        """state_size == output_size == reservoir_size (single-state cell)."""
        cell = ResESNCell(reservoir_size=32, feedback_size=3)
        assert cell.state_size == 32
        assert cell.output_size == 32

    def test_skip_matrix_is_orthogonal_and_frozen(self) -> None:
        """O is orthogonal and a non-trainable buffer."""
        cell = ResESNCell(reservoir_size=96, feedback_size=3, seed=2)
        o = cell.skip_matrix
        assert (o @ o.t() - torch.eye(96)).abs().max().item() < 1e-5
        assert "skip_matrix" not in dict(cell.named_parameters())

    def test_frozen_by_default(self) -> None:
        """trainable=False (default) freezes every parameter."""
        cell = ResESNCell(reservoir_size=16, feedback_size=3, input_size=2)
        assert all(not p.requires_grad for p in cell.parameters())


class TestResESNValidation:
    """alpha / beta validation and the live-write contract."""

    @pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
    def test_nonfinite_alpha_rejected(self, bad: float) -> None:
        """alpha must be finite."""
        with pytest.raises(ValueError, match="alpha must be finite"):
            ResESNCell(reservoir_size=8, feedback_size=2, alpha=bad)

    @pytest.mark.parametrize("bad", [float("inf"), float("nan")])
    def test_nonfinite_beta_rejected(self, bad: float) -> None:
        """beta must be finite."""
        with pytest.raises(ValueError, match="beta must be finite"):
            ResESNCell(reservoir_size=8, feedback_size=2, beta=bad)

    def test_alpha_beta_live_write_through_layer(self) -> None:
        """layer.alpha / layer.beta forward to the cell, no shadow attrs."""
        layer = ResESNLayer(16, feedback_size=3, alpha=0.5, beta=0.9, seed=1)
        layer.alpha = 0.7
        layer.beta = 0.2
        assert layer.cell.alpha == 0.7
        assert layer.cell.beta == 0.2
        assert "alpha" not in layer.__dict__
        assert "beta" not in layer.__dict__


class TestResESNUpdateMath:
    """The decoupled two-branch ResESN state update."""

    def test_update_matches_explicit_formula(self) -> None:
        """forward realises alpha*O x + beta*phi(W_r x + proj) exactly."""
        cell = ResESNCell(48, 3, alpha=0.7, beta=0.5, spectral_radius=0.9, seed=7)
        state, fb = torch.randn(2, 48), torch.randn(2, 3)

        out, _ = cell([fb], state.clone())

        proj = torch.nn.functional.linear(fb, cell.weight_feedback) + cell.bias_h
        nonlinear = torch.tanh(state @ cell.weight_hh.t() + proj)
        skip = state @ cell.skip_matrix.t()
        expected = 0.7 * skip + 0.5 * nonlinear
        assert torch.allclose(out, expected, atol=1e-6)

    def test_forward_equals_step(self) -> None:
        """forward and the projected step fast path agree (with a driver)."""
        cell = ResESNCell(64, 3, input_size=2, alpha=0.6, beta=0.4, spectral_radius=0.9, seed=5)
        state = torch.randn(2, 64)
        fb, dr = torch.randn(2, 3), torch.randn(2, 2)

        out_f, _ = cell.forward([fb, dr], state.clone())
        proj = cell.project_inputs([fb, dr])
        out_s, _ = cell.step(proj, state.clone())
        assert torch.equal(out_f, out_s)

    def test_reduces_to_es2n_when_alpha_is_one_minus_beta(self) -> None:
        """alpha = 1 - beta recovers ES2N bit-for-bit (same seed => same O, W)."""
        beta = 0.3
        res = ResESNCell(64, 3, alpha=1.0 - beta, beta=beta, spectral_radius=0.9, seed=11)
        es2 = ES2NCell(64, 3, beta=beta, spectral_radius=0.9, seed=11)
        assert torch.equal(res.weight_hh, es2.weight_hh)
        assert torch.equal(res.skip_matrix, es2.skip_matrix)

        state, fb = torch.randn(2, 64), torch.randn(2, 3)
        o1, _ = res([fb], state.clone())
        o2, _ = es2([fb], state.clone())
        assert torch.equal(o1, o2)


class TestResESNLayer:
    """Sequence-level behaviour of the layer wrapper."""

    def test_forward_shape_feedback_only(self) -> None:
        """(batch, time, feedback) -> (batch, time, reservoir_size)."""
        layer = ResESNLayer(200, feedback_size=3, alpha=0.5, beta=0.5, seed=1)
        out = layer(torch.randn(4, 40, 3))
        assert out.shape == (4, 40, 200)

    def test_forward_shape_with_driver(self) -> None:
        """A driving input is accepted alongside feedback."""
        layer = ResESNLayer(64, feedback_size=3, input_size=2, seed=1)
        out = layer(torch.randn(2, 25, 3), torch.randn(2, 25, 2))
        assert out.shape == (2, 25, 64)

    def test_seed_reproducibility_includes_skip(self) -> None:
        """Same seed => identical weights AND identical orthogonal skip."""
        a = ResESNLayer(50, feedback_size=3, alpha=0.5, beta=0.5, spectral_radius=0.9, seed=42)
        b = ResESNLayer(50, feedback_size=3, alpha=0.5, beta=0.5, spectral_radius=0.9, seed=42)
        assert torch.equal(a.weight_hh, b.weight_hh)
        assert torch.equal(a.skip_matrix, b.skip_matrix)

    def test_delegates_attributes_to_cell(self) -> None:
        """Unknown attribute reads fall through to the wrapped cell."""
        layer = ResESNLayer(32, feedback_size=3, alpha=0.6, beta=0.4, seed=1)
        assert layer.reservoir_size == 32
        assert layer.alpha == 0.6
        assert layer.beta == 0.4

    def test_state_reset(self) -> None:
        """State management inherited from BaseReservoirLayer works."""
        layer = ResESNLayer(32, feedback_size=3, seed=1)
        layer(torch.randn(2, 10, 3))
        assert layer.get_state().shape == (2, 32)
        layer.reset_state()
        assert layer.get_state() is None
