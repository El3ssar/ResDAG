"""Behavioural contract of the ES2N cell and layer.

Pins down the Edge-of-Stability State Network (Ceni & Gallicchio,
arXiv:2308.02902):

- construction: weight shapes, frozen orthogonal skip, ``beta`` validation,
- the two-branch update math (``beta * phi(...) + (1 - beta) * O x``),
- the edge-of-stability property (norm preservation as ``beta -> 0``),
- the ``beta = 1`` reduction to a standard non-leaky ESN,
- ``forward`` / ``step`` consistency (the compiled-forecast fast path),
- live ``beta`` writes through both cell and layer,
- seed reproducibility (weights *and* the orthogonal skip),
- driving-input support and basic state management.

Imports are from the concrete module files: the public ``__init__`` exports are
wired separately.
"""

import pytest
import torch

from resdag.layers.cells.es2n_cell import ES2NCell
from resdag.layers.cells.esn_cell import ESNCell
from resdag.layers.reservoirs.es2n import ES2NLayer


class TestES2NCellConstruction:
    """Construction: weight shapes, the orthogonal skip, and freezing."""

    def test_basic_construction(self) -> None:
        """Cell owns ESN weights plus a square orthogonal skip buffer."""
        cell = ES2NCell(reservoir_size=64, feedback_size=5, beta=0.3)

        assert cell.reservoir_size == 64
        assert cell.feedback_size == 5
        assert cell.beta == 0.3
        assert cell.weight_feedback.shape == (64, 5)
        assert cell.weight_hh.shape == (64, 64)
        assert cell.skip_matrix.shape == (64, 64)
        assert cell.weight_input is None
        assert cell.bias_h is not None

    def test_state_and_output_size_equal_reservoir_size(self) -> None:
        """state_size == output_size == reservoir_size (single-state cell)."""
        cell = ES2NCell(reservoir_size=32, feedback_size=3)

        assert cell.state_size == 32
        assert cell.output_size == 32

    def test_skip_matrix_is_orthogonal(self) -> None:
        """The skip matrix O satisfies O O^T == I to float precision."""
        cell = ES2NCell(reservoir_size=128, feedback_size=3, seed=1)
        o = cell.skip_matrix
        err = (o @ o.t() - torch.eye(128)).abs().max().item()
        assert err < 1e-5

    def test_skip_matrix_is_frozen_buffer(self) -> None:
        """O is a non-trainable buffer, not a parameter."""
        cell = ES2NCell(reservoir_size=16, feedback_size=3)
        assert "skip_matrix" in dict(cell.named_buffers())
        assert "skip_matrix" not in dict(cell.named_parameters())

    def test_frozen_by_default(self) -> None:
        """trainable=False (default) freezes every parameter."""
        cell = ES2NCell(reservoir_size=16, feedback_size=3, input_size=2)
        assert all(not p.requires_grad for p in cell.parameters())

    def test_trainable_exposes_grads_but_skip_stays_frozen(self) -> None:
        """trainable=True unfreezes parameters; the skip buffer is unaffected."""
        cell = ES2NCell(reservoir_size=16, feedback_size=3, trainable=True)
        assert all(p.requires_grad for p in cell.parameters())
        # The orthogonal skip is a buffer, so it never carries a grad.
        assert not cell.skip_matrix.requires_grad

    def test_input_size_creates_input_weights(self) -> None:
        """input_size creates the driving-input weight matrix."""
        cell = ES2NCell(reservoir_size=16, feedback_size=3, input_size=7)
        assert cell.input_size == 7
        assert cell.weight_input.shape == (16, 7)

    def test_orthogonal_scaling_scales_singular_values(self) -> None:
        """orthogonal_scaling=g makes every singular value equal g."""
        cell = ES2NCell(reservoir_size=64, feedback_size=3, orthogonal_scaling=2.0, seed=3)
        svals = torch.linalg.svdvals(cell.skip_matrix)
        assert torch.allclose(svals, torch.full_like(svals, 2.0), atol=1e-4)


class TestES2NBetaValidation:
    """The proximity hyperparameter ``beta`` and its live-write contract."""

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, 2.0])
    def test_beta_out_of_range_rejected(self, bad: float) -> None:
        """beta must be in (0, 1]."""
        with pytest.raises(ValueError, match="beta must be in"):
            ES2NCell(reservoir_size=8, feedback_size=2, beta=bad)

    def test_beta_live_write_on_cell(self) -> None:
        """cell.beta is a live knob validated on assignment."""
        cell = ES2NCell(reservoir_size=8, feedback_size=2, beta=0.9)
        cell.beta = 0.2
        assert cell.beta == 0.2
        with pytest.raises(ValueError):
            cell.beta = 3.0

    def test_beta_live_write_through_layer(self) -> None:
        """layer.beta forwards to the cell, with no shadow attribute."""
        layer = ES2NLayer(16, feedback_size=3, beta=0.9, seed=1)
        layer.beta = 0.25
        assert layer.cell.beta == 0.25
        assert layer.beta == 0.25
        assert "beta" not in layer.__dict__  # no silent shadow


class TestES2NUpdateMath:
    """The two-branch ES2N state update."""

    def test_update_matches_explicit_formula(self) -> None:
        """forward realises beta*phi(W_r x + proj) + (1-beta)*O x exactly."""
        cell = ES2NCell(reservoir_size=48, feedback_size=3, beta=0.4, spectral_radius=0.9, seed=7)
        state = torch.randn(2, 48)
        fb = torch.randn(2, 3)

        out, _ = cell([fb], state.clone())

        proj = torch.nn.functional.linear(fb, cell.weight_feedback) + cell.bias_h
        nonlinear = torch.tanh(state @ cell.weight_hh.t() + proj)
        skip = state @ cell.skip_matrix.t()
        expected = 0.4 * nonlinear + 0.6 * skip
        assert torch.allclose(out, expected, atol=1e-6)

    def test_forward_equals_step(self) -> None:
        """forward and the projected step fast path agree (with a driver)."""
        cell = ES2NCell(64, 3, input_size=2, beta=0.3, spectral_radius=0.9, seed=5)
        state = torch.randn(2, 64)
        fb, dr = torch.randn(2, 3), torch.randn(2, 2)

        out_f, _ = cell.forward([fb, dr], state.clone())
        proj = cell.project_inputs([fb, dr])
        out_s, _ = cell.step(proj, state.clone())
        assert torch.equal(out_f, out_s)

    def test_beta_one_reduces_to_standard_esn(self) -> None:
        """beta=1 makes ES2N bit-identical to a non-leaky ESN with the same W_r."""
        es2 = ES2NCell(128, 4, beta=1.0, spectral_radius=0.95, seed=99)
        esn = ESNCell(128, 4, spectral_radius=0.95, leak_rate=1.0, seed=99)
        # Subclass + same seed => identical recurrent/feedback/bias draws.
        assert torch.equal(es2.weight_hh, esn.weight_hh)
        assert torch.equal(es2.weight_feedback, esn.weight_feedback)
        assert torch.equal(es2.bias_h, esn.bias_h)

        state, fb = torch.randn(2, 128), torch.randn(2, 4)
        o1, _ = es2([fb], state.clone())
        o2, _ = esn([fb], state.clone())
        assert torch.equal(o1, o2)


class TestES2NEdgeOfStability:
    """The defining edge-of-stability / norm-preservation property."""

    def test_norm_preserved_as_beta_shrinks(self) -> None:
        """With zero input, ||x_t|| / ||x_{t-1}|| -> 1 as beta -> 0.

        As beta -> 0 the map approaches the orthogonal, norm-preserving skip
        x_t ~ O x_{t-1}, so the state norm is conserved step to step — the
        mechanism behind ES2N's long memory.
        """
        prev_ratio = 0.0
        for beta in (0.5, 0.1, 0.01, 0.001):
            cell = ES2NCell(300, 3, beta=beta, spectral_radius=0.9, bias=False, seed=7)
            state = torch.randn(1, 300)
            state = state / state.norm()
            zero_fb = torch.zeros(1, 3)
            ratios = []
            for _ in range(30):
                nxt, _ = cell([zero_fb], state)
                ratios.append((nxt.norm() / state.norm()).item())
                state = nxt
            mean_ratio = sum(ratios) / len(ratios)
            # Monotonically approaches 1 as beta shrinks.
            assert mean_ratio > prev_ratio
            prev_ratio = mean_ratio
        # At the smallest beta the norm is essentially conserved.
        assert prev_ratio > 0.99

    def test_longer_memory_than_plain_esn(self) -> None:
        """Small-beta ES2N conserves norm far better than a plain ESN.

        A plain contractive ESN (spectral_radius < 1) decays the zero-input
        state norm quickly; the orthogonal skip keeps it near-constant.
        """

        def mean_ratio(cell: object) -> float:
            state = torch.randn(1, 200)
            state = state / state.norm()
            zfb = torch.zeros(1, 3)
            rs = []
            for _ in range(40):
                nxt, _ = cell([zfb], state)  # type: ignore[operator]
                rs.append((nxt.norm() / state.norm()).item())
                state = nxt
            return sum(rs) / len(rs)

        es2 = ES2NCell(200, 3, beta=0.05, spectral_radius=0.9, bias=False, seed=7)
        esn = ESNCell(200, 3, spectral_radius=0.9, bias=False, seed=7)
        assert mean_ratio(es2) > mean_ratio(esn) + 0.05


class TestES2NLayer:
    """Sequence-level behaviour of the layer wrapper."""

    def test_forward_shape_feedback_only(self) -> None:
        """(batch, time, feedback) -> (batch, time, reservoir_size)."""
        layer = ES2NLayer(200, feedback_size=3, beta=0.1, seed=1)
        out = layer(torch.randn(4, 50, 3))
        assert out.shape == (4, 50, 200)

    def test_forward_shape_with_driver(self) -> None:
        """A driving input is accepted alongside feedback."""
        layer = ES2NLayer(64, feedback_size=3, input_size=2, beta=0.2, seed=1)
        out = layer(torch.randn(2, 30, 3), torch.randn(2, 30, 2))
        assert out.shape == (2, 30, 64)

    def test_delegates_attributes_to_cell(self) -> None:
        """Unknown attribute reads fall through to the wrapped cell."""
        layer = ES2NLayer(32, feedback_size=3, beta=0.4, seed=1)
        assert layer.reservoir_size == 32
        assert layer.beta == 0.4
        assert layer.skip_matrix.shape == (32, 32)

    def test_seed_reproducibility_includes_skip(self) -> None:
        """Same seed => identical weights AND identical orthogonal skip."""
        a = ES2NLayer(50, feedback_size=3, beta=0.3, spectral_radius=0.9, seed=42)
        b = ES2NLayer(50, feedback_size=3, beta=0.3, spectral_radius=0.9, seed=42)
        assert torch.equal(a.weight_hh, b.weight_hh)
        assert torch.equal(a.weight_feedback, b.weight_feedback)
        assert torch.equal(a.skip_matrix, b.skip_matrix)

    def test_skip_independent_of_recurrent(self) -> None:
        """O is drawn on a stream distinct from W_r (not accidentally equal)."""
        layer = ES2NLayer(64, feedback_size=3, seed=5)
        assert not torch.equal(layer.skip_matrix, layer.weight_hh)

    def test_state_reset_and_shape(self) -> None:
        """State management inherited from BaseReservoirLayer works."""
        layer = ES2NLayer(32, feedback_size=3, beta=0.5, seed=1)
        layer(torch.randn(2, 10, 3))
        assert layer.get_state().shape == (2, 32)
        layer.reset_state()
        assert layer.get_state() is None
