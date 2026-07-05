"""Behavioural contract of the EuSN cell and layer.

Pins down the Euler State Network (Gallicchio, arXiv:2203.09382):

- construction: weight shapes, epsilon/gamma, freezing,
- the exact EuSN update math (tanh((W_r - W_r^T - gamma*I) h + u), one Euler step),
- the antisymmetric recurrence and the near-norm-preservation / stability
  property (antisymmetric vs a non-antisymmetric Euler step),
- the diffusion term gamma (live knob) and its damping effect,
- forward / step consistency (compiled-forecast fast path),
- skew-operator spectral scaling (spectral_radius targets W_r - W_r^T),
- seed reproducibility and driving-input support.

Imports are from the concrete module files: the public ``__init__`` exports are
wired separately.
"""

import pytest
import torch

from resdag.init.topology import estimate_spectral_radius
from resdag.layers.cells.eusn_cell import EuSNCell
from resdag.layers.reservoirs.eusn import EuSNLayer


class TestEuSNCellConstruction:
    """Construction: weight shapes, epsilon/gamma, freezing."""

    def test_basic_construction(self) -> None:
        cell = EuSNCell(reservoir_size=64, feedback_size=5, epsilon=0.02, gamma=0.03)
        assert cell.reservoir_size == 64
        assert cell.feedback_size == 5
        assert cell.epsilon == 0.02
        assert cell.dt == 0.02  # epsilon is the base dt
        assert cell.gamma == 0.03
        assert cell.weight_feedback.shape == (64, 5)
        assert cell.weight_hh.shape == (64, 64)
        assert cell.weight_input is None
        assert cell.bias_h is not None

    def test_state_and_output_size(self) -> None:
        cell = EuSNCell(reservoir_size=32, feedback_size=3)
        assert cell.state_size == 32
        assert cell.output_size == 32

    def test_frozen_by_default(self) -> None:
        cell = EuSNCell(reservoir_size=16, feedback_size=3, input_size=2)
        assert all(not p.requires_grad for p in cell.parameters())

    def test_trainable_unfreezes(self) -> None:
        cell = EuSNCell(reservoir_size=16, feedback_size=3, trainable=True)
        assert all(p.requires_grad for p in cell.parameters())

    def test_input_size_creates_input_weights(self) -> None:
        cell = EuSNCell(reservoir_size=16, feedback_size=3, input_size=7)
        assert cell.input_size == 7
        assert cell.weight_input.shape == (16, 7)


class TestGammaValidation:
    """The diffusion coefficient ``gamma`` and its live-write contract."""

    @pytest.mark.parametrize("bad", [-0.1, -1.0])
    def test_negative_gamma_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match="gamma must be non-negative"):
            EuSNCell(reservoir_size=8, feedback_size=2, gamma=bad)

    def test_gamma_zero_allowed(self) -> None:
        cell = EuSNCell(reservoir_size=8, feedback_size=2, gamma=0.0)
        assert cell.gamma == 0.0

    def test_gamma_live_write_on_cell(self) -> None:
        cell = EuSNCell(reservoir_size=8, feedback_size=2, gamma=0.01)
        cell.gamma = 0.05
        assert cell.gamma == 0.05
        with pytest.raises(ValueError):
            cell.gamma = -1.0

    def test_gamma_live_write_through_layer(self) -> None:
        layer = EuSNLayer(16, feedback_size=3, gamma=0.01, seed=1)
        layer.gamma = 0.07
        assert layer.cell.gamma == 0.07
        assert layer.gamma == 0.07
        assert "gamma" not in layer.__dict__  # no silent shadow


class TestEuSNUpdateMath:
    """The exact EuSN forward-Euler update."""

    def test_update_matches_explicit_formula(self) -> None:
        """forward realises h + eps * tanh((W_r - W_r^T - gamma I) h + u)."""
        cell = EuSNCell(reservoir_size=48, feedback_size=3, epsilon=0.05, gamma=0.02, seed=7)
        state = torch.randn(2, 48)
        fb = torch.randn(2, 3)
        out, new_state = cell([fb], state.clone())

        w = cell.weight_hh
        u = cell.project_inputs([fb])
        skew = w - w.t()
        pre = state @ skew.t() - cell.gamma * state + u
        expected = state + cell.epsilon * torch.tanh(pre)
        assert torch.allclose(new_state, expected, atol=1e-5)
        assert torch.allclose(out, new_state, atol=1e-6)

    def test_gamma_zero_reduces_to_undamped(self) -> None:
        """gamma=0 drops the diffusion term entirely."""
        cell = EuSNCell(reservoir_size=32, feedback_size=3, epsilon=0.1, gamma=0.0, seed=3)
        state = torch.randn(2, 32)
        fb = torch.randn(2, 3)
        _, new_state = cell([fb], state.clone())
        w = cell.weight_hh
        u = cell.project_inputs([fb])
        pre = state @ (w - w.t()).t() + u  # no -gamma I term
        expected = state + 0.1 * torch.tanh(pre)
        assert torch.allclose(new_state, expected, atol=1e-5)


class TestForwardStepConsistency:
    """forward == step(project_inputs(inputs), state)."""

    def test_forward_matches_step(self) -> None:
        cell = EuSNCell(reservoir_size=32, feedback_size=3, epsilon=0.05, seed=2)
        state = torch.randn(2, 32)
        fb = torch.randn(2, 3)
        out_f, st_f = cell([fb], state.clone())
        proj = cell.project_inputs([fb])
        out_s, st_s = cell.step(proj, state.clone())
        assert torch.allclose(out_f, out_s, atol=1e-6)
        assert torch.allclose(st_f, st_s, atol=1e-6)


class TestStability:
    """Near-norm-preservation of the antisymmetric flow (the EuSN property)."""

    def test_antisymmetric_flow_is_near_norm_preserving(self) -> None:
        """Zero input, small eps, gamma=0: the state norm barely changes per step."""
        cell = EuSNCell(
            reservoir_size=64,
            feedback_size=3,
            epsilon=0.01,
            gamma=0.0,
            activation="identity",  # linear analysis of the antisymmetric flow
            spectral_radius=1.0,
            seed=1,
        )
        state = torch.randn(1, 64)
        state = state / state.norm()
        u = torch.zeros(1, 64)
        ratios = []
        for _ in range(500):
            new_state = cell._integrate(state, u)
            ratios.append((new_state.norm() / state.norm()).item())
            state = new_state
        mean_ratio = sum(ratios) / len(ratios)
        # Near-norm-preserving: each step's ratio hugs 1 (a pure rotation would be
        # exactly 1; the O(eps^2) Euler error gives a tiny drift).
        assert abs(mean_ratio - 1.0) < 1e-2
        # And it does not blow up over the whole run.
        assert torch.isfinite(state).all()
        assert state.norm() < 5.0

    def test_non_antisymmetric_euler_step_blows_up(self) -> None:
        """A raw (non-skew) Euler step of the same weights grows without bound."""
        import types

        # With the *skew* operator scaled to radius 1.0, the antisymmetric flow is
        # bounded, but the raw W_r (radius ~0.4, eigenvalues off the imaginary
        # axis) makes a non-skew forward-Euler step grow without bound — the very
        # instability the antisymmetric construction cures.  Same weights, same
        # eps, only the field differs.
        u = torch.zeros(1, 64)
        common = dict(
            reservoir_size=64,
            feedback_size=3,
            epsilon=0.05,
            gamma=0.0,
            activation="identity",
            spectral_radius=1.0,
            seed=1,
        )

        raw = EuSNCell(**common)

        def raw_field(self, state, projected_u):  # type: ignore[no-untyped-def]
            return state @ self.weight_hh.t() + projected_u

        raw.vector_field = types.MethodType(raw_field, raw)
        st = torch.randn(1, 64)
        st = st / st.norm()
        for _ in range(1000):
            st = raw._integrate(st, u)
        # Grows by many orders of magnitude.
        assert st.norm() > 1e3

        # Contrast: the *antisymmetric* field on the same weights stays bounded.
        anti = EuSNCell(**common)
        st = torch.randn(1, 64)
        st = st / st.norm()
        for _ in range(1000):
            st = anti._integrate(st, u)
        assert torch.isfinite(st).all() and st.norm() < 10.0

    def test_gamma_damps_the_norm(self) -> None:
        """A larger gamma contracts the state more (diffusion damping)."""
        u = torch.zeros(1, 64)

        def final_norm(gamma: float) -> float:
            cell = EuSNCell(
                reservoir_size=64,
                feedback_size=3,
                epsilon=0.05,
                gamma=gamma,
                activation="identity",
                spectral_radius=1.0,
                seed=1,
            )
            state = torch.randn(1, 64)
            state = state / state.norm()
            for _ in range(300):
                state = cell._integrate(state, u)
            return state.norm().item()

        assert final_norm(0.5) < final_norm(0.05)


class TestSkewSpectralScaling:
    """spectral_radius targets the *effective* antisymmetric operator."""

    def test_skew_operator_hits_target_radius(self) -> None:
        """W_r - W_r^T has spectral radius == the requested value."""
        target = 1.2
        cell = EuSNCell(reservoir_size=128, feedback_size=3, spectral_radius=target, seed=1)
        w = cell.weight_hh.detach()
        skew = w - w.t()
        assert estimate_spectral_radius(skew) == pytest.approx(target, abs=5e-3)

    def test_spectral_radius_none_leaves_weights_unscaled(self) -> None:
        """No radius -> no rescale (dense random W_r)."""
        cell = EuSNCell(reservoir_size=32, feedback_size=3, spectral_radius=None, seed=1)
        # Just a sanity check that the cell builds and runs.
        out, _ = cell([torch.randn(1, 3)], cell.init_state(1, torch.device("cpu"), torch.float32))
        assert out.shape == (1, 32)


class TestSolvers:
    """Euler / RK4 both run and agree in the small-eps limit."""

    def test_euler_and_rk4_agree_small_eps(self) -> None:
        common = dict(reservoir_size=48, feedback_size=3, gamma=0.01, seed=7)
        state = torch.randn(2, 48)
        fb = torch.randn(2, 3)
        diffs = []
        for eps in (0.1, 0.01, 0.001):
            ce = EuSNCell(**common, epsilon=eps, solver="euler")
            cr = EuSNCell(**common, epsilon=eps, solver="rk4")
            oe, _ = ce([fb], state.clone())
            orr, _ = cr([fb], state.clone())
            diffs.append((oe - orr).abs().max().item())
        assert diffs[0] > diffs[1] > diffs[2]


class TestSeedReproducibility:
    """Same seed -> identical weights."""

    def test_same_seed_identical_weights(self) -> None:
        a = EuSNCell(reservoir_size=32, feedback_size=3, spectral_radius=1.0, seed=42)
        b = EuSNCell(reservoir_size=32, feedback_size=3, spectral_radius=1.0, seed=42)
        assert torch.equal(a.weight_hh, b.weight_hh)
        assert torch.equal(a.weight_feedback, b.weight_feedback)
        assert torch.equal(a.bias_h, b.bias_h)


class TestEuSNLayer:
    """The public layer wrapper."""

    def test_forward_shape(self) -> None:
        layer = EuSNLayer(64, feedback_size=3, epsilon=0.01, seed=3)
        states = layer(torch.randn(4, 20, 3))
        assert states.shape == (4, 20, 64)

    def test_driving_input(self) -> None:
        layer = EuSNLayer(48, feedback_size=3, input_size=2, epsilon=0.05, seed=1)
        states = layer(torch.randn(2, 15, 3), torch.randn(2, 15, 2))
        assert states.shape == (2, 15, 48)

    def test_attribute_delegation(self) -> None:
        """Layer delegates cell attributes (reservoir_size, weight_hh, ...)."""
        layer = EuSNLayer(32, feedback_size=3, epsilon=0.02, gamma=0.01, seed=1)
        assert layer.reservoir_size == 32
        assert layer.epsilon == 0.02
        assert layer.weight_hh.shape == (32, 32)

    def test_state_management(self) -> None:
        layer = EuSNLayer(16, feedback_size=3, seed=1)
        assert layer.get_state() is None
        layer.reset_state(batch_size=2)
        state = layer.get_state()
        assert state is not None and state.shape == (2, 16)
