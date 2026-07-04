"""Behavioural contract of the ContinuousReservoirCell base.

Pins down the shared continuous-integration machinery used by the
continuous / liquid-state reservoir family (EuSN and successors):

- construction validation (substeps >= 1, known solver),
- the ``vector_field`` -> fixed-step integration contract (Euler / RK4),
- ``forward`` == ``step(project_inputs(inputs), state)`` consistency,
- Euler vs RK4 agree in the small-dt limit (same field),
- ``output_from_state`` default identity,
- the projected fast path (compile-clean ``step``),
- the torchdiffeq ImportError guard (when the optional extra is absent).

Uses a tiny concrete subclass with a *linear* vector field so the integration
math can be checked against a closed form.  Imports are from the concrete
module files: the public ``__init__`` exports are wired separately.
"""

import pytest
import torch

from resdag.layers.cells.continuous_cell import ContinuousReservoirCell


class LinearDecayCell(ContinuousReservoirCell):
    """Minimal concrete cell with field ``f(h, u) = -h + u`` (linear, checkable).

    The exact flow over one interval of length ``dt`` from ``h0`` with a constant
    drive ``u`` is ``h(dt) = u + (h0 - u) * exp(-dt)``, so a fixed-step
    integrator's error against that closed form is a clean convergence probe.
    """

    def vector_field(self, state: torch.Tensor, projected_u: torch.Tensor) -> torch.Tensor:
        return -state + projected_u


class TestConstruction:
    """Constructor validation and stored integration config."""

    def test_defaults(self) -> None:
        """dt=1, substeps=1, solver='euler' by default."""
        cell = LinearDecayCell(reservoir_size=8, feedback_size=3)
        assert cell.dt == 1.0
        assert cell.substeps == 1
        assert cell.solver == "euler"
        assert cell.state_size == 8
        assert cell.output_size == 8

    def test_stores_dt_and_substeps(self) -> None:
        cell = LinearDecayCell(reservoir_size=8, feedback_size=3, dt=0.5, substeps=4)
        assert cell.dt == 0.5
        assert cell.substeps == 4

    @pytest.mark.parametrize("bad", [0, -1, -3])
    def test_substeps_must_be_positive(self, bad: int) -> None:
        with pytest.raises(ValueError, match="substeps"):
            LinearDecayCell(reservoir_size=8, feedback_size=3, substeps=bad)

    def test_unknown_solver_rejected(self) -> None:
        with pytest.raises(ValueError, match="[Uu]nknown solver"):
            LinearDecayCell(reservoir_size=8, feedback_size=3, solver="rk5")

    def test_frozen_by_default(self) -> None:
        """Weights (inherited from ESNCell) are frozen when trainable=False."""
        cell = LinearDecayCell(reservoir_size=8, feedback_size=3, input_size=2)
        assert all(not p.requires_grad for p in cell.parameters())


class TestIntegrationMath:
    """The vector_field -> fixed-step integration contract."""

    def test_single_euler_step_matches_formula(self) -> None:
        """One Euler step: h + dt * f(h, u) with the raw projection as u."""
        cell = LinearDecayCell(reservoir_size=16, feedback_size=3, dt=0.1, seed=1)
        state = torch.randn(2, 16)
        proj = torch.randn(2, 16)
        _, new_state = cell.step(proj, state)
        expected = state + 0.1 * (-state + proj)
        assert torch.allclose(new_state, expected, atol=1e-6)

    def test_substeps_refine_toward_exact_flow(self) -> None:
        """More Euler sub-steps approach the closed-form exp decay."""
        proj = torch.ones(1, 4) * 2.0
        h0 = torch.zeros(1, 4)
        exact = proj + (h0 - proj) * torch.exp(torch.tensor(-1.0))  # dt=1

        errs = []
        for substeps in (1, 4, 16, 64):
            cell = LinearDecayCell(reservoir_size=4, feedback_size=2, dt=1.0, substeps=substeps)
            _, h = cell.step(proj, h0)
            errs.append((h - exact).abs().max().item())
        # Strictly decreasing error as the grid refines.
        assert errs[0] > errs[1] > errs[2] > errs[3]

    def test_euler_vs_rk4_agree_small_dt(self) -> None:
        """Euler and RK4 integrate the same field; they converge as dt -> 0."""
        proj = torch.randn(2, 8)
        h0 = torch.randn(2, 8)
        diffs = []
        for dt in (0.5, 0.05, 0.005):
            ce = LinearDecayCell(reservoir_size=8, feedback_size=2, dt=dt, solver="euler")
            cr = LinearDecayCell(reservoir_size=8, feedback_size=2, dt=dt, solver="rk4")
            _, he = ce.step(proj, h0.clone())
            _, hr = cr.step(proj, h0.clone())
            diffs.append((he - hr).abs().max().item())
        assert diffs[0] > diffs[1] > diffs[2]

    def test_rk4_more_accurate_than_euler(self) -> None:
        """RK4 is closer to the exact flow than Euler at the same dt."""
        proj = torch.ones(1, 4) * 1.5
        h0 = torch.zeros(1, 4)
        exact = proj + (h0 - proj) * torch.exp(torch.tensor(-1.0))
        ce = LinearDecayCell(reservoir_size=4, feedback_size=2, dt=1.0, solver="euler")
        cr = LinearDecayCell(reservoir_size=4, feedback_size=2, dt=1.0, solver="rk4")
        _, he = ce.step(proj, h0)
        _, hr = cr.step(proj, h0)
        assert (hr - exact).abs().max() < (he - exact).abs().max()


class TestForwardStepConsistency:
    """forward == step(project_inputs(inputs), state)."""

    def test_forward_matches_step(self) -> None:
        cell = LinearDecayCell(reservoir_size=16, feedback_size=3, dt=0.2, seed=2)
        state = torch.randn(2, 16)
        fb = torch.randn(2, 3)
        out_f, st_f = cell([fb], state.clone())
        proj = cell.project_inputs([fb])
        out_s, st_s = cell.step(proj, state.clone())
        assert torch.allclose(out_f, out_s, atol=1e-6)
        assert torch.allclose(st_f, st_s, atol=1e-6)

    def test_output_from_state_default_identity(self) -> None:
        """The default output equals the integrated state."""
        cell = LinearDecayCell(reservoir_size=8, feedback_size=2, dt=0.3, seed=3)
        state = torch.randn(1, 8)
        out, new_state = cell.step(torch.randn(1, 8), state)
        assert torch.equal(out, new_state)


class TestCompile:
    """The fixed-step step traces fullgraph=True (compiled forecast engine)."""

    @pytest.mark.parametrize("solver", ["euler", "rk4"])
    def test_step_compiles_fullgraph(self, solver: str) -> None:
        cell = LinearDecayCell(
            reservoir_size=32, feedback_size=3, dt=0.1, substeps=3, solver=solver, seed=1
        )
        state = cell.init_state(2, torch.device("cpu"), torch.float32)
        proj = torch.randn(2, 32)
        compiled = torch.compile(cell.step, fullgraph=True)
        out_c, st_c = compiled(proj, state.clone())
        out_e, st_e = cell.step(proj, state.clone())
        assert torch.allclose(out_c, out_e, atol=1e-5)
        assert torch.allclose(st_c, st_e, atol=1e-5)


class TestTorchdiffeqGuard:
    """The optional adaptive branch fails cleanly without the extra."""

    def test_missing_torchdiffeq_raises_pointing_at_extra(self) -> None:
        pytest.importorskip  # noqa: B018 - marker only; real check below
        try:
            import torchdiffeq  # noqa: F401
        except ImportError:
            cell = LinearDecayCell(reservoir_size=4, feedback_size=2, solver="torchdiffeq")
            state = cell.init_state(1, torch.device("cpu"), torch.float32)
            with pytest.raises(ImportError, match=r"resdag\[ode\]"):
                cell.step(torch.randn(1, 4), state)
        else:
            pytest.skip("torchdiffeq is installed; ImportError guard not exercised")
