"""Behavioural contract of the spiking (LIF) reservoir cell and layer.

Pins down the Liquid State Machine substrate (Maass et al. 2002):

- construction: weight shapes, packed multi-tensor state, freezing,
- the surrogate-gradient spike (exact Heaviside forward, finite non-zero grad
  near threshold, ~0 grad far away),
- the spike reset (subtract / zero) and the low-pass readout trace,
- MEASURED neuron physics: membrane time-constant recovery, the LIF f-I curve
  vs the analytic rate,
- CuBa-LIF (second-order) construction and update,
- forward / step consistency and the compiled-forecast fast path,
- surrogate-gradient SGD trainability (finite grads into W_rec / W_fb),
- the separation property (the LSM hallmark),
- seed reproducibility, driving-input support, and state round-trips.

Imports are from the concrete module files: the public ``__init__`` exports are
wired separately.
"""

import importlib
import math

import pytest
import torch

from resdag.layers.cells.lif_cell import LIFCell
from resdag.layers.cells.spiking_cell import (
    SpikingReservoirCell,
    _pack_state,
    _split_state,
)
from resdag.layers.reservoirs.lif import LIFReservoir


class TestLIFCellConstruction:
    """Construction: weight shapes, packed state, freezing."""

    def test_basic_construction(self) -> None:
        cell = LIFCell(reservoir_size=64, feedback_size=5, tau_mem=15.0, dt=1.0, seed=0)
        assert cell.reservoir_size == 64
        assert cell.feedback_size == 5
        assert cell.tau_mem == 15.0
        assert cell.weight_feedback.shape == (64, 5)
        assert cell.weight_rec.shape == (64, 64)
        assert cell.weight_input is None
        assert cell.bias_h is not None

    def test_beta_derived_from_tau_mem(self) -> None:
        cell = LIFCell(reservoir_size=8, feedback_size=2, tau_mem=20.0, dt=1.0)
        assert cell.beta == pytest.approx(math.exp(-1.0 / 20.0))

    def test_first_order_packs_three_tensors(self) -> None:
        """[V | s | r]: state_size == 3 * reservoir_size."""
        cell = LIFCell(reservoir_size=32, feedback_size=3)
        assert cell.n_state_tensors == 3
        assert cell.state_size == 96
        assert cell.output_size == 32

    def test_cuba_packs_four_tensors(self) -> None:
        """[I | V | s | r]: state_size == 4 * reservoir_size."""
        cell = LIFCell(reservoir_size=32, feedback_size=3, synaptic_current=True)
        assert cell.n_state_tensors == 4
        assert cell.state_size == 128
        assert cell.output_size == 32

    def test_frozen_by_default(self) -> None:
        cell = LIFCell(reservoir_size=16, feedback_size=3, input_size=2)
        assert all(not p.requires_grad for p in cell.parameters())

    def test_trainable_unfreezes(self) -> None:
        cell = LIFCell(reservoir_size=16, feedback_size=3, trainable=True)
        assert all(p.requires_grad for p in cell.parameters())

    def test_input_size_creates_input_weights(self) -> None:
        cell = LIFCell(reservoir_size=16, feedback_size=3, input_size=7)
        assert cell.input_size == 7
        assert cell.weight_input.shape == (16, 7)

    @pytest.mark.parametrize("bad_kwargs", [{"tau_mem": 0.0}, {"tau_syn": -1.0}, {"dt": 0.0}])
    def test_invalid_time_constants_rejected(self, bad_kwargs: dict) -> None:
        with pytest.raises(ValueError):
            LIFCell(reservoir_size=8, feedback_size=2, **bad_kwargs)

    def test_invalid_reset_and_output_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown reset"):
            LIFCell(reservoir_size=8, feedback_size=2, reset="bogus")
        with pytest.raises(ValueError, match="Unknown output"):
            LIFCell(reservoir_size=8, feedback_size=2, output="bogus")


class TestPackedState:
    """The packed multi-tensor-state convention (shared with the gated cell)."""

    def test_split_pack_roundtrip_three_state(self) -> None:
        h = 8
        v, s, r = (torch.randn(4, h) for _ in range(3))
        packed = _pack_state(v, s, r)
        assert packed.shape == (4, 3 * h)
        v2, s2, r2 = _split_state(packed, h)
        assert torch.equal(v, v2) and torch.equal(s, s2) and torch.equal(r, r2)


class TestSurrogateSpike:
    """The straight-through surrogate-gradient spike."""

    def test_forward_is_exact_heaviside(self) -> None:
        cell = LIFCell(reservoir_size=64, feedback_size=3, v_threshold=1.0, seed=0)
        v = torch.linspace(-2, 3, 64).unsqueeze(0)
        spikes = cell.spike(v)
        hard = (v - 1.0 > 0).to(v.dtype)
        # Forward value is EXACTLY the hard threshold (STE identity), and binary.
        assert torch.equal(spikes, hard)
        assert set(spikes.unique().tolist()) <= {0.0, 1.0}

    def test_gradient_nonzero_near_threshold_and_vanishes_far_away(self) -> None:
        cell = LIFCell(reservoir_size=1, feedback_size=1, v_threshold=1.0, surrogate_beta=10.0)
        v = torch.tensor([[0.99, 1.01, 6.0, -4.0]], requires_grad=True)
        cell.spike(v).sum().backward()
        grad = v.grad[0]
        # Near threshold: finite, non-zero surrogate gradient.
        assert torch.isfinite(grad).all()
        assert grad[0] > 1e-2 and grad[1] > 1e-2
        # Far from threshold: gradient decays toward zero.
        assert grad[2] < 1e-3 and grad[3] < 1e-3

    def test_surrogate_beta_validation_and_live_write(self) -> None:
        cell = LIFCell(reservoir_size=4, feedback_size=2, surrogate_beta=5.0)
        assert cell.surrogate_beta == 5.0
        cell.surrogate_beta = 20.0
        assert cell.surrogate_beta == 20.0
        with pytest.raises(ValueError):
            cell.surrogate_beta = 0.0


class TestResetAndTrace:
    """The spike reset (subtract / zero) and the low-pass readout trace."""

    def test_subtract_reset(self) -> None:
        cell = LIFCell(reservoir_size=4, feedback_size=1, v_threshold=1.0, reset="subtract")
        v = torch.tensor([[1.5, 0.2, 2.0, 0.9]])
        spikes = (v - 1.0 > 0).float()
        reset = cell.apply_reset(v, spikes)
        expected = v - 1.0 * spikes
        assert torch.equal(reset, expected)

    def test_zero_reset(self) -> None:
        cell = LIFCell(reservoir_size=4, feedback_size=1, v_threshold=1.0, reset="zero")
        v = torch.tensor([[1.5, 0.2, 2.0, 0.9]])
        spikes = (v - 1.0 > 0).float()
        reset = cell.apply_reset(v, spikes)
        assert torch.equal(reset, v * (1.0 - spikes))

    def test_trace_low_pass(self) -> None:
        cell = LIFCell(reservoir_size=3, feedback_size=1, trace_decay=0.5)
        trace = torch.tensor([[1.0, 0.0, 2.0]])
        spikes = torch.tensor([[1.0, 1.0, 0.0]])
        new = cell.update_trace(trace, spikes)
        assert torch.allclose(new, 0.5 * trace + spikes)

    def test_trace_decay_validation(self) -> None:
        with pytest.raises(ValueError):
            LIFCell(reservoir_size=4, feedback_size=1, trace_decay=1.0)
        with pytest.raises(ValueError):
            LIFCell(reservoir_size=4, feedback_size=1, trace_decay=-0.1)

    def test_output_modes(self) -> None:
        """spikes ∈ {0,1}; trace ≥ 0; voltage is the real-valued membrane V.

        Same ``seed`` across modes ⇒ identical weights, so the three outputs are
        directly comparable.  The voltage check guards the packing bug where the
        cell returned the (binary) spike train instead of ``V``: voltage must be
        genuinely real-valued and must NOT coincide with the spikes output.
        """
        # Drive a strong constant input over several steps at a low threshold so
        # the liquid actually fires (a single unit-scale step leaves it silent).
        fb = torch.ones(2, 3) * 6.0
        outs: dict[str, torch.Tensor] = {}
        for mode in ("spikes", "trace", "voltage"):
            cell = LIFCell(reservoir_size=16, feedback_size=3, output=mode, v_threshold=0.3, seed=0)
            st = cell.init_state(2, torch.device("cpu"), torch.float32)
            out = torch.empty(0)
            for _ in range(8):
                out, st = cell([fb], st)
            assert out.shape == (2, 16)
            outs[mode] = out

        # spikes are binary; at least one neuron actually fired under this drive.
        assert set(outs["spikes"].unique().tolist()) <= {0.0, 1.0}
        assert bool((outs["spikes"] == 1.0).any())
        # trace is a non-negative low-pass of the spikes.
        assert bool((outs["trace"] >= 0).all())
        # voltage is the membrane potential V, not the spike train: real-valued
        # (not a subset of {0,1}) and distinct from the spikes output.
        assert not (set(outs["voltage"].unique().tolist()) <= {0.0, 1.0})
        assert not torch.equal(outs["voltage"], outs["spikes"])


class TestMembranePhysics:
    """MEASURED neuron dynamics vs theory (the maintainer's check-against-data bar)."""

    def test_membrane_time_constant_recovered(self) -> None:
        """Zero input, sub-threshold: V decays as exp(-t/tau_mem); recover tau_mem."""
        tau_mem = 20.0
        cell = LIFCell(
            reservoir_size=1,
            feedback_size=1,
            tau_mem=tau_mem,
            dt=1.0,
            v_threshold=1e9,  # never spike
            bias=False,
            seed=0,
        )
        with torch.no_grad():
            cell.weight_rec.zero_()
            cell.weight_feedback.zero_()
        v0 = 0.5
        state = torch.zeros(1, cell.state_size)
        state[0, 0] = v0  # V is packed tensor 0 ([V | s | r])
        fb = torch.zeros(1, 1)
        vs = []
        for _ in range(60):
            _, state = cell([fb], state)
            vs.append(state[0, 0].item())
        vs_t = torch.tensor(vs)
        t = torch.arange(1, 61, dtype=torch.float32)
        # ln(V_t / V0) = -t / tau  (least squares slope through the log-decay).
        lnv = torch.log(vs_t / v0)
        slope = (t * lnv).sum() / (t * t).sum()
        tau_recovered = -1.0 / slope.item()
        assert tau_recovered == pytest.approx(tau_mem, rel=0.05)

    def test_fi_curve_matches_analytic_and_is_monotone(self) -> None:
        """Constant injected current: no spikes below rheobase, monotone f(I) above,
        matching the analytic discrete-LIF rate."""
        tau_mem, vth, dt = 20.0, 1.0, 1.0
        beta = math.exp(-dt / tau_mem)

        def firing_rate(current: float, nsteps: int = 8000) -> float:
            cell = LIFCell(
                reservoir_size=1,
                feedback_size=1,
                tau_mem=tau_mem,
                dt=dt,
                v_threshold=vth,
                reset="subtract",
                bias=False,
                output="spikes",
                seed=0,
            )
            with torch.no_grad():
                cell.weight_rec.zero_()
                cell.weight_feedback.fill_(1.0)  # drive == current
            state = cell.init_state(1, torch.device("cpu"), torch.float32)
            fb = torch.tensor([[current]])
            n = 0.0
            for _ in range(nsteps):
                out, state = cell([fb], state)
                n += out[0, 0].item()
            return n / (nsteps * dt)

        def analytic_rate(current: float) -> float:
            # V_t = current (1 - beta^t); time to threshold; rate = 1 / t_th.
            if current <= vth:
                return 0.0
            t_th = math.log(1 - vth / current) / math.log(beta)
            return 1.0 / (t_th * dt)

        # No spikes at or below rheobase (I <= v_threshold).
        assert firing_rate(0.5) == 0.0
        assert firing_rate(1.0) == 0.0

        currents = [1.2, 1.5, 2.0, 3.0]
        measured = [firing_rate(c) for c in currents]
        # Monotone increasing above rheobase.
        assert all(measured[i] < measured[i + 1] for i in range(len(measured) - 1))
        # Each matches the analytic LIF rate closely.
        for c, m in zip(currents, measured):
            assert m == pytest.approx(analytic_rate(c), rel=0.05)


class TestCuBaLIF:
    """The second-order current-based LIF variant."""

    def test_cuba_update_matches_explicit_formula(self) -> None:
        cell = LIFCell(
            reservoir_size=16,
            feedback_size=3,
            synaptic_current=True,
            tau_mem=15.0,
            tau_syn=8.0,
            v_threshold=0.5,
            reset="subtract",
            seed=1,
        )
        state = cell.init_state(2, torch.device("cpu"), torch.float32)
        fb = torch.randn(2, 3)
        _, new_state = cell([fb], state.clone())

        # Reconstruct: [I | V | s | r] packed.
        i_prev, v_prev, s_prev, _ = _split_state(state, 16)
        drive = cell.project_inputs([fb]) + torch.nn.functional.linear(s_prev, cell.weight_rec)
        i_new = cell.alpha * i_prev + drive
        v_pre = cell.beta * v_prev + (1.0 - cell.beta) * i_new
        spikes = (v_pre - 0.5 > 0).float()
        v_new = v_pre - 0.5 * spikes
        i_out, v_out, s_out, _ = _split_state(new_state, 16)
        assert torch.allclose(i_out, i_new, atol=1e-5)
        assert torch.allclose(v_out, v_new, atol=1e-5)
        assert torch.equal(s_out, spikes)


class TestForwardStepConsistency:
    """forward == step(project_inputs(inputs), state) for both variants."""

    @pytest.mark.parametrize("cuba", [False, True])
    def test_forward_matches_step(self, cuba: bool) -> None:
        cell = LIFCell(reservoir_size=32, feedback_size=3, synaptic_current=cuba, seed=2)
        state = cell.init_state(2, torch.device("cpu"), torch.float32)
        fb = torch.randn(2, 3)
        out_f, st_f = cell([fb], state.clone())
        proj = cell.project_inputs([fb])
        out_s, st_s = cell.step(proj, state.clone())
        assert torch.equal(out_f, out_s)
        assert torch.equal(st_f, st_s)


class TestCompile:
    """``torch.compile(cell.step, fullgraph=True)`` traces clean and matches eager."""

    @pytest.mark.parametrize("cuba", [False, True])
    def test_step_compiles_fullgraph_matches_eager(self, cuba: bool) -> None:
        cell = LIFCell(reservoir_size=32, feedback_size=3, synaptic_current=cuba, seed=1)
        state = cell.init_state(2, torch.device("cpu"), torch.float32)
        proj = cell.project_inputs([torch.randn(2, 3)])
        eager_out, eager_state = cell.step(proj, state.clone())
        compiled = torch.compile(cell.step, fullgraph=True)
        c_out, c_state = compiled(proj, state.clone())
        # allclose, not equal: the CPU inductor backend reorders the F.linear
        # reductions vs eager, so the compiled step can differ in the last ULP
        # (matches the tolerance the EuSN/continuous compile tests use).
        assert torch.allclose(eager_out, c_out, atol=1e-5)
        assert torch.allclose(eager_state, c_state, atol=1e-5)


class TestSGDTrainability:
    """Surrogate-gradient SGD: finite grads into the liquid, toy loss decreases."""

    def test_finite_gradients_into_liquid_weights(self) -> None:
        layer = LIFReservoir(64, feedback_size=1, tau_mem=10.0, trainable=True, seed=0)
        x = torch.randn(1, 40, 1)
        layer.reset_state()
        traces = layer(x)
        loss = traces.pow(2).mean()
        loss.backward()
        for name in ("weight_rec", "weight_feedback"):
            grad = getattr(layer.cell, name).grad
            assert grad is not None
            assert torch.isfinite(grad).all()
        # The surrogate genuinely reaches the recurrent weights (non-zero grad).
        assert layer.cell.weight_rec.grad.norm() > 0.0

    def test_toy_driven_regression_loss_decreases(self) -> None:
        torch.manual_seed(0)
        layer = LIFReservoir(
            80,
            feedback_size=1,
            tau_mem=8.0,
            trainable=True,
            spectral_radius=0.6,
            trace_decay=0.85,
            seed=0,
        )
        head = torch.nn.Linear(80, 1)
        opt = torch.optim.Adam(list(layer.parameters()) + list(head.parameters()), lr=5e-3)
        x = torch.randn(1, 60, 1)
        target = torch.tanh(2.0 * x.roll(3, dims=1))
        first = last = None
        for step in range(60):
            layer.reset_state()
            opt.zero_grad()
            pred = head(layer(x))
            loss = (pred - target).pow(2).mean()
            loss.backward()
            assert torch.isfinite(layer.cell.weight_rec.grad).all()
            opt.step()
            if step == 0:
                first = loss.item()
            last = loss.item()
        assert last < first


class TestSeparationProperty:
    """The LSM hallmark: same input -> same state; distinct inputs -> distinct states."""

    def _final_state(self, layer: LIFReservoir, x: torch.Tensor) -> torch.Tensor:
        layer.reset_state()
        return layer(x)[:, -1, :]

    def test_same_input_same_state(self) -> None:
        layer = LIFReservoir(128, feedback_size=1, spectral_radius=0.9, seed=1)
        x = torch.randn(1, 60, 1)
        s1 = self._final_state(layer, x.clone())
        s2 = self._final_state(layer, x.clone())
        assert torch.equal(s1, s2)

    def test_state_distance_grows_with_input_distance(self) -> None:
        """Perturbing the input along a fixed direction by growing eps grows the
        liquid-state distance monotonically (separation)."""
        layer = LIFReservoir(
            256,
            feedback_size=1,
            tau_mem=12.0,
            spectral_radius=1.0,
            trace_decay=0.9,
            v_threshold=0.5,
            bias_scaling=1.0,
            feedback_initializer=("normal", {"input_scaling": 3.0}),
            seed=1,
        )
        torch.manual_seed(0)
        t = 100
        base = torch.randn(1, t, 1)
        direction = torch.randn(1, t, 1)
        direction = direction / direction.norm()
        s_base = self._final_state(layer, base)
        dists = [
            (self._final_state(layer, base + eps * direction) - s_base).norm().item()
            for eps in (0.2, 0.5, 1.0, 2.0, 4.0)
        ]
        assert all(dists[i] < dists[i + 1] for i in range(len(dists) - 1))


class TestSeedReproducibility:
    """Same seed -> identical weights."""

    def test_same_seed_identical_weights(self) -> None:
        a = LIFCell(reservoir_size=32, feedback_size=3, spectral_radius=0.9, seed=42)
        b = LIFCell(reservoir_size=32, feedback_size=3, spectral_radius=0.9, seed=42)
        assert torch.equal(a.weight_rec, b.weight_rec)
        assert torch.equal(a.weight_feedback, b.weight_feedback)
        assert torch.equal(a.bias_h, b.bias_h)


class TestLIFReservoirLayer:
    """The public layer wrapper."""

    def test_forward_shape(self) -> None:
        layer = LIFReservoir(64, feedback_size=3, seed=3)
        traces = layer(torch.randn(4, 20, 3))
        assert traces.shape == (4, 20, 64)

    def test_driving_input(self) -> None:
        layer = LIFReservoir(48, feedback_size=3, input_size=2, seed=1)
        traces = layer(torch.randn(2, 15, 3), torch.randn(2, 15, 2))
        assert traces.shape == (2, 15, 48)

    def test_attribute_delegation(self) -> None:
        layer = LIFReservoir(32, feedback_size=3, tau_mem=12.0, seed=1)
        assert layer.reservoir_size == 32
        assert layer.tau_mem == 12.0
        assert layer.weight_rec.shape == (32, 32)

    def test_live_write_surrogate_beta_and_trace_decay_through_layer(self) -> None:
        layer = LIFReservoir(16, feedback_size=3, surrogate_beta=5.0, trace_decay=0.5, seed=1)
        layer.surrogate_beta = 15.0
        layer.trace_decay = 0.8
        assert layer.cell.surrogate_beta == 15.0
        assert layer.cell.trace_decay == 0.8
        # No silent shadow attribute on the layer.
        assert "surrogate_beta" not in layer.__dict__
        assert "trace_decay" not in layer.__dict__

    def test_state_roundtrip(self) -> None:
        layer = LIFReservoir(16, feedback_size=3, seed=1)
        assert layer.get_state() is None
        layer.reset_state(batch_size=2)
        state = layer.get_state()
        assert state is not None and state.shape == (2, 48)  # 3 * 16 packed
        # set_state round-trips and validates.
        layer.set_state(state)
        assert torch.equal(layer.get_state(), state)
        with pytest.raises(ValueError):
            layer.set_state(torch.randn(2, 47))

    def test_cuba_layer_state_width(self) -> None:
        layer = LIFReservoir(16, feedback_size=3, synaptic_current=True, seed=1)
        layer.reset_state(batch_size=2)
        assert layer.get_state().shape == (2, 64)  # 4 * 16 packed


class TestSpikingBaseIsAbstract:
    """SpikingReservoirCell cannot be instantiated directly."""

    def test_abstract(self) -> None:
        with pytest.raises(TypeError):
            SpikingReservoirCell(reservoir_size=8)  # type: ignore[abstract]


class TestPublicExports:
    """The spiking symbols are wired into ``__all__`` / ``__dir__`` at every level.

    These introspect the package namespaces via :func:`importlib.import_module`
    rather than ``import resdag`` on purpose: a static ``import resdag`` would tie
    this layers test file to the *whole* package in the affected-tests import
    graph (breaking the selector's subsystem-isolation invariants). Runtime
    imports keep the file's static footprint to the LIF modules it actually
    unit-tests; a change to any package ``__init__`` still reaches the top-level
    ``resdag/__init__`` (a broad-impact file that fans out to the full suite).
    """

    def test_exported_at_every_level(self) -> None:
        resdag = importlib.import_module("resdag")
        layers = importlib.import_module("resdag.layers")
        cells = importlib.import_module("resdag.layers.cells")
        reservoirs = importlib.import_module("resdag.layers.reservoirs")
        models = importlib.import_module("resdag.models")

        # Cells package.
        for name in ("SpikingReservoirCell", "LIFCell"):
            assert name in cells.__all__ and hasattr(cells, name)
        # Reservoirs package.
        assert "LIFReservoir" in reservoirs.__all__ and hasattr(reservoirs, "LIFReservoir")
        # layers aggregate.
        for name in ("SpikingReservoirCell", "LIFCell", "LIFReservoir"):
            assert name in layers.__all__ and hasattr(layers, name)
        # models package.
        assert "lsm" in models.__all__ and hasattr(models, "lsm")
        # Top-level resdag.
        for name in ("SpikingReservoirCell", "LIFCell", "LIFReservoir", "lsm"):
            assert name in resdag.__all__ and hasattr(resdag, name)

    def test_dir_equals_all(self) -> None:
        modules = [
            importlib.import_module(name)
            for name in (
                "resdag.layers.cells",
                "resdag.layers.reservoirs",
                "resdag.layers",
                "resdag.models",
                "resdag",
            )
        ]
        for module in modules:
            assert set(module.__dir__()) == set(module.__all__)
