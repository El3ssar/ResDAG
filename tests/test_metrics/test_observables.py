"""Tests for :mod:`resdag.metrics.observables`.

Each metric is validated against a **known reference value**, and the ones the
maintainer cares about reproducing (correlation dimension, memory capacity) are
checked **across seeds** so no single lucky draw is being reported:

* ``correlation_dimension`` — Lorenz-63 :math:`\\approx 2.05`, a limit cycle
  :math:`\\approx 1.0`, :math:`d`-dimensional noise :math:`\\approx d`.
* ``memory_capacity`` — ``0 <= MC <= N`` and monotone-ish rising with the
  spectral radius toward the linear bound.
* ``effective_spectral_radius`` — strictly below the nominal weight spectral
  radius for a saturating ``tanh`` reservoir, and lower under a stronger drive.
* ``nrmse`` / ``rmse`` / ``r2`` — vs hand-computed values.

Also covers the ``attractor_dimension`` HPO monitor loss (correlation-dimension
mismatch), which is registered in :mod:`resdag.hpo.losses`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import resdag
from resdag.layers import ESNLayer
from resdag.metrics import (
    correlation_dimension,
    effective_spectral_radius,
    memory_capacity,
    nrmse,
    r2,
    rmse,
)

# --------------------------------------------------------------------------- #
# Pointwise error helpers — vs hand-computed values.
# --------------------------------------------------------------------------- #


class TestPointwiseErrors:
    def test_identical_zero_error(self) -> None:
        x = torch.randn(1, 200, 3)
        assert rmse(x, x) == 0.0
        assert nrmse(x, x) == 0.0
        assert nrmse(x, x, normalization="range") == 0.0

    def test_r2_identical_is_one(self) -> None:
        x = torch.randn(1, 200, 3)
        assert r2(x, x) == pytest.approx(1.0)

    def test_rmse_constant_offset(self) -> None:
        # A constant offset c on every element has RMSE == |c|.
        truth = torch.randn(1, 100, 3)
        pred = truth + 0.5
        assert rmse(pred, truth) == pytest.approx(0.5, abs=1e-6)

    def test_r2_matches_hand_computed(self) -> None:
        rng = np.random.default_rng(0)
        truth_np = rng.standard_normal((1, 300, 3))
        pred_np = truth_np + 0.3 * rng.standard_normal((1, 300, 3))
        truth = torch.tensor(truth_np)
        pred = torch.tensor(pred_np)
        ss_res = float(((pred_np - truth_np) ** 2).sum())
        ss_tot = float(((truth_np - truth_np.mean()) ** 2).sum())
        expected = 1.0 - ss_res / ss_tot
        assert r2(pred, truth) == pytest.approx(expected, rel=1e-9)

    def test_r2_mean_predictor_is_zero(self) -> None:
        # Predicting the grand mean everywhere gives R^2 == 0.
        rng = np.random.default_rng(1)
        truth = torch.tensor(rng.standard_normal((1, 400, 2)))
        pred = torch.full_like(truth, float(truth.mean()))
        assert r2(pred, truth) == pytest.approx(0.0, abs=1e-9)

    def test_nrmse_std_matches_vpt_internal(self) -> None:
        # nrmse(std) must equal the RMS over (batch, time) of the same per-step
        # standardised-residual RMSE that valid_prediction_time thresholds.
        rng = np.random.default_rng(2)
        truth = torch.tensor(rng.standard_normal((1, 250, 3)))
        pred = truth + torch.tensor(rng.standard_normal((1, 250, 3)) * 0.4)
        t = truth.numpy()
        p = pred.numpy()
        scale = t.std(axis=(0, 1), keepdims=True)
        err = np.sqrt((((p - t) / scale) ** 2).mean(axis=2))  # (batch, time)
        expected = float(np.sqrt((err**2).mean()))
        assert nrmse(pred, truth) == pytest.approx(expected, rel=1e-9)

    def test_nrmse_range_hand_computed(self) -> None:
        truth = torch.zeros(1, 100, 1)
        truth[0, :, 0] = torch.linspace(-2.0, 2.0, 100)  # range == 4
        pred = truth + 0.4  # RMSE == 0.4
        # range-normalised nrmse == 0.4 / 4 == 0.1
        assert nrmse(pred, truth, normalization="range") == pytest.approx(0.1, abs=1e-6)

    def test_prefix_alignment(self) -> None:
        truth = torch.randn(1, 100, 3)
        pred = truth[:, :60].clone()
        # Scored only over the overlapping 60-step prefix -> zero error.
        assert rmse(pred, truth) == 0.0
        assert nrmse(pred, truth) == 0.0

    def test_bad_normalization_raises(self) -> None:
        x = torch.randn(1, 20, 3)
        with pytest.raises(ValueError):
            nrmse(x, x, normalization="l2")


# --------------------------------------------------------------------------- #
# Correlation dimension — Grassberger-Procaccia, vs literature references.
# --------------------------------------------------------------------------- #


class TestCorrelationDimension:
    def test_lorenz_reference(self) -> None:
        # Lorenz-63 correlation dimension ~2.05 (accept 1.9-2.15).
        traj = resdag.lorenz(6000, normalize=None, seed=0)
        d2 = correlation_dimension(traj, max_points=2500)
        assert 1.9 <= d2 <= 2.15, d2

    def test_lorenz_reference_across_seeds(self) -> None:
        # Reproduces across independent Lorenz trajectories (no lucky seed).
        vals = []
        for s in range(4):
            traj = resdag.lorenz(6000, normalize=None, seed=s)
            vals.append(correlation_dimension(traj, max_points=2500, seed=s))
        assert all(1.9 <= v <= 2.15 for v in vals), vals
        assert float(np.median(vals)) == pytest.approx(2.05, abs=0.1)

    def test_limit_cycle_is_one(self) -> None:
        # A closed curve (1-D manifold) has correlation dimension ~1.
        t = np.linspace(0, 2 * np.pi, 3000, endpoint=False)
        circle = np.stack([np.cos(t), np.sin(t)], axis=1)
        d2 = correlation_dimension(circle)
        assert 0.85 <= d2 <= 1.15, d2

    def test_noise_fills_embedding_dim(self) -> None:
        # d-dimensional i.i.d. noise has correlation dimension ~d.
        rng = np.random.default_rng(0)
        d2_2 = correlation_dimension(rng.standard_normal((3000, 2)))
        d2_3 = correlation_dimension(rng.standard_normal((3000, 3)))
        assert 1.8 <= d2_2 <= 2.2, d2_2
        assert 2.7 <= d2_3 <= 3.2, d2_3

    def test_deterministic_for_fixed_seed(self) -> None:
        traj = resdag.lorenz(6000, normalize=None, seed=0)
        a = correlation_dimension(traj, max_points=2000, seed=3)
        b = correlation_dimension(traj, max_points=2000, seed=3)
        assert a == b

    def test_accepts_tensor_and_numpy(self) -> None:
        traj_t = resdag.lorenz(4000, normalize=None, seed=0)
        traj_n = traj_t.squeeze(0).numpy()
        a = correlation_dimension(traj_t, max_points=1500, seed=0)
        b = correlation_dimension(traj_n, max_points=1500, seed=0)
        assert a == pytest.approx(b, abs=1e-9)

    def test_squeezes_batch_axis(self) -> None:
        traj = resdag.lorenz(3000, normalize=None, seed=0)  # (1, T, 3)
        d2 = correlation_dimension(traj, max_points=1500)
        assert 1.8 <= d2 <= 2.2

    def test_constant_trajectory_raises(self) -> None:
        with pytest.raises(ValueError):
            correlation_dimension(np.zeros((500, 3)))

    def test_too_few_points_raises(self) -> None:
        with pytest.raises(ValueError):
            correlation_dimension(np.zeros((1, 3)))


# --------------------------------------------------------------------------- #
# Memory capacity — Jaeger 2001, bounded by N and monotone in spectral radius.
# --------------------------------------------------------------------------- #


class TestMemoryCapacity:
    def _reservoir(self, sr: float, n: int = 80, scaling: float = 0.1) -> ESNLayer:
        # Small feedback scaling keeps tanh near its linear regime, where MC
        # rises clearly toward the reservoir-size bound with the spectral radius.
        return ESNLayer(
            n,
            feedback_size=1,
            spectral_radius=sr,
            topology="erdos_renyi",
            bias=False,
            activation="tanh",
            feedback_initializer=("random", {"input_scaling": scaling}),
            seed=0,
        )

    def test_bounded_by_reservoir_size(self) -> None:
        n = 80
        mc = memory_capacity(self._reservoir(0.9, n=n), max_delay=150, n_samples=3000)
        assert 0.0 <= mc <= n, mc

    def test_rises_with_spectral_radius(self) -> None:
        # Median-robust monotone trend across seeds: higher spectral radius ->
        # more linear memory, toward the N bound.
        radii = [0.3, 0.6, 0.9]
        mcs = [memory_capacity(self._reservoir(sr), max_delay=150, n_samples=3000) for sr in radii]
        assert mcs[0] < mcs[1] < mcs[2], mcs
        # And all stay within the N=80 bound.
        assert all(0.0 <= m <= 80.0 for m in mcs)

    def test_rises_toward_linear_bound_across_seeds(self) -> None:
        # For several reservoir seeds, sr=0.95 gives more memory than sr=0.5.
        for seed in range(3):
            low = ESNLayer(
                80,
                feedback_size=1,
                spectral_radius=0.5,
                topology="erdos_renyi",
                bias=False,
                feedback_initializer=("random", {"input_scaling": 0.1}),
                seed=seed,
            )
            high = ESNLayer(
                80,
                feedback_size=1,
                spectral_radius=0.95,
                topology="erdos_renyi",
                bias=False,
                feedback_initializer=("random", {"input_scaling": 0.1}),
                seed=seed,
            )
            mc_low = memory_capacity(low, max_delay=150, n_samples=3000)
            mc_high = memory_capacity(high, max_delay=150, n_samples=3000)
            assert mc_high > mc_low, (seed, mc_low, mc_high)

    def test_linear_reservoir_exceeds_tanh(self) -> None:
        # A linear (identity) reservoir retains strictly more linear memory than
        # a saturating tanh one at the same spectral radius, closer to the bound.
        linear = ESNLayer(
            60,
            feedback_size=1,
            spectral_radius=0.95,
            topology="erdos_renyi",
            bias=False,
            activation="identity",
            feedback_initializer=("random", {"input_scaling": 0.1}),
            seed=0,
        )
        tanh = ESNLayer(
            60,
            feedback_size=1,
            spectral_radius=0.95,
            topology="erdos_renyi",
            bias=False,
            activation="tanh",
            feedback_initializer=("random", {"input_scaling": 0.5}),
            seed=0,
        )
        mc_lin = memory_capacity(linear, max_delay=120, n_samples=3000)
        mc_tanh = memory_capacity(tanh, max_delay=120, n_samples=3000)
        assert mc_lin > mc_tanh, (mc_lin, mc_tanh)
        assert mc_lin <= 60.0

    def test_deterministic(self) -> None:
        res = self._reservoir(0.9)
        a = memory_capacity(res, max_delay=100, n_samples=2000, seed=5)
        b = memory_capacity(res, max_delay=100, n_samples=2000, seed=5)
        assert a == pytest.approx(b, abs=1e-9)

    def test_return_curve(self) -> None:
        res = self._reservoir(0.9)
        mc, curve = memory_capacity(res, max_delay=100, n_samples=2000, return_curve=True)
        assert isinstance(curve, np.ndarray)
        assert curve.shape == (100,)
        assert mc == pytest.approx(float(curve.sum()), abs=1e-9)
        # Forgetting curve: each R^2_k in [0, 1], and it decays (early > late).
        assert np.all((curve >= 0.0) & (curve <= 1.0))
        assert curve[:10].mean() > curve[-10:].mean()

    def test_multi_input_reservoir_raises(self) -> None:
        res = ESNLayer(50, feedback_size=3, spectral_radius=0.9, seed=0)
        with pytest.raises(ValueError, match="feedback_size == 1"):
            memory_capacity(res, max_delay=50, n_samples=500)

    def test_accepts_model(self) -> None:
        import pytorch_symbolic as ps

        from resdag import CGReadoutLayer, ESNModel

        inp = ps.Input((2000, 1))
        r = ESNLayer(60, feedback_size=1, spectral_radius=0.9, bias=False, seed=0)(inp)
        out = CGReadoutLayer(60, 1, name="output")(r)
        model = ESNModel(inp, out)
        mc = memory_capacity(model, max_delay=80, n_samples=1500)
        assert 0.0 <= mc <= 60.0


# --------------------------------------------------------------------------- #
# Effective spectral radius — below the nominal weight radius under tanh.
# --------------------------------------------------------------------------- #


class TestEffectiveSpectralRadius:
    def _drive(self, res: ESNLayer, amp: float = 1.0, seed: int = 0) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        u = (torch.rand(1, 1500, 1, generator=g) * 2 - 1) * amp
        res.reset_state(batch_size=1)
        with torch.no_grad():
            return res(u)

    def test_below_nominal_spectral_radius(self) -> None:
        res = ESNLayer(100, feedback_size=1, spectral_radius=0.9, seed=0)
        states = self._drive(res)
        esr = effective_spectral_radius(res, states)
        # tanh saturation contracts the realized Jacobian below the weight radius.
        assert esr < res.cell.spectral_radius_achieved
        assert esr < 0.9
        assert esr > 0.0

    def test_below_nominal_across_seeds(self) -> None:
        for seed in range(4):
            res = ESNLayer(80, feedback_size=1, spectral_radius=0.9, seed=seed)
            states = self._drive(res, seed=seed)
            esr = effective_spectral_radius(res, states)
            assert esr < res.cell.spectral_radius_achieved, (seed, esr)

    def test_stronger_drive_lowers_esr(self) -> None:
        res = ESNLayer(100, feedback_size=1, spectral_radius=0.9, seed=0)
        weak = effective_spectral_radius(res, self._drive(res, amp=1.0))
        strong = effective_spectral_radius(res, self._drive(res, amp=3.0))
        # A stronger drive saturates tanh more, reducing the realized radius.
        assert strong < weak, (weak, strong)

    def test_accepts_2d_and_3d_states(self) -> None:
        res = ESNLayer(60, feedback_size=1, spectral_radius=0.9, seed=0)
        states3d = self._drive(res)  # (1, T, N)
        states2d = states3d.squeeze(0)  # (T, N)
        a = effective_spectral_radius(res, states3d)
        b = effective_spectral_radius(res, states2d)
        assert a == pytest.approx(b, abs=1e-9)

    def test_wrong_state_dim_raises(self) -> None:
        res = ESNLayer(60, feedback_size=1, spectral_radius=0.9, seed=0)
        with pytest.raises(ValueError):
            effective_spectral_radius(res, torch.randn(100, 40))  # 40 != 60

    def test_non_tanh_raises(self) -> None:
        res = ESNLayer(40, feedback_size=1, activation="relu", seed=0)
        states = torch.randn(1, 100, 40)
        with pytest.raises(NotImplementedError):
            effective_spectral_radius(res, states)

    def test_accepts_model(self) -> None:
        import pytorch_symbolic as ps

        from resdag import CGReadoutLayer, ESNModel

        inp = ps.Input((1500, 1))
        r = ESNLayer(50, feedback_size=1, spectral_radius=0.9, seed=0)(inp)
        out = CGReadoutLayer(50, 1, name="output")(r)
        model = ESNModel(inp, out)
        res = ESNLayer(50, feedback_size=1, spectral_radius=0.9, seed=0)
        states = self._drive(res)
        esr = effective_spectral_radius(model, states)
        assert 0.0 < esr < 0.9


# --------------------------------------------------------------------------- #
# Attractor-fidelity HPO monitor loss.
# --------------------------------------------------------------------------- #


class TestAttractorDimensionLoss:
    def test_registered_in_losses(self) -> None:
        from resdag.hpo import LOSSES, attractor_dimension_error, get_loss

        assert "attractor_dimension" in LOSSES
        assert get_loss("attractor_dimension") is attractor_dimension_error

    def test_identical_is_zero(self) -> None:
        from resdag.hpo import attractor_dimension_error

        lor = resdag.lorenz(6000, normalize=None, seed=0).numpy()
        assert attractor_dimension_error(lor, lor) == pytest.approx(0.0, abs=1e-9)

    def test_same_attractor_small_noise_forecast_large(self) -> None:
        from resdag.hpo import attractor_dimension_error

        lor_a = resdag.lorenz(6000, normalize=None, seed=0).numpy()
        lor_b = resdag.lorenz(6000, normalize=None, seed=1).numpy()
        rng = np.random.default_rng(0)
        noise = rng.standard_normal(lor_a.shape)
        # Right climate (another Lorenz) -> small; wrong climate (noise) -> ~1.
        good = attractor_dimension_error(lor_a, lor_b)
        bad = attractor_dimension_error(lor_a, noise)
        assert good < 0.2, good
        assert bad > 0.6, bad
        assert bad > good

    def test_numpy_signature(self) -> None:
        # The HPO scorer calls losses positionally with numpy arrays + kwargs.
        from resdag.hpo import attractor_dimension_error

        lor = resdag.lorenz(4000, normalize=None, seed=0).numpy()
        val = attractor_dimension_error(lor, lor, max_points=1500, seed=0)
        assert isinstance(val, float)
        assert val >= 0.0


def test_effective_spectral_radius_inverts_leaky_states_before_jacobian() -> None:
    reservoir = ESNLayer(1, feedback_size=1, leak_rate=0.25, seed=0)
    with torch.no_grad():
        reservoir.cell.weight_hh.fill_(0.8)
    # tanh(pre) is [0.5, -0.6] for these two consecutive state transitions.
    states = torch.tensor([[[0.1], [0.2], [0.0]]], dtype=torch.float64)
    expected = ((0.75 + 0.25 * (1 - 0.5**2) * 0.8) + (0.75 + 0.25 * (1 - (-0.6) ** 2) * 0.8)) / 2
    assert effective_spectral_radius(reservoir, states) == pytest.approx(expected, abs=1e-8)


def test_effective_spectral_radius_does_not_join_distinct_trajectories() -> None:
    reservoir = ESNLayer(1, feedback_size=1, leak_rate=0.25, seed=0)
    states = torch.tensor([[[0.1], [0.2]], [[-0.3], [0.0]]])
    separate = sum(effective_spectral_radius(reservoir, row) for row in states) / 2
    assert effective_spectral_radius(reservoir, states) == pytest.approx(separate)


@pytest.mark.parametrize("kwargs", [{"max_states": 0}, {}])
def test_effective_spectral_radius_rejects_empty_sampling(kwargs: dict) -> None:
    reservoir = ESNLayer(1, feedback_size=1, seed=0)
    with pytest.raises(ValueError):
        effective_spectral_radius(reservoir, torch.empty(1, 0, 1), **kwargs)
