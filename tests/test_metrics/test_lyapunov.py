"""Tests for :mod:`resdag.metrics.lyapunov`.

Covers the pure step<->Lyapunov-time conversions, the canonical-constant tables
and the name-dispatch of :func:`largest_lyapunov`, the native-torch
:func:`reservoir_lyapunov` estimator (stable vs unstable sanity), and — gated on
``tsdynamics`` being importable — a cross-check that the committed canonical
constants reproduce a `tsdynamics` spectrum estimate to two decimals.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import resdag
from resdag.layers import ESNLayer
from resdag.metrics import (
    CANONICAL_LAMBDA_MAX,
    MACKEY_GLASS_LAMBDA,
    largest_lyapunov,
    lyapunov_time,
    lyapunov_to_steps,
    reservoir_lyapunov,
    steps_per_lyapunov_time,
    steps_to_lyapunov,
)

# --------------------------------------------------------------------------- #
# Pure conversions.
# --------------------------------------------------------------------------- #


class TestConversions:
    def test_lyapunov_time_is_reciprocal(self) -> None:
        assert lyapunov_time(0.9) == pytest.approx(1.0 / 0.9)
        assert lyapunov_time(0.9056) == pytest.approx(1.1042, abs=1e-3)

    def test_steps_per_lyapunov_time(self) -> None:
        # One Lyapunov time spans 1 / (dt * lambda) samples.
        assert steps_per_lyapunov_time(0.02, 0.9056) == pytest.approx(1.0 / (0.02 * 0.9056))

    def test_steps_to_lyapunov_headline(self) -> None:
        # The benchmark's reference number: VPT 357 steps at dt 0.02 on Lorenz.
        assert steps_to_lyapunov(357, 0.02, 0.9056) == pytest.approx(6.466, abs=1e-3)

    def test_round_trip_steps_lyapunov(self) -> None:
        lyap = steps_to_lyapunov(357, 0.02, 0.9056)
        assert lyapunov_to_steps(lyap, 0.02, 0.9056) == 357

    def test_lyapunov_to_steps_rounds(self) -> None:
        assert isinstance(lyapunov_to_steps(6.466, 0.02, 0.9056), int)
        assert lyapunov_to_steps(6.466, 0.02, 0.9056) == 357

    @pytest.mark.parametrize(
        "fn, args",
        [
            (lyapunov_time, (0.0,)),
            (lyapunov_time, (-1.0,)),
            (steps_per_lyapunov_time, (0.0, 0.9)),
            (steps_per_lyapunov_time, (0.02, 0.0)),
            (steps_to_lyapunov, (357, -0.02, 0.9)),
            (steps_to_lyapunov, (357, 0.02, -0.9)),
            (lyapunov_to_steps, (6.4, 0.0, 0.9)),
        ],
    )
    def test_non_positive_raises(self, fn, args) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError):
            fn(*args)


# --------------------------------------------------------------------------- #
# Canonical constants + name dispatch.
# --------------------------------------------------------------------------- #


class TestCanonical:
    def test_canonical_table_values(self) -> None:
        assert CANONICAL_LAMBDA_MAX["lorenz"] == pytest.approx(0.9056)
        assert CANONICAL_LAMBDA_MAX["rossler"] == pytest.approx(0.0714)
        assert MACKEY_GLASS_LAMBDA == {17: 0.0086, 23: 0.0097, 30: 0.0071}

    def test_largest_lyapunov_name_paths(self) -> None:
        assert largest_lyapunov("lorenz") == 0.9056
        assert largest_lyapunov("Rossler") == 0.0714  # case-insensitive
        assert largest_lyapunov("mackey_glass", tau=17) == 0.0086
        assert largest_lyapunov("mackey_glass", tau=30) == 0.0071

    def test_name_path_needs_no_tsdynamics(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        # Force the tsdynamics import to fail; the name path must still work.
        import resdag.metrics.lyapunov as lyap

        monkeypatch.setattr(
            lyap, "_import_tsdynamics", lambda: (_ for _ in ()).throw(ImportError())
        )
        assert largest_lyapunov("lorenz") == 0.9056

    def test_unknown_name_raises_keyerror(self) -> None:
        with pytest.raises(KeyError):
            largest_lyapunov("not_a_system")

    def test_unknown_mackey_tau_raises(self) -> None:
        with pytest.raises(KeyError):
            largest_lyapunov("mackey_glass", tau=99)

    def test_series_without_optin_raises_typeerror(self) -> None:
        series = np.zeros(100)
        with pytest.raises(TypeError):
            largest_lyapunov(series)  # missing from_="data"

    def test_missing_tsdynamics_raises_friendly_importerror(self) -> None:
        """The optional-dep guard names the ``resdag[dynamics]`` extra."""
        import importlib.util

        import resdag.metrics.lyapunov as lyap

        if importlib.util.find_spec("tsdynamics") is not None:
            pytest.skip("tsdynamics is installed; the ImportError path is unreachable")
        with pytest.raises(ImportError, match=r"resdag\[dynamics\]"):
            lyap._import_tsdynamics()


# --------------------------------------------------------------------------- #
# reservoir_lyapunov — native torch, always available.
# --------------------------------------------------------------------------- #


class TestReservoirLyapunov:
    def _feedback(self) -> torch.Tensor:
        torch.manual_seed(0)
        return torch.randn(1, 400, 3)

    def test_finite_scalar(self) -> None:
        res = ESNLayer(120, feedback_size=3, spectral_radius=0.9, topology="erdos_renyi", seed=1)
        lam = reservoir_lyapunov(res, self._feedback(), horizon=400, transient=50)
        assert isinstance(lam, float)
        assert math.isfinite(lam)

    def test_stable_more_negative_than_unstable(self) -> None:
        fb = self._feedback()
        stable = ESNLayer(150, feedback_size=3, spectral_radius=0.9, topology="erdos_renyi", seed=7)
        unstable = ESNLayer(
            150, feedback_size=3, spectral_radius=1.3, topology="erdos_renyi", seed=7
        )
        lam_stable = reservoir_lyapunov(stable, fb, horizon=700, transient=100)
        lam_unstable = reservoir_lyapunov(unstable, fb, horizon=700, transient=100)
        # A larger spectral radius pushes the reservoir toward the edge of chaos:
        # its maximal conditional exponent is larger (less negative).
        assert lam_unstable > lam_stable
        # A stable, ESP-satisfying reservoir contracts -> negative exponent.
        assert lam_stable < 0.0

    def test_topk_descending(self) -> None:
        res = ESNLayer(100, feedback_size=3, spectral_radius=0.9, seed=3)
        spectrum = reservoir_lyapunov(res, self._feedback(), horizon=300, k=3, transient=50)
        assert isinstance(spectrum, np.ndarray)
        assert spectrum.shape == (3,)
        assert all(spectrum[i] >= spectrum[i + 1] for i in range(2))

    def test_accepts_model(self) -> None:
        import pytorch_symbolic as ps

        from resdag import CGReadoutLayer, ESNModel

        inp = ps.Input((400, 3))
        r = ESNLayer(80, feedback_size=3, spectral_radius=0.95, seed=5)(inp)
        out = CGReadoutLayer(80, 3, name="output")(r)
        model = ESNModel(inp, out)
        lam = reservoir_lyapunov(model, self._feedback(), horizon=300)
        assert math.isfinite(lam)

    def test_reproducible_with_torch_seed(self) -> None:
        res = ESNLayer(100, feedback_size=3, spectral_radius=0.9, seed=11)
        fb = self._feedback()
        torch.manual_seed(42)
        a = reservoir_lyapunov(res, fb, horizon=300, transient=50)
        torch.manual_seed(42)
        b = reservoir_lyapunov(res, fb, horizon=300, transient=50)
        assert a == pytest.approx(b, abs=1e-9)

    def test_self_deterministic_independent_of_global_rng(self) -> None:
        """Default ``seed=0`` makes the estimate independent of the global RNG."""
        res = ESNLayer(60, feedback_size=3, spectral_radius=0.9, seed=2)
        fb = self._feedback()
        torch.rand(37)  # perturb the global RNG between the two calls
        a = reservoir_lyapunov(res, fb, horizon=300, transient=50)
        torch.rand(101)  # perturb it differently
        b = reservoir_lyapunov(res, fb, horizon=300, transient=50)
        assert a == b

    def test_seed_none_uses_global_rng(self) -> None:
        """``seed=None`` opts back into the (non-deterministic) global RNG."""
        res = ESNLayer(60, feedback_size=3, spectral_radius=0.9, seed=2)
        fb = self._feedback()
        val = reservoir_lyapunov(res, fb, horizon=200, transient=50, seed=None)
        assert math.isfinite(val)

    def test_k_out_of_range_raises(self) -> None:
        res = ESNLayer(20, feedback_size=3, seed=1)
        with pytest.raises(ValueError):
            reservoir_lyapunov(res, self._feedback(), horizon=50, k=21)

    def test_non_tanh_activation_raises(self) -> None:
        res = ESNLayer(30, feedback_size=3, activation="relu", seed=1)
        with pytest.raises(NotImplementedError):
            reservoir_lyapunov(res, self._feedback(), horizon=50)

    def test_ngrc_model_has_no_reservoir_lyapunov(self) -> None:
        import pytorch_symbolic as ps

        from resdag import CGReadoutLayer, ESNModel
        from resdag.layers import NGReservoirLayer

        inp = ps.Input((100, 3))
        ng = NGReservoirLayer(input_dim=3, k=2, s=1, p=2)
        feats = ng(inp)
        out = CGReadoutLayer(ng.feature_dim, 3, name="output")(feats)
        model = ESNModel(inp, out)
        with pytest.raises(ValueError):
            reservoir_lyapunov(model, torch.randn(1, 100, 3), horizon=50)


# --------------------------------------------------------------------------- #
# tsdynamics cross-checks (skipped when the optional dep is unavailable).
# --------------------------------------------------------------------------- #


@pytest.fixture
def tsdynamics():
    return pytest.importorskip("tsdynamics")


class TestTsdynamicsCrossCheck:
    def test_lorenz_canonical_matches_spectrum(self, tsdynamics) -> None:
        lam = largest_lyapunov(tsdynamics.systems.Lorenz(), final_time=2000)
        assert lam == pytest.approx(CANONICAL_LAMBDA_MAX["lorenz"], abs=1e-2)

    def test_rossler_canonical_matches_spectrum(self, tsdynamics) -> None:
        lam = largest_lyapunov(tsdynamics.systems.Rossler(), final_time=2000)
        assert lam == pytest.approx(CANONICAL_LAMBDA_MAX["rossler"], abs=1e-2)

    def test_known_lyapunov_agrees_with_constant(self, tsdynamics) -> None:
        # The system's own reference spectrum should also match to two decimals.
        known = tsdynamics.systems.Lorenz().info.known_lyapunov["spectrum"][0]
        assert known == pytest.approx(CANONICAL_LAMBDA_MAX["lorenz"], abs=1e-2)

    def test_from_data_runs_and_is_positive(self, tsdynamics) -> None:
        # A diagnostic (noisy) estimate: we only assert it runs and returns a
        # finite, positive number for a chaotic series -- not that it matches the
        # canonical value (data-based exponents are known to be biased).
        series = resdag.lorenz(4000, dt=0.02, normalize=None, seed=0).squeeze(0).numpy()
        lam = largest_lyapunov(series, dt=0.02, from_="data", method="rosenstein", fit=(2, 15))
        assert math.isfinite(lam)
        assert lam > 0.0
