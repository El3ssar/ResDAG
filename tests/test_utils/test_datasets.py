"""Unit tests for the canonical dataset generators (#241).

Each generator must return a ``(1, T, D)`` tensor, be deterministic under a
fixed seed, and produce finite values. These tests also lock in the documented
feature dimensions, the transient/discard contract, the normalization options,
and the public export surface.
"""

import numpy as np
import pytest
import torch

import resdag as rd
from resdag.utils.data import datasets

# (generator, expected feature dimension D)
GENERATORS = [
    (datasets.lorenz, 3),
    (datasets.rossler, 3),
    (datasets.henon, 2),
    (datasets.mackey_glass, 1),
    (datasets.narma, 2),
    (datasets.sine, 1),
]

GEN_IDS = [g.__name__ for g, _ in GENERATORS]


class TestShape:
    """Every generator returns a ``(1, T, D)`` tensor with the documented D."""

    @pytest.mark.parametrize("gen, dim", GENERATORS, ids=GEN_IDS)
    def test_shape(self, gen, dim) -> None:
        out = gen(256)

        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, 256, dim)

    @pytest.mark.parametrize("gen, dim", GENERATORS, ids=GEN_IDS)
    def test_n_timesteps_is_post_discard(self, gen, dim) -> None:
        """The returned length is exactly ``n_timesteps`` regardless of discard."""
        a = gen(128, discard=0)
        b = gen(128, discard=50)

        assert a.shape[1] == 128
        assert b.shape[1] == 128


class TestDeterminism:
    """Same seed -> identical output; different seed -> different output."""

    @pytest.mark.parametrize("gen, dim", GENERATORS, ids=GEN_IDS)
    def test_same_seed_reproducible(self, gen, dim) -> None:
        a = gen(200, seed=7)
        b = gen(200, seed=7)

        assert torch.equal(a, b)

    @pytest.mark.parametrize(
        "gen, dim",
        # mackey_glass / sine default trajectories are deterministic w.r.t. seed
        # (no stochastic component unless noise/perturb is enabled), so only the
        # seed-driven generators are checked for seed sensitivity here.
        [(datasets.lorenz, 3), (datasets.rossler, 3), (datasets.narma, 2)],
        ids=["lorenz", "rossler", "narma"],
    )
    def test_different_seed_differs(self, gen, dim) -> None:
        a = gen(200, seed=1)
        b = gen(200, seed=2)

        assert not torch.equal(a, b)

    def test_numpy_generator_seed_accepted(self) -> None:
        """A ``numpy.random.Generator`` is a valid seed and is reproducible."""
        a = datasets.lorenz(100, seed=np.random.default_rng(3))
        b = datasets.lorenz(100, seed=np.random.default_rng(3))

        assert torch.equal(a, b)


class TestFiniteOutput:
    """No NaNs or infinities anywhere in the returned trajectory."""

    @pytest.mark.parametrize("gen, dim", GENERATORS, ids=GEN_IDS)
    def test_finite(self, gen, dim) -> None:
        out = gen(500)

        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("gen, dim", GENERATORS, ids=GEN_IDS)
    def test_finite_minmax(self, gen, dim) -> None:
        out = gen(300, normalize="minmax")

        assert torch.isfinite(out).all()


class TestNormalization:
    """The ``normalize`` option behaves as documented."""

    def test_standard_zero_mean_unit_std(self) -> None:
        # float64 to test the normalization math itself, not the float32 cast.
        out = datasets.lorenz(2000, normalize="standard", dtype=torch.float64).squeeze(0)

        assert torch.allclose(out.mean(0), torch.zeros(3, dtype=out.dtype), atol=1e-6)
        assert torch.allclose(out.std(0), torch.ones(3, dtype=out.dtype), atol=1e-3)

    def test_minmax_in_range(self) -> None:
        out = datasets.rossler(2000, normalize="minmax")

        assert out.min().item() >= -1.0 - 1e-9
        assert out.max().item() <= 1.0 + 1e-9

    def test_none_leaves_raw_scale(self) -> None:
        """Unnormalized Lorenz keeps its native (large) amplitude."""
        out = datasets.lorenz(2000, normalize=None)

        assert out.abs().max().item() > 5.0

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError):
            datasets.sine(100, normalize="bogus")  # type: ignore[arg-type]


class TestDtypeAndDevice:
    """``dtype`` / ``device`` are honored on the returned tensor."""

    def test_dtype_override(self) -> None:
        out = datasets.henon(50, dtype=torch.float64)

        assert out.dtype == torch.float64

    def test_default_dtype_float32(self) -> None:
        """Default matches torch's default dtype so the series is model-ready."""
        out = datasets.lorenz(50)

        assert out.dtype == torch.float32


class TestPublicExports:
    """The generators are reachable from every documented entry point."""

    @pytest.mark.parametrize("name", GEN_IDS)
    def test_top_level(self, name) -> None:
        assert hasattr(rd, name)
        assert name in rd.__all__

    @pytest.mark.parametrize("name", GEN_IDS)
    def test_utils_namespace(self, name) -> None:
        assert hasattr(rd.utils, name)

    @pytest.mark.parametrize("name", GEN_IDS)
    def test_utils_data_namespace(self, name) -> None:
        assert hasattr(rd.utils.data, name)

    def test_datasets_alias(self) -> None:
        """``rd.datasets`` aliases the ``resdag.utils.data`` submodule."""
        assert rd.datasets is rd.utils.data
        assert "datasets" in rd.__all__

    def test_at_least_four_canonical_generators(self) -> None:
        """Acceptance criterion: >=4 canonical generators in the public API."""
        exported = [n for n in GEN_IDS if n in rd.__all__]

        assert len(exported) >= 4


class TestNarmaColumns:
    """NARMA returns an input/output pair with non-degenerate channels."""

    def test_two_distinct_channels(self) -> None:
        out = datasets.narma(500).squeeze(0)
        u, y = out[:, 0], out[:, 1]

        assert u.std().item() > 0.0
        assert y.std().item() > 0.0
        assert not torch.equal(u, y)


def _explicit_delay_mackey_glass(
    total: int,
    delay_samples: int,
    *,
    beta: float = 0.2,
    gamma: float = 0.1,
    n: float = 10.0,
    dt: float = 1.0,
    x0: float = 1.2,
) -> np.ndarray:
    """Reference MG integrator with an *explicit* integer delay in samples.

    The delayed term at step ``i`` is ``out[i - delay_samples]`` (or the constant
    history ``x0`` while ``i - delay_samples < 0``). This is the unambiguous
    ground truth used to pin the effective delay of ``mackey_glass``.
    """
    out = np.empty(total, dtype=np.float64)
    x_t = float(x0)
    for i in range(total):
        out[i] = x_t
        idx = i - delay_samples
        x_lag = out[idx] if idx >= 0 else float(x0)
        x_t = x_t + dt * (beta * x_lag / (1.0 + x_lag**n) - gamma * x_t)
    return out


class TestMackeyGlassDelay:
    """Regression: ``tau`` is the *effective* delay in samples (off-by-one fix).

    The old roll-buffer implementation integrated with an effective delay of
    ``lag + 1`` samples, so ``mackey_glass(tau=17)`` silently produced the
    ``tau=18`` system. The flat-buffer rewrite makes the effective delay exactly
    ``lag = round(tau / dt)`` samples, so ``tau=17`` is the canonical chaotic
    MG-17 benchmark. See ``src/resdag/utils/data/datasets.py::mackey_glass``.
    """

    def test_effective_delay_is_tau_not_tau_plus_one(self) -> None:
        """resdag(tau=17) is bit-exact with an explicit delay of 17 samples."""
        n = 2000
        got = (
            datasets.mackey_glass(n, tau=17, discard=0, normalize=None, dtype=torch.float64)
            .numpy()
            .reshape(-1)
        )
        ref_17 = _explicit_delay_mackey_glass(n, 17)
        ref_18 = _explicit_delay_mackey_glass(n, 18)

        # Effective delay is exactly 17 samples (the fix), not 18 (the old bug).
        assert np.max(np.abs(got - ref_17)) == 0.0
        assert np.max(np.abs(got - ref_18)) > 0.5

    def test_effective_delay_honors_dt(self) -> None:
        """With ``dt != 1`` the lag is ``round(tau / dt)`` samples, exactly."""
        n = 500
        # tau=8, dt=0.5 -> lag = 16 samples.
        got = (
            datasets.mackey_glass(n, tau=8, dt=0.5, discard=0, normalize=None, dtype=torch.float64)
            .numpy()
            .reshape(-1)
        )
        ref_16 = _explicit_delay_mackey_glass(n, 16, dt=0.5)

        assert np.max(np.abs(got - ref_16)) == 0.0

    def test_mg17_statistics_pinned(self) -> None:
        """Lock in summary statistics of the corrected MG-17 raw series.

        Values computed from the flat-buffer generator (float64, default params,
        ``discard=1000``). A regression that reintroduces the delay off-by-one
        (or otherwise perturbs the trajectory) will move these.
        """
        x = (
            datasets.mackey_glass(2000, tau=17, normalize=None, dtype=torch.float64)
            .numpy()
            .reshape(-1)
        )

        assert x.mean() == pytest.approx(0.9252045104, abs=1e-9)
        assert x.std() == pytest.approx(0.2331349184, abs=1e-9)
        assert x.min() == pytest.approx(0.3713110461, abs=1e-9)
        assert x.max() == pytest.approx(1.3431508051, abs=1e-9)
        assert x[0] == pytest.approx(0.6664374258, abs=1e-9)

    def test_initial_history_is_x0(self) -> None:
        """The first sample is ``x0`` and the pre-history is constant ``x0``.

        With ``discard=0`` the trajectory starts at ``x0``, and the first ``lag``
        steps use ``x0`` as the delayed term (constant history for ``t <= 0``).
        """
        x0 = 1.2
        x = (
            datasets.mackey_glass(60, tau=17, x0=x0, discard=0, normalize=None, dtype=torch.float64)
            .numpy()
            .reshape(-1)
        )

        assert x[0] == pytest.approx(x0, abs=1e-12)
        # First lag steps match an explicit-delay reference seeded from x0.
        ref = _explicit_delay_mackey_glass(60, 17, x0=x0)
        assert np.max(np.abs(x[:18] - ref[:18])) == 0.0

    def test_seed_none_does_not_advance_torch_rng(self) -> None:
        """``seed=None`` must not perturb torch's global RNG (no randomness used)."""
        torch.manual_seed(1234)
        expected = torch.rand(1).item()

        torch.manual_seed(1234)
        datasets.mackey_glass(100, seed=None, discard=0)
        after = torch.rand(1).item()

        assert after == expected


class TestSineNoise:
    """The optional sine noise is seed-controlled and finite."""

    def test_noise_changes_signal(self) -> None:
        clean = datasets.sine(200, noise=0.0)
        noisy = datasets.sine(200, noise=0.1, seed=0)

        assert not torch.equal(clean, noisy)
        assert torch.isfinite(noisy).all()

    def test_noise_reproducible(self) -> None:
        a = datasets.sine(200, noise=0.1, seed=4)
        b = datasets.sine(200, noise=0.1, seed=4)

        assert torch.equal(a, b)
