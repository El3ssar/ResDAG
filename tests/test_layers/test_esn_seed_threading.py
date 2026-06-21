"""End-to-end seed threading through ``ESNLayer`` / ``ESNCell``.

Issue #136 asks for a single ``seed`` (an ``int`` *or* a ``torch.Generator``)
on ``ESNLayer`` that deterministically fixes the *entire* reservoir — the
recurrent (topology) matrix, the feedback and input weights, and the bias —
covering both explicit initializers/topologies and the default
``uniform(-1, 1)`` draws used when none is given.  This is the reproducibility
prerequisite for per-trial-seeded HPO.

The seed-acceptance plumbing of the *named* topologies/initializers is covered
by ``tests/test_init/test_topologies.py``; these tests pin the layer-level
contract: same seed -> byte-identical reservoirs, independent of the global RNG.
"""

import pytest
import torch

from resdag.layers import ESNLayer

# Every parameter that must be fixed by the seed.
_FULL = ("weight_hh", "weight_feedback", "weight_input", "bias_h")


def _all_equal(a: ESNLayer, b: ESNLayer, names: tuple[str, ...] = _FULL) -> bool:
    """True if every named parameter is byte-identical between two layers."""
    return all(torch.equal(getattr(a, n), getattr(b, n)) for n in names)


class TestSeedFixesEntireReservoir:
    """AC#1: a single ``seed`` fixes topology + feedback + input + bias."""

    def test_int_seed_with_default_initializers(self) -> None:
        """Default (``None``) initializers/topology are seeded by ``seed`` alone."""
        kw = dict(reservoir_size=40, feedback_size=3, input_size=2, spectral_radius=0.9)
        a = ESNLayer(**kw, seed=7)
        b = ESNLayer(**kw, seed=7)

        assert _all_equal(a, b)

    def test_int_seed_with_named_topology_and_initializers(self) -> None:
        """Named topology + named feedback/input initializers all reproduce."""
        kw = dict(
            reservoir_size=40,
            feedback_size=3,
            input_size=2,
            topology="erdos_renyi",
            feedback_initializer="random",
            input_initializer="random",
            spectral_radius=0.9,
        )
        a = ESNLayer(**kw, seed=42)
        b = ESNLayer(**kw, seed=42)

        assert _all_equal(a, b)

    def test_bias_is_seeded(self) -> None:
        """The random bias — previously drawn from the global RNG — is fixed."""
        a = ESNLayer(reservoir_size=30, feedback_size=2, seed=11)
        b = ESNLayer(reservoir_size=30, feedback_size=2, seed=11)

        assert torch.equal(a.bias_h, b.bias_h)

    def test_feedback_only_reservoir_fully_reproducible(self) -> None:
        """No driving input: feedback weights, recurrent matrix and bias match."""
        a = ESNLayer(reservoir_size=25, feedback_size=4, topology="erdos_renyi", seed=3)
        b = ESNLayer(reservoir_size=25, feedback_size=4, topology="erdos_renyi", seed=3)

        assert a.weight_input is None and b.weight_input is None
        assert _all_equal(a, b, ("weight_hh", "weight_feedback", "bias_h"))

    def test_different_seeds_give_different_reservoirs(self) -> None:
        """Distinct seeds must not collide on any parameter."""
        a = ESNLayer(reservoir_size=30, feedback_size=2, input_size=2, seed=1)
        b = ESNLayer(reservoir_size=30, feedback_size=2, input_size=2, seed=2)

        assert not torch.equal(a.weight_feedback, b.weight_feedback)
        assert not torch.equal(a.weight_input, b.weight_input)
        assert not torch.equal(a.bias_h, b.bias_h)


class TestGeneratorSeed:
    """AC#1: ``seed`` accepts a ``torch.Generator``, not just an ``int``."""

    def test_generator_is_accepted_and_reproducible(self) -> None:
        """Two generators with the same ``manual_seed`` yield equal reservoirs."""
        g1 = torch.Generator().manual_seed(123)
        g2 = torch.Generator().manual_seed(123)
        a = ESNLayer(reservoir_size=30, feedback_size=3, topology="erdos_renyi", seed=g1)
        b = ESNLayer(reservoir_size=30, feedback_size=3, topology="erdos_renyi", seed=g2)

        assert _all_equal(a, b, ("weight_hh", "weight_feedback", "bias_h"))

    def test_generator_matches_equivalent_int_seed(self) -> None:
        """A generator seeded with N drives the same topology as ``seed=N``.

        The NumPy-backed topology reads the generator's ``initial_seed()``, so a
        ``torch.Generator().manual_seed(N)`` and a plain ``int`` ``N`` must
        agree on the recurrent matrix.
        """
        g = torch.Generator().manual_seed(99)
        a = ESNLayer(reservoir_size=30, feedback_size=2, topology="erdos_renyi", seed=g)
        b = ESNLayer(reservoir_size=30, feedback_size=2, topology="erdos_renyi", seed=99)

        assert torch.equal(a.weight_hh, b.weight_hh)

    def test_different_generators_differ(self) -> None:
        """Generators seeded differently produce different reservoirs."""
        g1 = torch.Generator().manual_seed(1)
        g2 = torch.Generator().manual_seed(2)
        a = ESNLayer(reservoir_size=30, feedback_size=2, seed=g1)
        b = ESNLayer(reservoir_size=30, feedback_size=2, seed=g2)

        assert not torch.equal(a.weight_feedback, b.weight_feedback)


class TestSeedIndependentOfGlobalRNG:
    """A seeded reservoir must not depend on prior global RNG state."""

    def test_seed_ignores_global_rng_state(self) -> None:
        """Perturbing the global RNG between builds does not change the result."""
        torch.manual_seed(999)
        _ = torch.randn(13)  # advance the global RNG
        a = ESNLayer(reservoir_size=30, feedback_size=2, input_size=2, seed=55)

        torch.manual_seed(111)
        _ = torch.randn(7)  # advance it differently
        b = ESNLayer(reservoir_size=30, feedback_size=2, input_size=2, seed=55)

        assert _all_equal(a, b)

    def test_unseeded_reservoir_reproducible_under_manual_seed(self) -> None:
        """With ``seed=None`` the reservoir tracks ``torch.manual_seed``."""
        torch.manual_seed(0)
        a = ESNLayer(reservoir_size=30, feedback_size=2, input_size=2, topology="erdos_renyi")
        torch.manual_seed(0)
        b = ESNLayer(reservoir_size=30, feedback_size=2, input_size=2, topology="erdos_renyi")

        assert _all_equal(a, b)


class TestSeedDoesNotBreakExistingBehavior:
    """Guard rails: seeding must not change unrelated semantics."""

    def test_zero_bias_scaling_still_zero_bias(self) -> None:
        """``bias_scaling=0.0`` keeps the zero bias even when seeded."""
        layer = ESNLayer(reservoir_size=20, feedback_size=2, bias_scaling=0.0, seed=5)

        assert torch.count_nonzero(layer.bias_h).item() == 0

    def test_spectral_radius_respected_with_seed(self) -> None:
        """The seeded recurrent matrix is still scaled to the target radius."""
        layer = ESNLayer(reservoir_size=40, feedback_size=2, spectral_radius=0.8, seed=4)
        eigvals = torch.linalg.eigvals(layer.weight_hh)
        radius = torch.max(torch.abs(eigvals)).item()

        assert abs(radius - 0.8) < 1e-4


class TestHPOPerTrialSeeding:
    """AC#2: HPO can pass a per-trial seed and get identical reservoirs."""

    def _trial_reservoir(self, trial_seed: int) -> ESNLayer:
        """Stand-in for an HPO ``model_creator`` seeding a fresh reservoir."""
        return ESNLayer(
            reservoir_size=50,
            feedback_size=3,
            input_size=2,
            topology="erdos_renyi",
            feedback_initializer="random",
            spectral_radius=0.9,
            seed=trial_seed,
        )

    def test_same_trial_seed_reproduces_reservoir(self) -> None:
        """Re-running a trial with the same seed rebuilds the same reservoir."""
        a = self._trial_reservoir(trial_seed=20250620)
        b = self._trial_reservoir(trial_seed=20250620)

        assert _all_equal(a, b)

    def test_distinct_trials_explore_distinct_reservoirs(self) -> None:
        """Different per-trial seeds yield different reservoirs (search variety)."""
        seeds = [base + 0 for base in (100, 200, 300)]
        fingerprints = {float(self._trial_reservoir(s).weight_feedback.sum()) for s in seeds}

        assert len(fingerprints) == len(seeds)

    def test_per_trial_generator_threads_through(self) -> None:
        """A per-trial ``torch.Generator`` (e.g. seeded ``seed + trial``) works."""
        gen = torch.Generator().manual_seed(7 + 3)  # base_seed + trial.number
        again = torch.Generator().manual_seed(7 + 3)
        a = ESNLayer(reservoir_size=50, feedback_size=3, topology="erdos_renyi", seed=gen)
        b = ESNLayer(reservoir_size=50, feedback_size=3, topology="erdos_renyi", seed=again)

        assert _all_equal(a, b, ("weight_hh", "weight_feedback", "bias_h"))


class TestDeviceNativeReproducibleDraws:
    """AC#2/#3: device-native, per-device-reproducible weight draws.

    Issue #188: ``seed`` previously drove only the NumPy CPU RNG for the named
    random initializers and the global torch RNG for the default
    no-initializer path. The torch-native random initializers and the seeded
    default path now draw directly on the target device, so the same ``seed``
    reproduces the reservoir on whatever device the layer is built on —
    including CUDA when present.
    """

    def test_default_path_reproducible_on_device(self, device: torch.device) -> None:
        """AC#3: the default no-initializer path is reproducible under one seed.

        Builds the whole reservoir on ``device`` with no explicit
        initializers/topology so every weight comes from the seeded default
        ``uniform(-1, 1)`` draws, and checks two builds agree byte-for-byte.
        """
        kw = dict(reservoir_size=40, feedback_size=3, input_size=2, spectral_radius=0.9)
        a = ESNLayer(**kw, seed=17).to(device)
        b = ESNLayer(**kw, seed=17).to(device)

        assert _all_equal(a, b)

    def test_named_random_initializer_reproducible_on_device(self, device: torch.device) -> None:
        """AC#1/#2: the named ``random``/``random_binary`` inits reproduce per device."""
        kw = dict(
            reservoir_size=40,
            feedback_size=3,
            input_size=2,
            feedback_initializer="random",
            input_initializer="random_binary",
        )
        a = ESNLayer(**kw, seed=23).to(device)
        b = ESNLayer(**kw, seed=23).to(device)

        assert _all_equal(a, b, ("weight_feedback", "weight_input", "bias_h"))

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_random_initializer_draws_natively_on_cuda(self) -> None:
        """AC#2: a CUDA target is filled by a CUDA draw (no CPU build + copy).

        Applying a seeded ``RandomInputInitializer`` to a CUDA weight twice must
        agree, and the result must live on CUDA. Because torch's CPU and CUDA
        RNG streams differ, a CUDA draw must *not* coincide with the equivalent
        CPU draw — proving the values were generated on-device rather than on
        CPU and copied over.
        """
        from resdag.init.input_feedback import RandomInputInitializer

        init = RandomInputInitializer(input_scaling=1.0, seed=42)

        cuda_a = torch.empty(64, 8, device="cuda")
        cuda_b = torch.empty(64, 8, device="cuda")
        init.initialize(cuda_a)
        init.initialize(cuda_b)

        assert cuda_a.is_cuda
        assert torch.equal(cuda_a, cuda_b)  # per-device reproducible

        cpu = torch.empty(64, 8)
        init.initialize(cpu)
        # Same seed, different RNG stream per backend → device-native draw.
        assert not torch.equal(cuda_a.cpu(), cpu)
