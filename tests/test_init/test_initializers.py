"""Input/feedback initializer contracts (resdag.init.input_feedback).

Pins down:

- the initializer registry (``get_input_feedback`` / ``show_input_initializers``
  / ``register_input_feedback``),
- ``FunctionInitializer``: build-style and in-place initializer callables,
- callable initializer specs through ``resolve_initializer``,
- custom initializer objects driving an ESNLayer on every device.
"""

import inspect
import warnings

import numpy as np
import pytest
import torch

from resdag.init.input_feedback import (
    BernoulliInputInitializer,
    BinaryBalancedInitializer,
    ChainOfNeuronsInputInitializer,
    ChebyshevInitializer,
    ChessboardInitializer,
    DendrocycleInputInitializer,
    FunctionInitializer,
    InputFeedbackInitializer,
    NormalInputInitializer,
    OppositeAnchorsInitializer,
    PseudoDiagonalInitializer,
    RandomBinaryInitializer,
    RandomInputInitializer,
    RingWindowInputInitializer,
    UniformInputInitializer,
    get_input_feedback,
    register_input_feedback,
    show_input_initializers,
)
from resdag.init.utils import resolve_initializer
from resdag.layers import ESNLayer


class TestInputFeedbackRegistry:
    """Input/feedback initializer registry."""

    def test_show_initializers_returns_list(self) -> None:
        """``show_input_initializers()`` returns the sorted list of names."""
        names = show_input_initializers()

        assert isinstance(names, list)
        assert len(names) > 0
        assert names == sorted(names)
        # A couple of names we know are registered.
        assert "random" in names
        assert "chebyshev" in names

    def test_show_initializers_with_name_returns_none(self) -> None:
        """When a name is given, only details are printed and ``None`` returned."""
        result = show_input_initializers("chebyshev")
        assert result is None

    def test_show_initializers_unknown_raises(self) -> None:
        """Unknown names raise a ``ValueError`` with the available list."""
        with pytest.raises(ValueError, match="Unknown initializer"):
            show_input_initializers("definitely_not_an_initializer")

    def test_get_input_feedback_known_name(self) -> None:
        """``get_input_feedback`` returns an initializer for a known name."""
        init = get_input_feedback("chebyshev")
        assert init is not None
        assert hasattr(init, "initialize")


class TestChainOfNeuronsInputInference:
    """``chain_of_neurons_input`` infers ``features`` from the weight (issue #194).

    The registry historically defaulted ``features`` to ``None`` while the
    constructor rejected ``None``, so the bare-name path crashed and the show
    helper advertised a default for an effectively required parameter. The fix
    makes ``features`` optional: ``None`` infers the chain count from the
    weight's ``in_features`` at :meth:`initialize` time.
    """

    def test_get_input_feedback_bare_name_does_not_crash(self) -> None:
        """``get_input_feedback("chain_of_neurons_input")`` builds without error."""
        init = get_input_feedback("chain_of_neurons_input")

        assert isinstance(init, ChainOfNeuronsInputInitializer)
        assert init.features is None  # inferred lazily at initialize() time

    def test_bare_name_inits_weight_from_inferred_features(self) -> None:
        """The bare-name initializer fills a weight, inferring chains from columns."""
        init = get_input_feedback("chain_of_neurons_input")
        weight = torch.empty(12, 3)  # 3 columns -> 3 inferred chains
        init.initialize(weight)

        # Input i connects to the first neuron of chain i; everything else zero.
        block_len = 12 // 3
        for i in range(3):
            assert weight[i * block_len, i] == 1.0
        assert int((weight != 0).sum()) == 3

    def test_inferred_matches_explicit_features(self) -> None:
        """Inferring ``features`` yields the same weight as pinning it explicitly."""
        inferred = torch.empty(12, 3)
        explicit = torch.empty(12, 3)
        ChainOfNeuronsInputInitializer().initialize(inferred)
        ChainOfNeuronsInputInitializer(features=3).initialize(explicit)

        assert torch.equal(inferred, explicit)

    def test_explicit_features_mismatch_still_raises(self) -> None:
        """An explicit ``features`` that disagrees with the weight still errors."""
        init = ChainOfNeuronsInputInitializer(features=4)
        with pytest.raises(ValueError, match="must equal 'features'"):
            init.initialize(torch.empty(12, 3))

    def test_features_below_one_rejected(self) -> None:
        """A non-positive explicit ``features`` is rejected at construction."""
        with pytest.raises(ValueError, match="must be >= 1"):
            ChainOfNeuronsInputInitializer(features=0)

    def test_show_helper_does_not_render_required_features(self, capsys) -> None:
        """``show_input_initializers`` no longer presents ``features`` as required.

        With inference, ``None`` is a genuine optional default rather than a
        hidden-required trap, and the table must not render ``<required>``.
        """
        show_input_initializers("chain_of_neurons_input")
        out = capsys.readouterr().out

        assert "features" in out
        assert "<required>" not in out


class TestFunctionInitializer:
    """FunctionInitializer wraps plain callables as initializers."""

    def test_build_style(self) -> None:
        """A build-style (rows, cols) callable fills the weight."""

        def first_neuron_only(rows: int, cols: int, scale: float = 1.0) -> torch.Tensor:
            w = torch.zeros(rows, cols)
            w[0, :] = scale
            return w

        init = FunctionInitializer(first_neuron_only, scale=0.5)
        weight = torch.empty(10, 3)
        init.initialize(weight)

        assert torch.all(weight[0] == 0.5)
        assert torch.all(weight[1:] == 0)

    def test_inplace_style(self) -> None:
        """An in-place torch.nn.init callable is applied to the weight."""
        init = FunctionInitializer(torch.nn.init.xavier_uniform_)
        weight = torch.empty(50, 5)
        init.initialize(weight)

        assert not torch.all(weight == 0)
        assert weight.abs().max() < 1.0  # xavier bound for this shape

    def test_registered_function(self) -> None:
        """A function registered via register_input_feedback resolves by name."""

        @register_input_feedback("_test_constant", value=2.0)
        def constant(rows: int, cols: int, value: float = 1.0) -> torch.Tensor:
            return torch.full((rows, cols), value)

        init = get_input_feedback("_test_constant")
        weight = torch.empty(4, 2)
        init.initialize(weight)
        assert torch.all(weight == 2.0)

        init = get_input_feedback("_test_constant", value=-1.0)
        init.initialize(weight)
        assert torch.all(weight == -1.0)


class TestInitializerResolverCallables:
    """resolve_initializer accepts bare callables and (callable, params) tuples."""

    def test_bare_callable_initializer(self) -> None:
        """A bare callable resolves to FunctionInitializer."""
        resolved = resolve_initializer(torch.nn.init.xavier_uniform_)
        assert isinstance(resolved, FunctionInitializer)

    def test_callable_tuple_initializer(self) -> None:
        """A (callable, params) tuple resolves with the given kwargs."""

        def scaled(rows: int, cols: int, scale: float = 1.0) -> torch.Tensor:
            return torch.full((rows, cols), scale)

        resolved = resolve_initializer((scaled, {"scale": 0.25}))
        weight = torch.empty(6, 3)
        resolved.initialize(weight)
        assert torch.all(weight == 0.25)


# Seeded randomized initializers and the constructor kwargs to exercise them.
# Each entry produces a *fresh* instance so repeated-call determinism is tested
# on a single object, while two-instance agreement is tested across objects.
_SEEDED_INITIALIZERS = [
    pytest.param(RandomInputInitializer, {"input_scaling": 1.0}, id="random"),
    pytest.param(RandomBinaryInitializer, {"input_scaling": 0.5}, id="random_binary"),
    pytest.param(NormalInputInitializer, {"loc": 0.0, "scale": 1.0}, id="normal"),
    pytest.param(UniformInputInitializer, {"low": -1.0, "high": 1.0}, id="uniform"),
    pytest.param(BernoulliInputInitializer, {"p": 0.5}, id="bernoulli"),
    pytest.param(
        PseudoDiagonalInitializer,
        {"input_scaling": 1.0, "binarize": False},
        id="pseudo_diagonal",
    ),
    pytest.param(
        DendrocycleInputInitializer,
        {"c": 0.2, "draw_width": 0.5},
        id="dendrocycle_input",
    ),
]


class TestSeededInitializerDeterminism:
    """Seeded initializers are a pure function of ``(seed, shape)``.

    Regression coverage for the stateful-RNG bug: the RNG must be constructed
    inside ``initialize()`` so repeated calls on one instance agree.
    """

    @pytest.mark.parametrize(("cls", "kwargs"), _SEEDED_INITIALIZERS)
    def test_repeated_calls_same_instance_are_identical(self, cls, kwargs) -> None:
        """A second ``initialize()`` on the same instance reproduces the first."""
        init = cls(seed=42, **kwargs)

        w1 = torch.empty(100, 10)
        w2 = torch.empty(100, 10)
        init.initialize(w1)
        init.initialize(w2)

        assert torch.allclose(w1, w2)

    @pytest.mark.parametrize(("cls", "kwargs"), _SEEDED_INITIALIZERS)
    def test_two_instances_same_seed_agree(self, cls, kwargs) -> None:
        """Two separate instances with the same seed produce identical matrices."""
        init_a = cls(seed=7, **kwargs)
        init_b = cls(seed=7, **kwargs)

        wa = torch.empty(100, 10)
        wb = torch.empty(100, 10)
        init_a.initialize(wa)
        init_b.initialize(wb)

        assert torch.allclose(wa, wb)

    @pytest.mark.parametrize(("cls", "kwargs"), _SEEDED_INITIALIZERS)
    def test_seed_none_draws_fresh_each_call(self, cls, kwargs) -> None:
        """With ``seed=None`` repeated calls draw fresh (non-identical) matrices."""
        init = cls(seed=None, **kwargs)

        w1 = torch.empty(200, 5)
        w2 = torch.empty(200, 5)
        init.initialize(w1)
        init.initialize(w2)

        assert not torch.allclose(w1, w2)

    def test_get_input_feedback_two_instances_agree(self) -> None:
        """Two ``get_input_feedback('random', seed=42)`` instances still agree."""
        init_a = get_input_feedback("random", seed=42)
        init_b = get_input_feedback("random", seed=42)

        wa = torch.empty(100, 10)
        wb = torch.empty(100, 10)
        init_a.initialize(wa)
        init_b.initialize(wb)

        assert torch.allclose(wa, wb)


class TestInitializersOnDevice:
    """Custom initializer objects driving an ESNLayer on every device."""

    def test_custom_initializer_objects_on_device(self, device: torch.device) -> None:
        """Chebyshev/random initializer objects produce a working layer."""
        feedback_init = ChebyshevInitializer(p=0.3, k=3.5, input_scaling=0.8)
        input_init = RandomInputInitializer(input_scaling=1.0, seed=42)

        reservoir = ESNLayer(
            reservoir_size=100,
            feedback_size=10,
            input_size=5,
            feedback_initializer=feedback_init,
            input_initializer=input_init,
        ).to(device)

        # All parameters should be on the device
        assert reservoir.weight_feedback.device.type == device.type
        assert reservoir.weight_input.device.type == device.type
        assert reservoir.weight_hh.device.type == device.type

        # Forward pass should work
        feedback = torch.randn(2, 10, 10, device=device)
        driving = torch.randn(2, 10, 5, device=device)
        output = reservoir(feedback, driving)

        assert output.device.type == device.type
        assert output.shape == (2, 10, 100)


class TestRandomInitializersDeviceNativeSeed:
    """Issue #188: torch-native, device-native, ``torch.Generator``-aware seed.

    The ``random``/``random_binary`` initializers used to draw via NumPy on CPU
    and copy to the device, accepting only an ``int`` seed. They now draw with a
    :class:`torch.Generator` directly on the target tensor's device and accept a
    generator seed as well, so reproducible draws are possible on any device.
    """

    _CLASSES = [
        pytest.param(RandomInputInitializer, {"input_scaling": 1.0}, id="random"),
        pytest.param(RandomBinaryInitializer, {"input_scaling": 0.5}, id="random_binary"),
    ]

    @pytest.mark.parametrize(("cls", "kwargs"), _CLASSES)
    def test_draws_on_target_device(self, cls, kwargs, device: torch.device) -> None:
        """The filled weight stays on the device it was passed in on."""
        init = cls(seed=42, **kwargs)
        weight = torch.empty(50, 8, device=device)
        init.initialize(weight)

        assert weight.device.type == device.type

    @pytest.mark.parametrize(("cls", "kwargs"), _CLASSES)
    def test_reproducible_per_device(self, cls, kwargs, device: torch.device) -> None:
        """Two seeded draws on the same device are byte-identical."""
        a = torch.empty(50, 8, device=device)
        b = torch.empty(50, 8, device=device)
        cls(seed=7, **kwargs).initialize(a)
        cls(seed=7, **kwargs).initialize(b)

        assert torch.equal(a, b)

    @pytest.mark.parametrize(("cls", "kwargs"), _CLASSES)
    def test_accepts_torch_generator_seed(self, cls, kwargs) -> None:
        """A ``torch.Generator`` seed is accepted and reproducible."""
        g1 = torch.Generator().manual_seed(123)
        g2 = torch.Generator().manual_seed(123)
        a = torch.empty(40, 6)
        b = torch.empty(40, 6)
        cls(seed=g1, **kwargs).initialize(a)
        cls(seed=g2, **kwargs).initialize(b)

        assert torch.equal(a, b)

    @pytest.mark.parametrize(("cls", "kwargs"), _CLASSES)
    def test_generator_seed_matches_equivalent_int(self, cls, kwargs) -> None:
        """A generator seeded with ``N`` draws the same matrix as ``seed=N``.

        The generator is reduced to its ``initial_seed()``, so the two seed
        forms must agree on the produced weights.
        """
        g = torch.Generator().manual_seed(99)
        a = torch.empty(40, 6)
        b = torch.empty(40, 6)
        cls(seed=g, **kwargs).initialize(a)
        cls(seed=99, **kwargs).initialize(b)

        assert torch.equal(a, b)

    def test_random_binary_values_are_signed_scaled(self) -> None:
        """``random_binary`` stays a ``{-s, +s}`` matrix after the torch draw."""
        init = RandomBinaryInitializer(input_scaling=0.5, seed=0)
        weight = torch.empty(64, 8)
        init.initialize(weight)

        assert set(torch.unique(weight).tolist()) <= {-0.5, 0.5}


class TestStandardDistributionInitializers:
    """Issue #192: bread-and-butter ``normal`` / ``uniform`` / ``bernoulli``.

    The reservoirpy migration story: standard input-matrix recipes with a
    ``connectivity`` density knob and the shared ``input_scaling`` magnitude
    knob. Covers shape, sparsity (verified statistically), scaling, and
    reproducibility for each.
    """

    # (registry name, kwargs that distinguish the draw) for the three new ones.
    _STANDARD = [
        pytest.param("normal", {"loc": 0.0, "scale": 1.0}, id="normal"),
        pytest.param("uniform", {"low": -1.0, "high": 1.0}, id="uniform"),
        pytest.param("bernoulli", {"p": 0.5}, id="bernoulli"),
    ]

    @pytest.mark.parametrize(("name", "kwargs"), _STANDARD)
    def test_registry_returns_working_initializer(self, name: str, kwargs: dict) -> None:
        """``get_input_feedback`` resolves each new name to a working initializer."""
        init = get_input_feedback(name, **kwargs)
        weight = torch.empty(100, 10)
        out = init.initialize(weight)

        assert out is weight  # in-place
        assert weight.shape == (100, 10)
        assert torch.isfinite(weight).all()

    @pytest.mark.parametrize(("name", "kwargs"), _STANDARD)
    def test_connectivity_yields_expected_sparsity(self, name: str, kwargs: dict) -> None:
        """``connectivity=0.1`` keeps ~10% nonzero entries (verified statistically).

        The base contract keeps exactly ``round(connectivity * rows)`` nonzeros
        per column, so a tall matrix lands the global density on ~0.1. The
        ``bernoulli`` draw is never zero before masking, so its density is the
        mask density exactly; ``normal`` / ``uniform`` could in principle draw a
        zero, but the probability over the kept entries is negligible.
        """
        rows = 1000
        init = get_input_feedback(name, connectivity=0.1, seed=0, **kwargs)
        weight = torch.empty(rows, 8)
        init.initialize(weight)

        per_column_nonzero = (weight != 0).sum(dim=0)
        # round(0.1 * 1000) == 100 kept per column.
        assert torch.equal(per_column_nonzero, torch.full((8,), 100))

        density = (weight != 0).float().mean().item()
        assert density == pytest.approx(0.1, abs=0.005)

    @pytest.mark.parametrize(("name", "kwargs"), _STANDARD)
    def test_connectivity_none_is_dense(self, name: str, kwargs: dict) -> None:
        """``connectivity=None`` leaves a fully dense matrix (no zeroed entries)."""
        init = get_input_feedback(name, connectivity=None, seed=0, **kwargs)
        weight = torch.empty(200, 5)
        init.initialize(weight)

        # bernoulli/uniform/normal are a.s. nonzero with these params, so a dense
        # build keeps every entry.
        assert (weight != 0).all()

    @pytest.mark.parametrize(("name", "kwargs"), _STANDARD)
    def test_input_scaling_is_uniform_pointwise_multiply(self, name: str, kwargs: dict) -> None:
        """``input_scaling=0.5`` is exactly the elementwise half of ``input_scaling=1.0``.

        Same ``seed`` + same ``connectivity`` mask, so the only difference is the
        documented uniform final multiply.
        """
        base = torch.empty(128, 6, dtype=torch.float64)
        half = torch.empty(128, 6, dtype=torch.float64)
        get_input_feedback(name, input_scaling=1.0, connectivity=0.5, seed=7, **kwargs).initialize(
            base
        )
        get_input_feedback(name, input_scaling=0.5, connectivity=0.5, seed=7, **kwargs).initialize(
            half
        )

        assert torch.allclose(half, 0.5 * base, atol=1e-12)
        # The connectivity mask must coincide so the comparison is meaningful.
        assert torch.equal(base != 0, half != 0)

    @pytest.mark.parametrize(("name", "kwargs"), _STANDARD)
    def test_reproducible_under_fixed_seed(self, name: str, kwargs: dict) -> None:
        """Two instances with the same seed (value + mask) are byte-identical."""
        a = torch.empty(100, 10)
        b = torch.empty(100, 10)
        get_input_feedback(name, connectivity=0.1, seed=99, **kwargs).initialize(a)
        get_input_feedback(name, connectivity=0.1, seed=99, **kwargs).initialize(b)

        assert torch.equal(a, b)

    def test_normal_matches_loc_and_scale(self) -> None:
        """``normal`` draws track the requested mean and standard deviation."""
        init = NormalInputInitializer(loc=2.0, scale=0.5, seed=0)
        weight = torch.empty(2000, 4, dtype=torch.float64)
        init.initialize(weight)

        assert weight.mean().item() == pytest.approx(2.0, abs=0.05)
        assert weight.std().item() == pytest.approx(0.5, abs=0.05)

    def test_uniform_respects_bounds(self) -> None:
        """``uniform`` draws stay within ``[low, high)`` and span most of it."""
        init = UniformInputInitializer(low=0.0, high=0.5, seed=0)
        weight = torch.empty(2000, 4, dtype=torch.float64)
        init.initialize(weight)

        assert weight.min().item() >= 0.0
        assert weight.max().item() < 0.5
        # A 8000-sample draw covers the bulk of the interval.
        assert weight.min().item() < 0.02
        assert weight.max().item() > 0.48

    def test_bernoulli_values_are_signed(self) -> None:
        """``bernoulli`` produces only ``{-1, +1}`` before any scaling/masking."""
        init = BernoulliInputInitializer(p=0.5, seed=0)
        weight = torch.empty(64, 8)
        init.initialize(weight)

        assert set(torch.unique(weight).tolist()) <= {-1.0, 1.0}

    def test_bernoulli_p_controls_positive_fraction(self) -> None:
        """``p`` is the probability of drawing ``+1`` (verified statistically)."""
        init = BernoulliInputInitializer(p=0.8, seed=0)
        weight = torch.empty(2000, 4)
        init.initialize(weight)

        positive_fraction = (weight > 0).float().mean().item()
        assert positive_fraction == pytest.approx(0.8, abs=0.03)

    def test_bernoulli_scaled_values_are_signed_scaled(self) -> None:
        """``bernoulli`` stays a ``{-s, +s}`` matrix after the scaling contract."""
        init = BernoulliInputInitializer(p=0.5, input_scaling=0.25, seed=0)
        weight = torch.empty(64, 8)
        init.initialize(weight)

        assert set(torch.unique(weight).tolist()) <= {-0.25, 0.25}

    @pytest.mark.parametrize(("name", "kwargs"), _STANDARD)
    def test_draws_on_target_device(self, name: str, kwargs: dict, device: torch.device) -> None:
        """The filled weight stays on the device it was passed in on."""
        init = get_input_feedback(name, connectivity=0.2, seed=0, **kwargs)
        weight = torch.empty(50, 8, device=device)
        init.initialize(weight)

        assert weight.device.type == device.type

    def test_normal_rejects_negative_scale(self) -> None:
        """A negative ``scale`` is rejected at construction."""
        with pytest.raises(ValueError, match="scale"):
            NormalInputInitializer(scale=-1.0)

    @pytest.mark.parametrize(("low", "high"), [(1.0, 1.0), (1.0, 0.0)])
    def test_uniform_rejects_bad_bounds(self, low: float, high: float) -> None:
        """``low`` must be strictly less than ``high``."""
        with pytest.raises(ValueError, match="low"):
            UniformInputInitializer(low=low, high=high)

    @pytest.mark.parametrize("bad_p", [-0.1, 1.1])
    def test_bernoulli_rejects_p_out_of_range(self, bad_p: float) -> None:
        """``p`` must lie in ``[0, 1]``."""
        with pytest.raises(ValueError, match="p"):
            BernoulliInputInitializer(p=bad_p)


class TestOppositeAnchorsCollision:
    """Regression coverage for the duplicate-column bug (issue #142).

    The original semicircle anchor placement mapped distinct input channels to
    the same ``(j0, j1)`` anchor pair once ``in_features > reservoir_size // 2``,
    producing bit-identical columns (a silent rank deficiency in ``W_in``/``W_fb``).
    The fix spreads anchors around the full ring so columns stay distinct for any
    ``in_features <= reservoir_size``, and raises when ``in_features > reservoir_size``.
    """

    def test_regression_n6_m5_distinct_columns(self) -> None:
        """The reported case ``n=6, m=5`` yields five distinct columns."""
        init = OppositeAnchorsInitializer(input_scaling=1.0)
        weight = torch.empty(6, 5)  # (reservoir_size=6, in_features=5)
        init.initialize(weight)

        for i in range(5):
            for j in range(i + 1, 5):
                assert not torch.allclose(
                    weight[:, i], weight[:, j]
                ), f"columns {i} and {j} are identical (n=6, m=5)"

    @pytest.mark.parametrize("n", [6, 7, 16, 17, 50])
    def test_no_identical_columns_up_to_reservoir_size(self, n: int) -> None:
        """No two columns are identical for ``in_features`` up to ``reservoir_size``."""
        for m in range(1, n + 1):
            weight = torch.empty(n, m)
            OppositeAnchorsInitializer(input_scaling=1.0).initialize(weight)

            for i in range(m):
                for j in range(i + 1, m):
                    assert not torch.allclose(
                        weight[:, i], weight[:, j]
                    ), f"columns {i} and {j} are identical (n={n}, m={m})"

    def test_in_features_exceeds_reservoir_size_raises(self) -> None:
        """``in_features > reservoir_size`` raises a clear ``ValueError``."""
        init = OppositeAnchorsInitializer(input_scaling=1.0)
        weight = torch.empty(6, 7)  # (reservoir_size=6, in_features=7)

        with pytest.raises(ValueError, match="in_features <= reservoir_size"):
            init.initialize(weight)

    def test_single_channel_full_ring(self) -> None:
        """A single channel (``m=1``) still produces a valid bipolar column."""
        init = OppositeAnchorsInitializer(input_scaling=1.0)
        weight = torch.empty(8, 1)
        init.initialize(weight)

        # Exactly one positive and one negative anchor, equal magnitude.
        nonzero = weight[weight != 0]
        assert nonzero.numel() == 2
        assert torch.isclose(nonzero.abs()[0], nonzero.abs()[1])


# Structured initializers whose intermediates historically went through a
# hard-coded ``np.float32`` array, truncating precision in float64 reservoirs
# (issue #143). Each factory takes a single ``cls``/``kwargs`` pair plus a
# ``shape`` whose generated values are *not* all float32-representable, so a
# genuine-float64 weight can be distinguished from a float32-widened one.
#
# An irrational ``input_scaling`` / ``gain`` (1/sqrt(2) ~ 0.7071...) guarantees
# at least one stored value lands strictly between two float32 grid points.
_IRRATIONAL_SCALE = 2.0**-0.5  # 0.7071067811865476, not float32-representable

_PRECISION_INITIALIZERS = [
    pytest.param(
        ChebyshevInitializer,
        {"p": 0.3, "k": 3.5, "input_scaling": _IRRATIONAL_SCALE},
        (64, 8),
        id="chebyshev",
    ),
    pytest.param(
        PseudoDiagonalInitializer,
        {"input_scaling": _IRRATIONAL_SCALE, "seed": 42},
        (64, 8),
        id="pseudo_diagonal",
    ),
    pytest.param(
        ChessboardInitializer,
        {"input_scaling": _IRRATIONAL_SCALE},
        (64, 8),
        id="chessboard",
    ),
    pytest.param(
        BinaryBalancedInitializer,
        {"input_scaling": _IRRATIONAL_SCALE},
        (64, 8),
        id="binary_balanced",
    ),
    pytest.param(
        OppositeAnchorsInitializer,
        {"input_scaling": 1.0},  # input_scaling / sqrt(2) is already irrational
        (64, 8),
        id="opposite_anchors",
    ),
    pytest.param(
        DendrocycleInputInitializer,
        {"c": 0.5, "input_scaling": _IRRATIONAL_SCALE, "seed": 42},
        (64, 8),
        id="dendrocycle_input",
    ),
    pytest.param(
        ChainOfNeuronsInputInitializer,
        {"features": 8, "weights": _IRRATIONAL_SCALE},
        (64, 8),
        id="chain_of_neurons_input",
    ),
    pytest.param(
        RingWindowInputInitializer,
        {"c": 0.5, "window": 0.5, "taper": "cosine", "signed": "alt_ring"},
        (64, 8),
        id="ring_window",
    ),
]


class TestStructuredInitializerPrecision:
    """Structured initializers must respect the target dtype (issue #143).

    Previously every structured initializer built its value matrix in a
    hard-coded ``np.float32`` array and only widened at the final
    ``.to(dtype=weight.dtype)``, so a float64 reservoir silently lost true
    double precision. The fix builds intermediates at the target precision.
    """

    @pytest.mark.parametrize(("cls", "kwargs", "shape"), _PRECISION_INITIALIZERS)
    def test_float64_weight_keeps_double_precision(self, cls, kwargs, shape) -> None:
        """A float64 weight carries at least one non-float32-representable value."""
        weight = torch.empty(*shape, dtype=torch.float64)
        cls(**kwargs).initialize(weight)

        # If the intermediate had been float32, every stored value would equal
        # its float32 round-trip. Genuine float64 precision breaks that for at
        # least one entry.
        round_tripped = weight.to(torch.float32).to(torch.float64)
        assert torch.any(weight != round_tripped), (
            f"{cls.__name__} float64 weight is fully float32-representable; "
            "precision was truncated."
        )

    @pytest.mark.parametrize(("cls", "kwargs", "shape"), _PRECISION_INITIALIZERS)
    def test_float32_weight_is_deterministic_and_representable(self, cls, kwargs, shape) -> None:
        """A float32 weight is reproducible and stays genuinely float32 (unchanged).

        For a float32 target the intermediate dtype is float32, so the
        computation path is identical to the historical behavior. We pin down
        that the result is deterministic and never carries hidden sub-float32
        bits, guarding against accidental widening.
        """
        w1 = torch.empty(*shape, dtype=torch.float32)
        w2 = torch.empty(*shape, dtype=torch.float32)
        cls(**kwargs).initialize(w1)
        cls(**kwargs).initialize(w2)

        # float32 path is deterministic (seeded / fully deterministic builders).
        assert torch.equal(w1, w2)
        # And every value is genuinely float32 (no widening surprises).
        assert w1.dtype == torch.float32
        assert torch.equal(w1, w1.to(torch.float64).to(torch.float32))


def _max_abs(weight: torch.Tensor) -> float:
    """Documented magnitude statistic for elementwise initializers: ``max|W|``."""
    return float(weight.abs().max())


def _max_channel_l2(weight: torch.Tensor) -> float:
    """Documented magnitude statistic for ring initializers: per-channel L2 norm.

    Returns the largest column (input-channel) L2 norm, which the structured ring
    initializers (``opposite_anchors``, ``ring_window``) pin to ``input_scaling``.
    """
    return float(torch.linalg.vector_norm(weight, dim=0).max())


# (factory, base_kwargs, statistic, shape) for every initializer that honors the
# shared ``input_scaling`` contract. ``statistic`` maps a weight to the magnitude
# the initializer documents as scaling linearly with ``input_scaling``:
#   - elementwise initializers -> ``max|W|``
#   - structured ring initializers -> per-channel L2 norm
_SCALING_CONTRACT = [
    pytest.param(
        lambda s: RandomInputInitializer(input_scaling=s, seed=0),
        _max_abs,
        (128, 6),
        id="random",
    ),
    pytest.param(
        lambda s: RandomBinaryInitializer(input_scaling=s, seed=0),
        _max_abs,
        (128, 6),
        id="random_binary",
    ),
    pytest.param(
        lambda s: NormalInputInitializer(input_scaling=s, seed=0),
        _max_abs,
        (128, 6),
        id="normal",
    ),
    pytest.param(
        lambda s: UniformInputInitializer(input_scaling=s, seed=0),
        _max_abs,
        (128, 6),
        id="uniform",
    ),
    pytest.param(
        lambda s: BernoulliInputInitializer(input_scaling=s, seed=0),
        _max_abs,
        (128, 6),
        id="bernoulli",
    ),
    pytest.param(
        lambda s: PseudoDiagonalInitializer(input_scaling=s, seed=0),
        _max_abs,
        (128, 6),
        id="pseudo_diagonal",
    ),
    pytest.param(
        lambda s: ChessboardInitializer(input_scaling=s),
        _max_abs,
        (64, 8),
        id="chessboard",
    ),
    pytest.param(
        lambda s: ChebyshevInitializer(p=0.3, k=3.5, input_scaling=s),
        _max_abs,
        (64, 8),
        id="chebyshev",
    ),
    pytest.param(
        lambda s: BinaryBalancedInitializer(input_scaling=s),
        _max_abs,
        (64, 8),
        id="binary_balanced",
    ),
    pytest.param(
        lambda s: DendrocycleInputInitializer(c=0.5, draw_width=1.0, input_scaling=s, seed=0),
        _max_abs,
        (64, 8),
        id="dendrocycle_input",
    ),
    pytest.param(
        lambda s: OppositeAnchorsInitializer(input_scaling=s),
        _max_channel_l2,
        (64, 8),
        id="opposite_anchors",
    ),
    pytest.param(
        lambda s: RingWindowInputInitializer(
            c=0.5, window=0.5, taper="cosine", signed="alt_ring", input_scaling=s
        ),
        _max_channel_l2,
        (64, 8),
        id="ring_window",
    ),
]


class TestScalingContract:
    """The shared ``input_scaling`` contract: a uniform, linear magnitude knob.

    Acceptance criterion: ``max|W|`` (or the documented per-channel L2 norm for
    the structured ring initializers) scales **linearly** with ``input_scaling``
    for every initializer.
    """

    @pytest.mark.parametrize(("factory", "statistic", "shape"), _SCALING_CONTRACT)
    def test_statistic_scales_linearly_with_input_scaling(self, factory, statistic, shape) -> None:
        """Doubling/halving ``input_scaling`` doubles/halves the magnitude statistic."""
        base = torch.empty(*shape, dtype=torch.float64)
        half = torch.empty(*shape, dtype=torch.float64)
        double = torch.empty(*shape, dtype=torch.float64)

        factory(1.0).initialize(base)
        factory(0.5).initialize(half)
        factory(2.0).initialize(double)

        s_base = statistic(base)
        assert s_base > 0.0  # the chosen shape produces a non-trivial matrix

        # input_scaling=0.5 -> magnitude * 0.5; input_scaling=2.0 -> magnitude * 2.0.
        assert statistic(half) == pytest.approx(0.5 * s_base, rel=1e-9)
        assert statistic(double) == pytest.approx(2.0 * s_base, rel=1e-9)

    @pytest.mark.parametrize(("factory", "statistic", "shape"), _SCALING_CONTRACT)
    def test_input_scaling_half_is_pointwise_half(self, factory, statistic, shape) -> None:
        """``input_scaling=0.5`` is exactly the elementwise half of ``input_scaling=1.0``.

        The contract states ``input_scaling`` is a *uniform* final multiply, so
        every entry — not just the magnitude statistic — must scale together.
        """
        base = torch.empty(*shape, dtype=torch.float64)
        half = torch.empty(*shape, dtype=torch.float64)

        factory(1.0).initialize(base)
        factory(0.5).initialize(half)

        assert torch.allclose(half, 0.5 * base, atol=1e-12)


class TestChebyshevColumnBalance:
    """The Chebyshev map must not de-weight the first feedback dimension (issue #190).

    The seed column lives on ``[-p, p]`` while the chaotic columns span ``[-1, 1]``,
    so the raw map left column 0 ~``1/p`` weaker than the rest. Each column is now
    normalized to unit peak amplitude before ``input_scaling`` applies, so every
    feedback dimension peaks at the same magnitude.
    """

    def test_per_column_max_abs_is_near_uniform(self) -> None:
        """Per-column ``max|W|`` of a (100, 5) Chebyshev matrix are all near-equal."""
        weight = torch.empty(100, 5, dtype=torch.float64)
        ChebyshevInitializer().initialize(weight)

        col_max = weight.abs().max(dim=0).values
        # Without the fix col 0 was ~0.151 against ~1.0 elsewhere (a ~6.6x gap);
        # every column now peaks at 1.0 (no input_scaling).
        assert torch.allclose(col_max, torch.ones_like(col_max), atol=1e-9)
        # The first column is no longer the weak one.
        assert float(col_max[0]) == pytest.approx(float(col_max.max()), rel=1e-9)

    def test_per_column_max_abs_tracks_input_scaling(self) -> None:
        """Every column peaks at ``input_scaling`` once the scaling contract applies."""
        scale = 0.5
        weight = torch.empty(100, 5, dtype=torch.float64)
        ChebyshevInitializer(input_scaling=scale).initialize(weight)

        col_max = weight.abs().max(dim=0).values
        assert torch.allclose(col_max, torch.full_like(col_max, scale), atol=1e-9)

    def test_single_column_feedback_matches_input_scaling(self) -> None:
        """A 1D-feedback matrix peaks at ``input_scaling``, not the seed amplitude ``p``."""
        # Without scaling the lone seed column now peaks at 1.0 instead of ~p.
        unscaled = torch.empty(64, 1, dtype=torch.float64)
        ChebyshevInitializer(p=0.3).initialize(unscaled)
        assert float(unscaled.abs().max()) == pytest.approx(1.0, rel=1e-9)

        # And with input_scaling it peaks at exactly that scale.
        scaled = torch.empty(64, 1, dtype=torch.float64)
        ChebyshevInitializer(p=0.3, input_scaling=0.8).initialize(scaled)
        assert float(scaled.abs().max()) == pytest.approx(0.8, rel=1e-9)


class TestScalingContractBase:
    """Unit coverage for the contract helpers owned by ``InputFeedbackInitializer``."""

    class _Identity(InputFeedbackInitializer):
        """Minimal subclass exposing the protected helpers for testing."""

        def initialize(self, weight: torch.Tensor, **kwargs) -> torch.Tensor:
            return weight

    def test_apply_scaling_none_is_identity_numpy(self) -> None:
        """``input_scaling=None`` returns the array unchanged (numpy)."""
        init = self._Identity(input_scaling=None)
        values = np.array([[1.0, -2.0], [3.0, -4.0]])
        out = init._apply_scaling(values)
        assert np.array_equal(out, values)

    def test_apply_scaling_multiplies_numpy(self) -> None:
        """A float ``input_scaling`` multiplies a numpy array uniformly."""
        init = self._Identity(input_scaling=0.5)
        values = np.array([[1.0, -2.0], [3.0, -4.0]])
        out = init._apply_scaling(values)
        assert np.allclose(out, 0.5 * values)

    def test_apply_scaling_multiplies_torch(self) -> None:
        """A float ``input_scaling`` multiplies a torch tensor uniformly."""
        init = self._Identity(input_scaling=2.0)
        values = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
        out = init._apply_scaling(values)
        assert torch.allclose(out, 2.0 * values)

    def test_apply_scaling_preserves_dtype(self) -> None:
        """Scaling a float64 numpy intermediate stays float64."""
        init = self._Identity(input_scaling=_IRRATIONAL_SCALE)
        values = np.ones((3, 3), dtype=np.float64)
        assert init._apply_scaling(values).dtype == np.float64

    def test_apply_connectivity_none_is_identity(self) -> None:
        """``connectivity=None`` leaves the produced sparsity untouched."""
        init = self._Identity(connectivity=None, seed=0)
        values = np.ones((10, 4))
        assert np.array_equal(init._apply_connectivity(values), values)

    def test_apply_connectivity_density_numpy(self) -> None:
        """``connectivity=c`` keeps round(c * rows) nonzeros per column (numpy)."""
        init = self._Identity(connectivity=0.5, seed=0)
        values = np.ones((10, 4))
        out = init._apply_connectivity(values)
        # Each column keeps exactly round(0.5 * 10) == 5 nonzeros.
        assert np.array_equal((out != 0).sum(axis=0), np.full(4, 5))

    def test_apply_connectivity_density_torch(self) -> None:
        """``connectivity=c`` keeps round(c * rows) nonzeros per column (torch)."""
        init = self._Identity(connectivity=0.3, seed=1)
        values = torch.ones(20, 3)
        out = init._apply_connectivity(values)
        keep = round(0.3 * 20)
        assert torch.equal((out != 0).sum(dim=0), torch.full((3,), keep))

    def test_apply_connectivity_keeps_at_least_one(self) -> None:
        """A tiny ``connectivity`` still keeps at least one nonzero per column."""
        init = self._Identity(connectivity=0.01, seed=0)
        values = np.ones((10, 4))
        out = init._apply_connectivity(values)
        assert np.all((out != 0).sum(axis=0) >= 1)

    def test_invalid_input_scaling_raises(self) -> None:
        """A non-finite ``input_scaling`` is rejected at construction."""
        with pytest.raises(ValueError, match="input_scaling"):
            self._Identity(input_scaling=float("inf"))

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_invalid_connectivity_raises(self, bad: float) -> None:
        """``connectivity`` outside ``(0, 1]`` is rejected at construction."""
        with pytest.raises(ValueError, match="connectivity"):
            self._Identity(connectivity=bad)


class TestGainDeprecatedAlias:
    """``gain`` is a deprecated alias for ``input_scaling`` where it existed."""

    @pytest.mark.parametrize("cls", [OppositeAnchorsInitializer])
    def test_gain_warns_and_aliases_opposite_anchors(self, cls) -> None:
        """Passing ``gain`` warns and behaves identically to ``input_scaling``."""
        with pytest.warns(DeprecationWarning, match="gain"):
            via_gain = cls(gain=0.7)
        via_scaling = cls(input_scaling=0.7)

        assert via_gain.input_scaling == via_scaling.input_scaling
        assert via_gain.gain == 0.7  # property mirrors input_scaling

        w_gain = torch.empty(32, 4, dtype=torch.float64)
        w_scaling = torch.empty(32, 4, dtype=torch.float64)
        via_gain.initialize(w_gain)
        via_scaling.initialize(w_scaling)
        assert torch.allclose(w_gain, w_scaling)

    def test_gain_warns_and_aliases_ring_window(self) -> None:
        """``ring_window`` accepts the deprecated ``gain`` alias too."""
        with pytest.warns(DeprecationWarning, match="gain"):
            via_gain = RingWindowInputInitializer(c=0.5, window=0.5, gain=0.7)
        via_scaling = RingWindowInputInitializer(c=0.5, window=0.5, input_scaling=0.7)

        w_gain = torch.empty(32, 4, dtype=torch.float64)
        w_scaling = torch.empty(32, 4, dtype=torch.float64)
        via_gain.initialize(w_gain)
        via_scaling.initialize(w_scaling)
        assert torch.allclose(w_gain, w_scaling)

    def test_both_input_scaling_and_gain_raises(self) -> None:
        """Supplying both names is ambiguous and raises."""
        with pytest.raises(ValueError, match="only one of"):
            OppositeAnchorsInitializer(input_scaling=0.5, gain=0.5)

    def test_gain_alias_via_registry(self) -> None:
        """``get_input_feedback`` honors the ``gain`` alias with a warning."""
        with pytest.warns(DeprecationWarning, match="gain"):
            init = get_input_feedback("opposite_anchors", gain=0.5)
        assert init.input_scaling == 0.5


class TestDendrocycleDrawWidthVsScaling:
    """Dendrocycle separates its draw half-width from the uniform scaling knob."""

    def test_draw_width_sets_raw_spread(self) -> None:
        """With ``input_scaling=None``, ``max|W|`` is bounded by ``draw_width``."""
        weight = torch.empty(200, 4, dtype=torch.float64)
        DendrocycleInputInitializer(c=0.5, draw_width=0.5, seed=0).initialize(weight)
        assert 0.0 < float(weight.abs().max()) <= 0.5

    def test_input_scaling_multiplies_on_top_of_draw_width(self) -> None:
        """``input_scaling`` is a uniform multiply *on top of* the raw draw."""
        raw = torch.empty(200, 4, dtype=torch.float64)
        scaled = torch.empty(200, 4, dtype=torch.float64)
        DendrocycleInputInitializer(c=0.5, draw_width=1.0, seed=0).initialize(raw)
        DendrocycleInputInitializer(c=0.5, draw_width=1.0, input_scaling=0.25, seed=0).initialize(
            scaled
        )
        assert torch.allclose(scaled, 0.25 * raw, atol=1e-12)


# ---------------------------------------------------------------------------
# Registry-driven sweep over every input/feedback initializer (issue #207)
# ---------------------------------------------------------------------------

_SWEEP_RESERVOIR = 36
# Feedback and driving-input widths are kept equal so a single per-name
# override (notably ``chain_of_neurons_input``'s ``features``, which must equal
# the weight's column count) is valid for both the feedback and the input
# weight matrices the forward-pass test fills.
_SWEEP_FEEDBACK = 4
_SWEEP_INPUT = 4
_SWEEP_SEED = 1234

# A handful of initializers declare *required* constructor parameters that the
# registry defaults leave unset (they default to ``None`` and the initializer
# rejects that). Supply minimal valid values keyed by name so the sweep can
# build them; every other initializer uses its registry defaults, so a newly
# registered one is swept with no edit here.
_REQUIRED_KWARGS: dict[str, dict[str, object]] = {
    # ``features`` must equal the weight's column count (the feedback/input width).
    "chain_of_neurons_input": {"features": _SWEEP_FEEDBACK},
    # Exactly one of ``c`` / ``C`` is required.
    "dendrocycle_input": {"c": 0.5},
    # ``c`` and ``window`` are both required.
    "ring_window": {"c": 0.5, "window": 0.5},
}


def _accepts_seed(initializer: InputFeedbackInitializer) -> bool:
    """Return ``True`` when the initializer's constructor names a ``seed`` param."""
    return "seed" in inspect.signature(type(initializer).__init__).parameters


def _build_swept_initializer(name: str, *, seed: int | None) -> InputFeedbackInitializer:
    """Build a fully-resolved initializer for the sweep via the registry.

    The required-kwarg overrides are merged in, and ``seed`` is forwarded only
    to initializers whose constructor accepts it (so deterministic builders are
    left untouched). The resolved *object* is returned so callers can pass it to
    ``ESNLayer`` directly, side-stepping the ``(name, params)`` tuple-spec path.
    """
    kwargs = dict(_REQUIRED_KWARGS.get(name, {}))
    probe = get_input_feedback(name, **kwargs)
    if seed is not None and _accepts_seed(probe):
        return get_input_feedback(name, seed=seed, **kwargs)
    return probe


class TestEveryInitializerBuildsAndForwards:
    """Every registered input/feedback initializer drives a working ``ESNLayer``.

    Data-driven coverage for the initializers the targeted tests above never
    exercise end to end (``binary_balanced``, ``chain_of_neurons_input``,
    ``chessboard``, ``dendrocycle_input``, ``opposite_anchors``,
    ``random_binary``, ``ring_window``, ...). Iterating over
    :func:`show_input_initializers` means a newly registered initializer is
    swept automatically (issue #207).
    """

    @pytest.mark.parametrize("name", show_input_initializers())
    def test_initializer_produces_finite_weight(self, name: str) -> None:
        """A direct ``initialize`` fills a finite ``(reservoir, feedback)`` weight."""
        initializer = _build_swept_initializer(name, seed=_SWEEP_SEED)
        weight = torch.empty(_SWEEP_RESERVOIR, _SWEEP_FEEDBACK)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            initializer.initialize(weight)

        assert weight.shape == (_SWEEP_RESERVOIR, _SWEEP_FEEDBACK)
        assert torch.isfinite(weight).all(), f"{name} produced non-finite weight"

    @pytest.mark.parametrize("name", show_input_initializers())
    def test_initializer_drives_finite_forward_pass(self, name: str) -> None:
        """Each initializer wires both feedback and input weights of a layer.

        The resolved object is passed for *both* the feedback and the driving
        input weights, so a single forward pass exercises the initializer on the
        two rectangular weight matrices it is meant to fill.
        """
        feedback_init = _build_swept_initializer(name, seed=_SWEEP_SEED)
        input_init = _build_swept_initializer(name, seed=_SWEEP_SEED)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            layer = ESNLayer(
                reservoir_size=_SWEEP_RESERVOIR,
                feedback_size=_SWEEP_FEEDBACK,
                input_size=_SWEEP_INPUT,
                feedback_initializer=feedback_init,
                input_initializer=input_init,
                seed=_SWEEP_SEED,
            )

        feedback = torch.randn(2, 5, _SWEEP_FEEDBACK)
        driving = torch.randn(2, 5, _SWEEP_INPUT)
        output = layer(feedback, driving)

        assert output.shape == (2, 5, _SWEEP_RESERVOIR)
        assert torch.isfinite(output).all(), f"{name} produced non-finite output"
        assert torch.isfinite(layer.weight_feedback).all()
        assert torch.isfinite(layer.weight_input).all()

    @pytest.mark.parametrize("name", show_input_initializers())
    def test_sweep_is_seeded_and_reproducible(self, name: str) -> None:
        """Two builds with the same seed produce identical weights for every name.

        Deterministic initializers pass trivially; the seeded random ones
        (``random``, ``random_binary``, ``pseudo_diagonal``, ...) require the
        forwarded seed to make this hold.
        """
        wa = torch.empty(_SWEEP_RESERVOIR, _SWEEP_FEEDBACK)
        wb = torch.empty(_SWEEP_RESERVOIR, _SWEEP_FEEDBACK)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            _build_swept_initializer(name, seed=_SWEEP_SEED).initialize(wa)
            _build_swept_initializer(name, seed=_SWEEP_SEED).initialize(wb)

        assert torch.equal(wa, wb), f"{name} not reproducible under a fixed seed"


# ---------------------------------------------------------------------------
# Targeted initializer-branch coverage (issue #207)
# ---------------------------------------------------------------------------
#
# The sweep above builds every initializer with default parameters and an even,
# reservoir-shaped weight (tall: out_features >> in_features). That leaves
# several builder branches unreached: the odd-row balancing paths of
# ``BinaryBalancedInitializer`` and the binarize / wide-matrix paths of
# ``PseudoDiagonalInitializer``.


class TestBinaryBalancedOddRowBranches:
    """Odd-row balancing paths of ``BinaryBalancedInitializer``.

    An even row count takes the ``n_work == rows`` fast path, so the
    row-deletion and global-column-balancing branches only run for an odd
    ``out_features``.
    """

    def test_odd_rows_with_global_balance(self) -> None:
        """Odd rows + ``balance_global`` exercises delete-row and column rebalancing."""
        init = BinaryBalancedInitializer(balance_global=True, seed=0)
        weight = torch.empty(7, 4)

        init.initialize(weight)

        assert weight.shape == (7, 4)
        assert torch.isfinite(weight).all()
        assert set(torch.unique(weight).tolist()) <= {-1.0, 1.0}

    def test_odd_rows_wide_with_global_balance(self) -> None:
        """A second odd-row shape covers the opposite column-count rebalance branch."""
        init = BinaryBalancedInitializer(balance_global=True, seed=0)
        weight = torch.empty(9, 6)

        init.initialize(weight)

        assert weight.shape == (9, 6)
        assert torch.isfinite(weight).all()

    def test_odd_rows_without_global_balance(self) -> None:
        """Odd rows with ``balance_global=False`` skips the global rebalance."""
        init = BinaryBalancedInitializer(balance_global=False, seed=0)
        weight = torch.empty(9, 5)

        init.initialize(weight)

        assert weight.shape == (9, 5)
        assert torch.isfinite(weight).all()

    def test_input_scaling_scales_entries(self) -> None:
        """A float ``input_scaling`` rescales the ``{-1, +1}`` entries."""
        init = BinaryBalancedInitializer(input_scaling=0.5, balance_global=True, seed=0)
        weight = torch.empty(7, 4)

        init.initialize(weight)

        assert set(torch.unique(weight).tolist()) <= {-0.5, 0.5}

    def test_preferred_coprime_step(self) -> None:
        """An explicit coprime ``step`` takes the preferred-step branch of ``_choose_step``."""
        init = BinaryBalancedInitializer(step=3, seed=0)
        weight = torch.empty(8, 4)

        init.initialize(weight)

        assert torch.isfinite(weight).all()


class TestPseudoDiagonalBranches:
    """Binarize and wide-matrix (``out_features < in_features``) branches."""

    def test_binarize_tall_matrix(self) -> None:
        """``binarize=True`` collapses each block to ``{-scaling, +scaling}``."""
        init = PseudoDiagonalInitializer(binarize=True, input_scaling=1.0, seed=0)
        weight = torch.empty(36, 4)

        init.initialize(weight)

        nonzero = weight[weight != 0]
        assert set(torch.unique(nonzero).tolist()) <= {-1.0, 1.0}

    def test_wide_matrix_out_lt_in(self) -> None:
        """``out_features < in_features`` routes through the per-row block branch."""
        init = PseudoDiagonalInitializer(seed=0)
        weight = torch.empty(3, 12)

        init.initialize(weight)

        assert weight.shape == (3, 12)
        assert torch.isfinite(weight).all()
        assert (weight != 0).any()

    def test_wide_matrix_binarize(self) -> None:
        """The wide path also honours ``binarize``."""
        init = PseudoDiagonalInitializer(binarize=True, input_scaling=1.0, seed=0)
        weight = torch.empty(3, 12)

        init.initialize(weight)

        nonzero = weight[weight != 0]
        assert set(torch.unique(nonzero).tolist()) <= {-1.0, 1.0}


class TestBinaryBalancedHelpers:
    """Deterministic coverage of the column-balancing helpers + degenerate shape.

    The registry sweep cannot reach the negative-sum column flip, the global
    rebalance branches (positive and negative total), or the zero-dimension
    early return, so these drive the helpers directly with crafted inputs.
    """

    def test_balance_columns_zero_sum_positive_and_negative(self) -> None:
        """Both the positive- and negative-sum flip branches drive a column to zero."""
        vw = np.array(
            [[1, -1], [1, -1], [-1, 1], [1, -1]],
            dtype=np.int8,
        )
        # column 0 sums to +2 (positive flip), column 1 sums to -2 (negative flip)
        BinaryBalancedInitializer._balance_columns_zero_sum(vw)

        assert int(vw[:, 0].sum()) == 0
        assert int(vw[:, 1].sum()) == 0

    def test_balance_global_positive_total(self) -> None:
        """A positive global total flips a ``+1`` column toward the target."""
        v = np.array([[1, 1, 1, -1]], dtype=np.int8)  # col sums [1,1,1,-1], total +2

        BinaryBalancedInitializer._balance_global_column_counts(v)

        assert int(v.sum()) == 0  # m even -> target 0

    def test_balance_global_negative_total(self) -> None:
        """A negative global total flips a ``-1`` column toward the target."""
        v = np.array([[-1, -1, -1, 1]], dtype=np.int8)  # col sums [-1,-1,-1,1], total -2

        BinaryBalancedInitializer._balance_global_column_counts(v)

        assert int(v.sum()) == 0

    def test_zero_dimension_weight_is_zeroed(self) -> None:
        """A zero-sized weight short-circuits to an all-zero return."""
        init = BinaryBalancedInitializer(seed=0)
        weight = torch.empty(0, 4)

        out = init.initialize(weight)

        assert out.shape == (0, 4)
