"""Input/feedback initializer contracts (resdag.init.input_feedback).

Pins down:

- the initializer registry (``get_input_feedback`` / ``show_input_initializers``
  / ``register_input_feedback``),
- ``FunctionInitializer``: build-style and in-place initializer callables,
- callable initializer specs through ``resolve_initializer``,
- custom initializer objects driving an ESNLayer on every device.
"""

import pytest
import torch

from resdag.init.input_feedback import (
    BinaryBalancedInitializer,
    ChainOfNeuronsInputInitializer,
    ChebyshevInitializer,
    ChessboardInitializer,
    DendrocycleInputInitializer,
    FunctionInitializer,
    OppositeAnchorsInitializer,
    PseudoDiagonalInitializer,
    RandomBinaryInitializer,
    RandomInputInitializer,
    RingWindowInputInitializer,
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
    pytest.param(
        PseudoDiagonalInitializer,
        {"input_scaling": 1.0, "binarize": False},
        id="pseudo_diagonal",
    ),
    pytest.param(
        DendrocycleInputInitializer,
        {"c": 0.2, "input_scaling": 0.5},
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
        init = OppositeAnchorsInitializer(gain=1.0)
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
            OppositeAnchorsInitializer(gain=1.0).initialize(weight)

            for i in range(m):
                for j in range(i + 1, m):
                    assert not torch.allclose(
                        weight[:, i], weight[:, j]
                    ), f"columns {i} and {j} are identical (n={n}, m={m})"

    def test_in_features_exceeds_reservoir_size_raises(self) -> None:
        """``in_features > reservoir_size`` raises a clear ``ValueError``."""
        init = OppositeAnchorsInitializer(gain=1.0)
        weight = torch.empty(6, 7)  # (reservoir_size=6, in_features=7)

        with pytest.raises(ValueError, match="in_features <= reservoir_size"):
            init.initialize(weight)

    def test_single_channel_full_ring(self) -> None:
        """A single channel (``m=1``) still produces a valid bipolar column."""
        init = OppositeAnchorsInitializer(gain=1.0)
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
        {"gain": 1.0},  # gain / sqrt(2) is already irrational
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
