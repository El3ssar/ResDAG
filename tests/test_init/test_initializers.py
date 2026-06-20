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
    ChebyshevInitializer,
    DendrocycleInputInitializer,
    FunctionInitializer,
    PseudoDiagonalInitializer,
    RandomBinaryInitializer,
    RandomInputInitializer,
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
