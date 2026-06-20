"""Unit tests for the RNG/seed helpers in ``resdag.utils.general``.

These back the seed-threading work in #136: ``coerce_seed_to_int`` reduces an
int/``torch.Generator``/None seed to the plain int the NumPy-backed builders
expect, and ``create_torch_generator`` yields a torch ``Generator`` for the
default ``nn.init`` weight/bias draws.
"""

import numpy as np
import pytest
import torch

from resdag.utils.general import coerce_seed_to_int, create_rng, create_torch_generator


class TestCoerceSeedToInt:
    """``coerce_seed_to_int`` normalizes int / Generator / None seeds."""

    def test_none_passthrough(self) -> None:
        assert coerce_seed_to_int(None) is None

    def test_int_passthrough(self) -> None:
        assert coerce_seed_to_int(42) == 42

    def test_generator_uses_initial_seed(self) -> None:
        """A generator collapses to the value it was seeded with."""
        gen = torch.Generator().manual_seed(1234)

        assert coerce_seed_to_int(gen) == 1234

    def test_equal_generators_collapse_to_same_int(self) -> None:
        """Two generators with the same manual_seed give the same int."""
        g1 = torch.Generator().manual_seed(77)
        g2 = torch.Generator().manual_seed(77)

        assert coerce_seed_to_int(g1) == coerce_seed_to_int(g2)

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError):
            coerce_seed_to_int("not-a-seed")  # type: ignore[arg-type]


class TestCreateTorchGenerator:
    """``create_torch_generator`` builds reproducible torch generators."""

    def test_int_seed_is_reproducible(self) -> None:
        """Same int -> identical draws."""
        a = torch.empty(8)
        b = torch.empty(8)
        torch.nn.init.uniform_(a, -1, 1, generator=create_torch_generator(5))
        torch.nn.init.uniform_(b, -1, 1, generator=create_torch_generator(5))

        assert torch.equal(a, b)

    def test_different_int_seeds_differ(self) -> None:
        a = torch.empty(8)
        b = torch.empty(8)
        torch.nn.init.uniform_(a, -1, 1, generator=create_torch_generator(5))
        torch.nn.init.uniform_(b, -1, 1, generator=create_torch_generator(6))

        assert not torch.equal(a, b)

    def test_generator_passthrough(self) -> None:
        """A torch.Generator is returned unchanged (caller's stream threads on)."""
        gen = torch.Generator().manual_seed(1)

        assert create_torch_generator(gen) is gen

    def test_none_derives_from_global_rng(self) -> None:
        """With ``seed=None`` the draw tracks ``torch.manual_seed``."""
        torch.manual_seed(0)
        a = torch.empty(8)
        torch.nn.init.uniform_(a, -1, 1, generator=create_torch_generator(None))

        torch.manual_seed(0)
        b = torch.empty(8)
        torch.nn.init.uniform_(b, -1, 1, generator=create_torch_generator(None))

        assert torch.equal(a, b)

    def test_int_seed_independent_of_global_rng(self) -> None:
        """An explicit int seed ignores prior global RNG state."""
        torch.manual_seed(999)
        _ = torch.randn(5)
        a = torch.empty(8)
        torch.nn.init.uniform_(a, -1, 1, generator=create_torch_generator(7))

        torch.manual_seed(123)
        _ = torch.randn(3)
        b = torch.empty(8)
        torch.nn.init.uniform_(b, -1, 1, generator=create_torch_generator(7))

        assert torch.equal(a, b)


class TestCreateRngAcceptsCoercedSeeds:
    """``create_rng`` keeps its contract; pairs with ``coerce_seed_to_int``."""

    def test_int_seed_reproducible(self) -> None:
        a = create_rng(42).random(4)
        b = create_rng(42).random(4)

        assert np.array_equal(a, b)

    def test_generator_passthrough(self) -> None:
        rng = np.random.default_rng(1)

        assert create_rng(rng) is rng

    def test_coerced_generator_drives_create_rng(self) -> None:
        """A torch.Generator -> int -> create_rng is deterministic end to end."""
        gen = torch.Generator().manual_seed(321)
        a = create_rng(coerce_seed_to_int(gen)).random(4)
        b = create_rng(coerce_seed_to_int(torch.Generator().manual_seed(321))).random(4)

        assert np.array_equal(a, b)
