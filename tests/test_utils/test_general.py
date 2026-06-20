"""Unit tests for the RNG/seed helpers in ``resdag.utils.general``.

These back the seed-threading work in #136: ``coerce_seed_to_int`` reduces an
int/``torch.Generator``/None seed to the plain int the NumPy-backed builders
expect, and ``create_torch_generator`` yields a torch ``Generator`` for the
default ``nn.init`` weight/bias draws.

They also cover the whole-program reproducibility helpers from #243:
``seed_everything`` (torch + NumPy + ``random``) and ``resolve_device``.
"""

import random

import numpy as np
import pytest
import torch

from resdag.utils import resolve_device as resolve_device_public
from resdag.utils import seed_everything as seed_everything_public
from resdag.utils.general import (
    coerce_seed_to_int,
    create_rng,
    create_torch_generator,
    resolve_device,
    seed_everything,
)


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


class TestSeedEverything:
    """``seed_everything`` seeds torch + NumPy + ``random`` and yields the seed."""

    def test_returns_seed(self) -> None:
        """The applied seed is returned for logging/threading."""
        assert seed_everything(123) == 123

    def test_default_seed(self) -> None:
        """The default seed is 42."""
        assert seed_everything() == 42

    def test_torch_reproducible(self) -> None:
        """Same seed -> identical torch global draws."""
        seed_everything(0)
        a = torch.randn(5)
        seed_everything(0)
        b = torch.randn(5)

        assert torch.equal(a, b)

    def test_numpy_reproducible(self) -> None:
        """Same seed -> identical NumPy legacy global draws."""
        seed_everything(7)
        a = np.random.rand(5)
        seed_everything(7)
        b = np.random.rand(5)

        assert np.array_equal(a, b)

    def test_random_reproducible(self) -> None:
        """Same seed -> identical ``random`` draws."""
        seed_everything(99)
        a = [random.random() for _ in range(5)]
        seed_everything(99)
        b = [random.random() for _ in range(5)]

        assert a == b

    def test_different_seeds_differ(self) -> None:
        """Different seeds yield different torch draws."""
        seed_everything(1)
        a = torch.randn(5)
        seed_everything(2)
        b = torch.randn(5)

        assert not torch.equal(a, b)

    def test_all_streams_at_once(self) -> None:
        """A single seed pins torch, NumPy, and ``random`` simultaneously."""
        seed_everything(2024)
        a = (torch.randn(3), np.random.rand(3), random.random())
        seed_everything(2024)
        b = (torch.randn(3), np.random.rand(3), random.random())

        assert torch.equal(a[0], b[0])
        assert np.array_equal(a[1], b[1])
        assert a[2] == b[2]

    def test_deterministic_flag_sets_cudnn(self) -> None:
        """``deterministic=True`` enables the deterministic cuDNN flags."""
        try:
            seed_everything(0, deterministic=True)
            assert torch.backends.cudnn.deterministic is True
            assert torch.backends.cudnn.benchmark is False
            assert torch.are_deterministic_algorithms_enabled() is True
        finally:
            # Reset so the global flag does not leak into other tests.
            torch.use_deterministic_algorithms(False)

    def test_exported_from_utils(self) -> None:
        """The public ``resdag.utils`` export is the same callable."""
        assert seed_everything_public is seed_everything


class TestResolveDevice:
    """``resolve_device`` maps a spec to a concrete ``torch.device``."""

    def test_cpu_string(self) -> None:
        assert resolve_device("cpu") == torch.device("cpu")

    def test_torch_device_passthrough(self) -> None:
        dev = torch.device("cpu")

        assert resolve_device(dev) is dev

    def test_auto_returns_valid_backend(self) -> None:
        """``'auto'`` resolves to one of cuda/mps/cpu by availability."""
        dev = resolve_device("auto")

        assert dev.type in {"cuda", "mps", "cpu"}

    def test_none_behaves_like_auto(self) -> None:
        assert resolve_device(None) == resolve_device("auto")

    def test_auto_prefers_available_accelerator(self) -> None:
        """``'auto'`` follows the cuda -> mps -> cpu priority order."""
        dev = resolve_device("auto")
        if torch.cuda.is_available():
            assert dev.type == "cuda"
        elif torch.backends.mps.is_available():
            assert dev.type == "mps"
        else:
            assert dev.type == "cpu"

    def test_explicit_cuda_index_passthrough_when_available(self) -> None:
        """An indexed cuda spec parses through when CUDA is present."""
        dev = resolve_device("cuda:0")
        if torch.cuda.is_available():
            assert dev.type == "cuda"
            assert dev.index == 0
        else:
            assert dev.type == "cpu"

    def test_unavailable_cuda_falls_back_to_cpu(self) -> None:
        """A cuda spec falls back to cpu when CUDA is unavailable."""
        dev = resolve_device("cuda")
        if not torch.cuda.is_available():
            assert dev.type == "cpu"
        else:
            assert dev.type == "cuda"

    def test_unavailable_mps_falls_back_to_cpu(self) -> None:
        """An mps spec falls back to cpu when MPS is unavailable."""
        dev = resolve_device("mps")
        if not torch.backends.mps.is_available():
            assert dev.type == "cpu"
        else:
            assert dev.type == "mps"

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError):
            resolve_device(123)  # type: ignore[arg-type]

    def test_exported_from_utils(self) -> None:
        """The public ``resdag.utils`` export is the same callable."""
        assert resolve_device_public is resolve_device
