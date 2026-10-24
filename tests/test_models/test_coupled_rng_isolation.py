"""Seeded ensemble construction preserves caller RNG streams on active devices."""

import json
import subprocess
import sys
import textwrap
from contextlib import ExitStack

import pytest
import torch

from resdag.models import classic_esn, coupled_ensemble_esn


def _cpu_factory():
    return classic_esn(4, 1, 1, readout="ridge")


def test_cpu_initialized_weights_match_the_previous_seed_schedule_bit_for_bit():
    with ExitStack() as cleanup:
        cleanup.enter_context(torch.random.fork_rng(devices=[]))
        if torch.backends.mps.is_available():
            cleanup.enter_context(torch.random.fork_rng(devices=[0], device_type="mps"))
        expected = []
        for index in range(3):
            torch.default_generator.manual_seed(31 + index)
            expected.append(_cpu_factory().state_dict())
        actual = coupled_ensemble_esn(3, model_factory=_cpu_factory, seed=31)
        for member, reference in zip(actual.models, expected):
            for name, value in member.state_dict().items():
                torch.testing.assert_close(value, reference[name], rtol=0, atol=0)


@pytest.mark.parametrize("raises", [False, True])
def test_cpu_state_and_following_draw_are_restored_even_if_factory_raises(raises):
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(991)
        before = torch.get_rng_state().clone()
        expected = torch.randn(8)
        torch.set_rng_state(before)

        def factory():
            torch.randn(5)
            if raises:
                raise RuntimeError("intentional construction failure")
            return _cpu_factory()

        if raises:
            with pytest.raises(RuntimeError, match="intentional construction failure"):
                coupled_ensemble_esn(2, model_factory=factory, seed=17)
        else:
            coupled_ensemble_esn(2, model_factory=factory, seed=17)
        assert torch.equal(torch.get_rng_state(), before)
        assert torch.equal(torch.randn(8), expected)


def test_subseeds_are_forwarded_and_cpu_draws_keep_the_existing_sequence():
    seen = []

    def factory(seed=None):
        seen.append((seed, torch.rand(4)))
        return _cpu_factory()

    coupled_ensemble_esn(3, model_factory=factory, seed=42)
    for index, (received_seed, draw) in enumerate(seen):
        assert received_seed == 42 + index
        generator = torch.Generator().manual_seed(42 + index)
        torch.testing.assert_close(draw, torch.rand(4, generator=generator), rtol=0, atol=0)


def test_cpu_factory_does_not_initialize_or_broadcast_to_unused_accelerators(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("CPU construction must not touch unused accelerator RNGs")

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", unexpected)
    monkeypatch.setattr(torch.cuda, "get_rng_state", unexpected)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", unexpected)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.mps, "get_rng_state", unexpected)
    monkeypatch.setattr(torch.mps, "manual_seed", unexpected)
    monkeypatch.setattr(torch, "manual_seed", unexpected)
    coupled_ensemble_esn(2, model_factory=_cpu_factory, seed=17)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Native MPS unavailable")
@pytest.mark.parametrize("raises", [False, True])
def test_fresh_process_direct_mps_draw_is_isolated_before_python_rng_api_use(raises):
    script = textwrap.dedent("""
        import json
        import sys
        import torch
        from resdag.models import classic_esn, coupled_ensemble_esn

        raises = sys.argv[1] == "True"
        torch.rand(4, device="mps")
        assert torch.mps._default_mps_generator is None
        # Only the independent oracle accesses the native generator directly:
        # a public get_rng_state would populate the cache and mask the defect.
        generator = torch._C._mps_get_default_generator()
        before = generator.get_state().clone()
        expected_next = torch.rand(8, device="mps").cpu()
        generator.set_state(before)
        assert torch.mps._default_mps_generator is None
        seen = []

        def factory():
            seen.append(torch.rand(7, device="mps").cpu())
            if raises:
                raise RuntimeError("intentional native MPS failure")
            return classic_esn(4, 1, 1, readout="ridge")

        def construct():
            if raises:
                try:
                    coupled_ensemble_esn(2, model_factory=factory, seed=31)
                except RuntimeError as error:
                    assert "intentional native MPS failure" in str(error)
                else:
                    raise AssertionError("factory must fail")
            else:
                coupled_ensemble_esn(2, model_factory=factory, seed=31)
            assert torch.equal(generator.get_state(), before)

        construct()
        first = list(seen)
        seen.clear()
        construct()
        assert len(first) == len(seen)
        assert all(torch.equal(a, b) for a, b in zip(first, seen))
        assert torch.equal(torch.rand(8, device="mps").cpu(), expected_next)
        print(json.dumps({"restored": True, "repeatable": True, "raises": raises}))
        """)
    result = subprocess.run(
        [sys.executable, "-c", script, str(raises)],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert json.loads(result.stdout) == {"restored": True, "repeatable": True, "raises": raises}


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Native MPS unavailable")
@pytest.mark.parametrize("raises", [False, True])
def test_native_mps_state_and_following_draw_are_restored(raises):
    with torch.random.fork_rng(devices=[0], device_type="mps"):
        torch.mps.manual_seed(661)
        torch.rand(3, device="mps")
        before = torch.mps.get_rng_state().clone()
        expected = torch.rand(8, device="mps").cpu()
        torch.mps.set_rng_state(before)

        def factory():
            torch.rand(5, device="mps")
            if raises:
                raise RuntimeError("intentional MPS construction failure")
            return _cpu_factory()

        if raises:
            with pytest.raises(RuntimeError, match="intentional MPS construction failure"):
                coupled_ensemble_esn(2, model_factory=factory, seed=17)
        else:
            first = coupled_ensemble_esn(2, model_factory=factory, seed=17)
            assert len(first.models) == 2
        assert torch.equal(torch.mps.get_rng_state(), before)
        assert torch.equal(torch.rand(8, device="mps").cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("raises", [False, True])
def test_cuda_state_and_following_draw_are_restored(raises):
    torch.rand(1, device="cuda")
    devices = list(range(torch.cuda.device_count()))
    with torch.random.fork_rng(devices=devices):
        torch.cuda.manual_seed_all(661)
        before = torch.cuda.get_rng_state_all()
        expected = [torch.rand(8, device=index).cpu() for index in devices]
        torch.cuda.set_rng_state_all(before)

        def factory():
            for index in devices:
                torch.rand(5, device=index)
            if raises:
                raise RuntimeError("intentional CUDA construction failure")
            return _cpu_factory()

        if raises:
            with pytest.raises(RuntimeError, match="intentional CUDA construction failure"):
                coupled_ensemble_esn(2, model_factory=factory, seed=17)
        else:
            coupled_ensemble_esn(2, model_factory=factory, seed=17)
        for actual, saved in zip(torch.cuda.get_rng_state_all(), before):
            assert torch.equal(actual, saved)
        for index, reference in zip(devices, expected):
            assert torch.equal(torch.rand(8, device=index).cpu(), reference)
