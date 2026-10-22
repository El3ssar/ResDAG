"""Named seeds are independent of draw order, process hash salt and global RNGs."""

import os
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

import resdag as rd
from resdag.utils import SeedStreams


def test_versioned_seed_derivation_has_fixed_cross_platform_vectors():
    streams = SeedStreams(np.int64(42))
    assert streams.algorithm == "blake2b-v1"
    assert streams.master_seed == 42
    assert {key: streams[key] for key in ("fast", "slow", "branch/input", "α")} == {
        "fast": 111052144843092782,
        "slow": 74971046297675534,
        "branch/input": 4042400974837766525,
        "α": 8595932717000606184,
    }
    assert rd.SeedStreams is SeedStreams


def test_component_additions_and_order_do_not_change_existing_reservoir_weights():
    streams = SeedStreams(19)
    fast = rd.ESNLayer(6, feedback_size=2, seed=streams["fast"], dtype=torch.float64)
    rd.ESNLayer(11, feedback_size=3, seed=streams["new_branch"])
    reordered = SeedStreams(19)
    reordered["slow"]
    again = rd.ESNLayer(6, feedback_size=2, seed=reordered["fast"], dtype=torch.float64)
    for name, tensor in fast.state_dict().items():
        torch.testing.assert_close(tensor, again.state_dict()[name], rtol=0, atol=0)
    assert fast is not again
    assert all(a is not b for a, b in zip(fast.parameters(), again.parameters()))
    assert fast.state is again.state is None


def test_seed_lookup_does_not_consume_or_seed_python_numpy_or_torch_rng():
    python_before, numpy_before, torch_before = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state().clone(),
    )
    streams = SeedStreams(42)
    for name in ("fast", "slow", "a", "b", "fast"):
        assert streams.seed_for(name) == streams[name]
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0] and numpy_after[2:] == numpy_before[2:]
    np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
    assert torch.equal(torch.get_rng_state(), torch_before)


@pytest.mark.parametrize("master", [0, -1, 2**100])
def test_derived_seeds_work_with_standard_numpy_and_torch_generators(master):
    seed = SeedStreams(master)["weights"]
    assert 0 <= seed < 2**63
    left = np.random.default_rng(seed).normal(size=4)
    right = np.random.default_rng(seed).normal(size=4)
    np.testing.assert_array_equal(left, right)
    a, b = (torch.Generator().manual_seed(seed) for _ in range(2))
    torch.testing.assert_close(torch.randn(4, generator=a), torch.randn(4, generator=b))


@pytest.mark.parametrize("invalid", [True, None, 1.5, "42", torch.tensor(42)])
def test_noninteger_master_seeds_fail_clearly(invalid):
    with pytest.raises(TypeError, match="must be an integer"):
        SeedStreams(invalid)


@pytest.mark.parametrize("invalid", [None, 4, "", "  "])
def test_missing_or_blank_names_fail_clearly(invalid):
    error = TypeError if not isinstance(invalid, str) else ValueError
    with pytest.raises(error, match="component name"):
        SeedStreams(42)[invalid]


def test_process_hash_randomization_does_not_change_seeds():
    source = "from resdag import SeedStreams; print(SeedStreams(42)['fast'])"
    results = [
        subprocess.check_output(
            [sys.executable, "-c", source],
            env={**os.environ, "PYTHONHASHSEED": salt},
            text=True,
            timeout=30,
        ).strip()
        for salt in ("1", "999")
    ]
    assert results == ["111052144843092782"] * 2
