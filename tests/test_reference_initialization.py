"""The fixed quality recipes preserve the independently specified old draws."""

import pytest
import torch

from resdag import ESNCell
from tests._reference_initialization import sequential_uniform_initializers


@pytest.mark.parametrize("named_feedback", [False, True])
@pytest.mark.parametrize("seed", [0, 3])
@pytest.mark.parametrize("feedback_size", [1, 4])
def test_reference_recipe_matches_sequential_uniform_oracle(named_feedback, seed, feedback_size):
    generator = torch.Generator().manual_seed(seed)
    if named_feedback:
        expected_feedback = torch.empty(8, feedback_size).uniform_(
            -0.1, 0.1, generator=torch.Generator().manual_seed(seed)
        )
    else:
        expected_feedback = torch.empty(8, feedback_size).uniform_(-1, 1, generator=generator)
    expected_recurrent = torch.empty(8, 8).uniform_(-1, 1, generator=generator)
    expected_bias = torch.empty(8).uniform_(-1, 1, generator=generator)
    before = torch.get_rng_state().clone()
    cell = ESNCell(
        8,
        feedback_size,
        spectral_radius=None,
        seed=seed,
        feedback_initializer=("uniform", {"low": -0.1, "high": 0.1}) if named_feedback else None,
        **sequential_uniform_initializers(8, feedback_size, seed, named_feedback=named_feedback),
    )
    assert torch.equal(before, torch.get_rng_state())
    torch.testing.assert_close(cell.weight_feedback, expected_feedback, rtol=0, atol=0)
    torch.testing.assert_close(cell.weight_hh, expected_recurrent, rtol=0, atol=0)
    torch.testing.assert_close(cell.bias_h, expected_bias, rtol=0, atol=0)
