"""Inference output assembly must preserve each prediction immediately."""

import torch

from resdag.core._flat_inference import FlatStep, rollout


def test_inference_rollout_copies_predictions_from_reused_output_buffer() -> None:
    # Custom inference layers may reuse a non-contiguous output buffer. Keeping
    # references until a later concatenation would overwrite earlier predictions.
    buffer = torch.empty(2, 1, 3).permute(2, 1, 0)

    def step(inputs, states):
        buffer.copy_(inputs[0] + 1)
        return (buffer,), states

    flat = FlatStep(step, [], "", n_inputs=1, n_outputs=1)
    seed = torch.zeros(3, 1, 2)
    with torch.no_grad():
        outputs, states = rollout(flat, seed, (), [], 7, [2], no_grad=True)
    expected = torch.arange(1, 8, dtype=seed.dtype).view(1, 7, 1).expand(3, 7, 2)
    torch.testing.assert_close(outputs[0], expected, rtol=0, atol=0)
    assert states == []
