"""Inference output assembly must preserve each prediction immediately."""

from contextlib import nullcontext

import pytest
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


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("no_grad", [True, False])
def test_driven_rollout_matches_original_slices_and_gradients(batch, no_grad) -> None:
    """Prepared views preserve noncontiguous frames, heads and recurrent gradients."""
    generator = torch.Generator().manual_seed(803)
    seed = torch.randn(batch, 1, 2, generator=generator, dtype=torch.float64)
    # Both time and feature axes have nontrivial strides.
    drivers = tuple(
        torch.randn(batch, 18, 4, generator=generator, dtype=torch.float64)[:, ::2, ::2]
        .detach()
        .requires_grad_(not no_grad)
        for _ in range(2)
    )
    seed.requires_grad_(not no_grad)
    initial = torch.randn(batch, 1, 2, generator=generator, dtype=torch.float64)
    initial.requires_grad_(not no_grad)

    def step(inputs, states):
        feedback, first, second = inputs
        assert feedback.shape == first.shape == second.shape == (batch, 1, 2)
        state = torch.tanh(states[0] * 0.2 + first * 0.3 - second * 0.1)
        prediction = feedback * 0.4 + state
        return (prediction, state.sum(dim=-1, keepdim=True)), [state]

    flat = FlatStep(step, [], "", n_inputs=3, n_outputs=2)
    context = torch.no_grad() if no_grad else nullcontext()
    with context:
        actual, actual_state = rollout(flat, seed, drivers, [initial], 7, [2, 1], no_grad=no_grad)
        feedback, states = seed, [initial]
        frames = [[], []]
        for t in range(7):
            outputs, states = step(
                (feedback, *(driver[:, t : t + 1, :] for driver in drivers)), states
            )
            for head, value in zip(frames, outputs):
                head.append(value)
            feedback = outputs[0]
        expected = [torch.cat(head, dim=1) for head in frames]

    for value, reference in zip(actual + actual_state, expected + states):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
    if not no_grad:
        variables = (seed, *drivers, initial)
        actual_grad = torch.autograd.grad(
            sum(value.square().sum() for value in actual + actual_state), variables
        )
        expected_grad = torch.autograd.grad(
            sum(value.square().sum() for value in expected + states), variables
        )
        for value, reference in zip(actual_grad, expected_grad):
            torch.testing.assert_close(value, reference, rtol=1e-14, atol=1e-14)
