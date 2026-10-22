"""Exposing an intermediate fitted head supports staged custom DAG training."""

import copy

import torch

import resdag as rd


def test_named_intermediate_head_trains_before_downstream_reservoir_features():
    x = rd.reservoir_input(1, dtype=torch.float64)
    internal = rd.RidgeReadoutLayer(1, 1, alpha=0, name="internal_fit").double()
    intermediate = internal(x)
    reservoir = rd.ESNLayer(5, feedback_size=1, seed=23, dtype=torch.float64)
    manual_reservoir = copy.deepcopy(reservoir)
    features = rd.Concatenate()(reservoir(intermediate), intermediate)
    final = rd.RidgeReadoutLayer(6, 1, alpha=0.01, name="outer_fit").double()
    model = rd.Model(
        inputs={"state": x},
        outputs={"intermediate": intermediate, "prediction": final(features)},
        feedback=[rd.Feedback("prediction", "state")],
        preprocessing="off",
        washout=3,
    )
    data = torch.sin(torch.linspace(0, 8, 45, dtype=torch.float64)).reshape(1, -1, 1)
    observed, target = data[:, :-1], data[:, 1:]
    mid_target = 3 * observed + 2
    model.fit(observed, {"intermediate": mid_target, "prediction": target})
    torch.testing.assert_close(internal.weight, torch.tensor([[3.0]]).double(), atol=1e-12, rtol=0)
    torch.testing.assert_close(internal.bias, torch.tensor([2.0]).double(), atol=1e-12, rtol=0)

    # Independent downstream teacher forcing: use the learned physical signal
    # directly, then solve a centered ridge system without the DAG trainer.
    manual_reservoir.reset_state()
    states = manual_reservoir(mid_target)
    matrix = torch.cat((states, mid_target), -1)[:, 3:].reshape(-1, 6)
    truth = target[:, 3:].reshape(-1, 1)
    mean_x, mean_y = matrix.mean(0), truth.mean(0)
    centered = matrix - mean_x
    coefficients = torch.linalg.solve(
        centered.T @ centered + 0.01 * torch.eye(6, dtype=torch.float64),
        centered.T @ (truth - mean_y),
    )
    intercept = mean_y - mean_x @ coefficients
    torch.testing.assert_close(final.weight, coefficients.T, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(final.bias, intercept, rtol=1e-10, atol=1e-10)
    result = model.forecast(history=observed[:, -5:], horizon=4)
    assert list(result) == ["intermediate", "prediction"]
    assert all(
        value.shape == (1, 4, 1) and torch.isfinite(value).all() for value in result.values()
    )
