"""Finite-member aggregation and ensemble interfaces fail predictably."""

import numpy as np
import pytest
import torch
from torch import nn

import resdag as rd


@pytest.mark.parametrize("method", ["z_score", "iqr"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_nonfinite_members_are_excluded_from_statistics_and_mean(method, dtype):
    members = torch.tensor([1.0, 1.0, 1.0, 1.0, 100.0, float("inf"), float("nan")], dtype=dtype)
    result = rd.OutliersFilteredMean(method=method)(members.reshape(-1, 1, 1, 1))
    assert result.dtype == dtype
    torch.testing.assert_close(result, torch.ones_like(result), rtol=0, atol=0)


@pytest.mark.parametrize("method", ["z_score", "iqr"])
def test_all_invalid_reports_affected_position_and_member_count(method):
    members = torch.ones(3, 2, 4, 1)
    members[:, 1, 2] = float("nan")
    with pytest.raises(ValueError, match=r"All 3 ensemble members.*\[1, 2\]"):
        rd.OutliersFilteredMean(method=method)(members)


@pytest.mark.parametrize("method", ["z_score", "iqr"])
def test_invalid_members_receive_zero_gradient(method):
    members = torch.tensor([1.0, 1.0, 1.0, float("inf")], dtype=torch.float64, requires_grad=True)
    result = rd.OutliersFilteredMean(method=method)(members.reshape(-1, 1, 1, 1))
    result.sum().backward()
    torch.testing.assert_close(
        members.grad, torch.tensor([1 / 3, 1 / 3, 1 / 3, 0.0], dtype=torch.float64)
    )


@pytest.mark.parametrize("method", ["z_score", "iqr"])
def test_finite_results_match_independent_numpy_statistics(method):
    samples = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 90.0])
    if method == "z_score":
        median = np.sort(samples)[(len(samples) - 1) // 2]
        deviation = np.abs(samples - median)
        mad = np.sort(deviation)[(len(samples) - 1) // 2]
        mask = deviation / (mad / 0.6745) < 3.5
    else:
        q1, q3 = np.quantile(samples, [0.25, 0.75])
        mask = (samples >= q1 - 1.5 * (q3 - q1)) & (samples <= q3 + 1.5 * (q3 - q1))
    actual = rd.OutliersFilteredMean(method=method)(torch.tensor(samples).reshape(-1, 1, 1, 1))
    torch.testing.assert_close(actual, torch.full_like(actual, samples[mask].mean()))


@pytest.mark.parametrize("method", ["z_score", "iqr"])
def test_each_position_can_have_a_different_number_of_finite_members(method):
    nan = float("nan")
    members = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [nan, 3.0, 1.0, 2.0],
            [nan, nan, 2.0, 3.0],
            [nan, nan, 3.0, 4.0],
            [nan, nan, nan, 100.0],
        ],
        dtype=torch.float64,
    ).reshape(5, 1, 4, 1)
    expected = [1.0, 1.0, 1.0, 2.5] if method == "z_score" else [1.0, 2.0, 1.75, 2.5]
    actual = rd.OutliersFilteredMean(method=method)(members)
    torch.testing.assert_close(actual.flatten(), torch.tensor(expected, dtype=torch.float64))


@pytest.mark.parametrize("method", ["z_score", "iqr"])
def test_fallback_averages_finite_members_only(method):
    members = torch.tensor([1.0, 3.0, 5.0, float("nan")]).reshape(-1, 1, 1, 1)
    result = rd.OutliersFilteredMean(method=method, threshold=0)(members)
    # z-score threshold zero excludes every member; IQR threshold zero keeps
    # the central member. Both should produce the same finite answer here.
    torch.testing.assert_close(result, torch.full_like(result, 3))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
@pytest.mark.parametrize("method", ["z_score", "iqr"])
def test_robust_aggregation_runs_on_mps(method):
    members = torch.tensor([1.0, 1.0, 1.0, float("inf")], device="mps").reshape(-1, 1, 1, 1)
    actual = rd.OutliersFilteredMean(method=method)(members)
    assert actual.device.type == "mps"
    torch.testing.assert_close(actual.cpu(), torch.ones(1, 1, 1))


@pytest.mark.parametrize(
    "second_kwargs", [{"feedback_size": 2, "output_size": 2}, {"input_size": 2}]
)
def test_ensemble_rejects_mismatched_input_interfaces_before_execution(second_kwargs):
    first = rd.classic_esn(4, 1, 1)
    kwargs = {"feedback_size": 1, "output_size": 1, **second_kwargs}
    second = rd.classic_esn(reservoir_size=4, **kwargs)
    with pytest.raises(ValueError, match="input feature widths"):
        rd.CoupledEnsembleESNModel([first, second])
    assert first.get_state() == second.get_state() == {}


def test_ensemble_rejects_mixed_precision_before_execution():
    with pytest.raises(ValueError, match="all ensemble members must share"):
        rd.CoupledEnsembleESNModel([rd.classic_esn(4, 1, 1), rd.classic_esn(4, 1, 1).double()])


@pytest.mark.parametrize("wrong_type", [False, True])
def test_custom_aggregator_output_contract(wrong_type):
    class BadAggregator(nn.Module):
        def forward(self, values):
            return [] if wrong_type else values.mean(dim=0).mean(dim=1)

    ensemble = rd.CoupledEnsembleESNModel([rd.classic_esn(4, 1, 1)], BadAggregator())
    with pytest.raises(TypeError if wrong_type else ValueError, match="aggregator must"):
        ensemble(torch.ones(2, 3, 1))
