"""Analytic, fixed-scale forecast scoring oracles independent of HPO engines."""

import math
from dataclasses import FrozenInstanceError

import pytest
import torch

from resdag.hpo._tuning_scores import score_forecast


def _case(*, dtype=torch.float64):
    truth = {"state": torch.zeros(8, 2, dtype=dtype)}
    return {"state": truth["state"].clone()}, truth, {"state": torch.ones(2, dtype=dtype)}


def test_weakest_output_default_and_dimension_weighted_error_have_analytic_values():
    truth = {"position": torch.zeros(500, 1), "velocity": torch.zeros(500, 2)}
    predicted = {name: value.clone() for name, value in truth.items()}
    predicted["position"][400:] = 1
    predicted["velocity"][100:] = 1
    scales = {"position": torch.ones(1), "velocity": torch.ones(2)}

    result = score_forecast(predicted, truth, scales)

    assert result.score == 100
    assert result.head_horizons == {"position": 400, "velocity": 100}
    assert result.head_errors == pytest.approx(
        {"position": math.sqrt(0.2), "velocity": math.sqrt(0.8)}
    )
    assert result.nrmse == pytest.approx((math.sqrt(0.2) + 2 * math.sqrt(0.8)) / 3)
    assert score_forecast(predicted, truth, scales, aggregation="mean").score == 250


def test_conventional_even_batch_median_averages_both_middle_horizons():
    truth = {"state": torch.zeros(4, 8, 1)}
    predicted = {"state": torch.ones_like(truth["state"])}
    for batch, horizon in enumerate([0, 1, 4, 7]):
        predicted["state"][batch, :horizon] = 0
    result = score_forecast(predicted, truth, {"state": torch.ones(1)})
    assert result.score == 2.5
    assert result.head_horizons == {"state": 2.5}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_threshold_is_strict_and_recovery_does_not_restart_contiguous_horizon(dtype):
    threshold = torch.tensor(0.2, dtype=dtype)
    below = torch.nextafter(threshold, torch.zeros_like(threshold))
    truth = {"state": torch.zeros(5, 1, dtype=dtype)}
    predicted = {"state": torch.stack([below, below, threshold, below, below]).reshape(5, 1)}
    result = score_forecast(predicted, truth, {"state": torch.ones(1, dtype=dtype)})
    assert result.score == 2


@pytest.mark.parametrize("accurate", [True, False])
def test_all_accurate_or_first_sample_inaccurate(accurate):
    predictions, truth, scales = _case()
    if not accurate:
        predictions["state"][0] = 1
    result = score_forecast(predictions, truth, scales)
    assert result.score == (8 if accurate else 0)


def test_supplied_training_scale_is_used_even_for_constant_validation_signal():
    truth = {"state": torch.full((6, 2), 100.0, dtype=torch.float64)}
    predictions = {"state": truth["state"] + torch.tensor([0.3, 0.6])}
    scales = {"state": torch.tensor([2.0, 4.0], dtype=torch.float64)}
    result = score_forecast(predictions, truth, scales)
    assert result.score == 6
    assert result.nrmse == pytest.approx(0.15)


def test_growing_prefix_and_future_target_variance_cannot_reestimate_score_scale():
    truth = {"state": torch.zeros(12, 1, dtype=torch.float64)}
    predicted = {"state": torch.full((12, 1), 0.15, dtype=torch.float64)}
    predicted["state"][5:] = 0.3
    scales = {"state": torch.ones(1, dtype=torch.float64)}
    assert (
        score_forecast(
            {"state": predicted["state"][:4]}, {"state": truth["state"][:4]}, scales
        ).score
        == 4
    )
    assert score_forecast(predicted, truth, scales).score == 5
    # Translate both signals after the early crossing: future target variance
    # changes dramatically while every original-unit prediction error stays fixed.
    translation = torch.arange(12, dtype=torch.float64).reshape(12, 1) * 1000
    shifted = score_forecast(
        {"state": predicted["state"] + translation},
        {"state": truth["state"] + translation},
        scales,
    )
    assert shifted.score == 5
    assert shifted.nrmse == pytest.approx(score_forecast(predicted, truth, scales).nrmse)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rms_is_stable_when_naively_squaring_errors_would_overflow(dtype):
    magnitude = 1e30 if dtype == torch.float32 else 1e300
    truth = {"state": torch.zeros(4, 2, dtype=dtype)}
    predictions = {"state": torch.full((4, 2), magnitude, dtype=dtype)}
    result = score_forecast(predictions, truth, {"state": torch.ones(2, dtype=dtype)})
    assert result.score == 0
    assert math.isfinite(result.nrmse)
    assert result.nrmse == pytest.approx(magnitude)


def test_normalization_avoids_overflowing_a_representable_scaled_difference():
    predictions = {"state": torch.full((3, 2), 1e308, dtype=torch.float64)}
    truth = {"state": -predictions["state"]}
    result = score_forecast(
        predictions, truth, {"state": torch.full((2,), 1e308, dtype=torch.float64)}
    )
    assert result.score == 0
    assert result.nrmse == 2


def test_equal_large_values_remain_exact_with_tiny_scale_and_small_rms_does_not_underflow():
    predictions = {"state": torch.full((3, 2), 1e308, dtype=torch.float64)}
    scales = {"state": torch.full((2,), 1e-308, dtype=torch.float64)}
    assert score_forecast(predictions, predictions, scales).nrmse == 0
    predictions = {"state": torch.full((3, 2), 1e-300, dtype=torch.float64)}
    truth = {"state": torch.zeros_like(predictions["state"])}
    scales = {"state": torch.ones(2, dtype=torch.float64)}
    assert score_forecast(predictions, truth, scales).nrmse == pytest.approx(
        1e-300, rel=1e-12, abs=0
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("heads", [2, 3, 5, 7])
def test_weighted_head_error_avoids_overflowing_sum_of_large_finite_head_errors(dtype, heads):
    maximum = torch.finfo(dtype).max
    truth = {str(index): torch.zeros(2, 1, dtype=dtype) for index in range(heads)}
    predictions = {name: torch.full_like(value, maximum) for name, value in truth.items()}
    scales = {name: torch.ones(1, dtype=dtype) for name in truth}
    result = score_forecast(predictions, truth, scales)
    assert math.isfinite(result.nrmse)
    assert result.nrmse == pytest.approx(maximum)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_scoring_does_not_track_gradients_or_mutate_inputs(dtype):
    predictions = {"state": torch.full((4, 1), 0.125, dtype=dtype, requires_grad=True)}
    truth = {"state": torch.zeros(4, 1, dtype=dtype)}
    scales = {"state": torch.ones(1, dtype=dtype)}
    original = predictions["state"].detach().clone()
    result = score_forecast(predictions, truth, scales)
    assert result.score == 4
    assert result.nrmse == 0.125
    assert predictions["state"].grad is None
    assert torch.equal(predictions["state"], original)
    assert torch.is_grad_enabled()
    with pytest.raises(FrozenInstanceError):
        result.score = 0


@pytest.mark.parametrize("shape", [(), (2,), (1, 2), (1, 1, 2)])
def test_fixed_scale_broadcast_forms_are_supported(shape):
    predictions, truth, _ = _case()
    result = score_forecast(predictions, truth, {"state": torch.ones(shape)})
    assert result.score == 8


@pytest.mark.parametrize("shape", [(8, 2), (2, 1, 2), (1, 1, 1, 2), (1,), (0,)])
def test_time_or_batch_dependent_or_wrong_width_scales_are_rejected(shape):
    predictions, truth, _ = _case()
    with pytest.raises(ValueError, match="fixed feature vector"):
        score_forecast(predictions, truth, {"state": torch.ones(shape)})


@pytest.mark.parametrize("container", ["predictions", "truth", "scales"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_data_or_scale_is_rejected(container, value):
    predictions, truth, scales = _case()
    {"predictions": predictions, "truth": truth, "scales": scales}[container]["state"].flatten()[
        0
    ] = value
    with pytest.raises(ValueError, match="finite"):
        score_forecast(predictions, truth, scales)


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_nonpositive_scale_is_rejected(value):
    predictions, truth, scales = _case()
    scales["state"][0] = value
    with pytest.raises(ValueError, match="positive finite scales"):
        score_forecast(predictions, truth, scales)


def test_unrepresentable_normalization_raises_instead_of_returning_nonfinite_diagnostic():
    predictions, truth, scales = _case(dtype=torch.float32)
    predictions["state"].fill_(1e38)
    scales["state"].fill_(1e-38)
    with pytest.raises(ValueError, match="normalized errors exceed"):
        score_forecast(predictions, truth, scales)


@pytest.mark.parametrize("shape", [(8,), (1, 1, 8, 2), (0, 2), (2, 0, 2), (2, 8, 0)])
def test_invalid_or_empty_forecast_layout_is_rejected(shape):
    value = torch.zeros(shape)
    with pytest.raises(ValueError, match="matching|empty"):
        score_forecast({"state": value}, {"state": value}, {"state": torch.ones(2)})


def test_prediction_truth_shape_and_cross_head_batch_time_layout_must_match():
    predictions, truth, scales = _case()
    with pytest.raises(ValueError, match="matching"):
        score_forecast(predictions, {"state": truth["state"][:4]}, scales)
    predictions["other"] = torch.zeros(7, 1)
    truth["other"] = torch.zeros(7, 1)
    scales["other"] = torch.ones(1)
    with pytest.raises(ValueError, match="same batch/time layout"):
        score_forecast(predictions, truth, scales)


def test_named_keys_must_match_exactly():
    predictions, truth, scales = _case()
    with pytest.raises(ValueError, match="same output names"):
        score_forecast(predictions, {}, scales)
    with pytest.raises(ValueError, match="same output names"):
        score_forecast(predictions, truth, {"different": torch.ones(2)})
    with pytest.raises(ValueError, match="At least one"):
        score_forecast({}, {}, {})
    with pytest.raises(ValueError, match="nonempty strings"):
        score_forecast({1: predictions["state"]}, {1: truth["state"]}, {1: scales["state"]})


@pytest.mark.parametrize("threshold", [0, -0.2, float("nan"), float("inf")])
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="threshold"):
        score_forecast(*_case(), threshold=threshold)


def test_invalid_aggregation_is_rejected():
    with pytest.raises(ValueError, match="aggregation"):
        score_forecast(*_case(), aggregation="max")


def test_non_tensor_or_nonfloating_data_is_rejected():
    predictions, truth, scales = _case()
    with pytest.raises(TypeError, match="must be tensors"):
        score_forecast({"state": [[0, 0]]}, truth, scales)
    with pytest.raises(TypeError, match="floating tensors"):
        score_forecast({"state": predictions["state"].long()}, truth, scales)
    with pytest.raises(TypeError, match="dictionaries"):
        score_forecast([], truth, scales)


def test_mismatched_devices_are_rejected_before_device_computation():
    predictions, truth, scales = _case()
    with pytest.raises(ValueError, match="share a device"):
        score_forecast(predictions, truth, {"state": torch.ones(2, device="meta")})
    predictions["other"] = torch.zeros(8, 1, device="meta")
    truth["other"] = torch.zeros(8, 1, device="meta")
    scales["other"] = torch.ones(1, device="meta")
    with pytest.raises(ValueError, match="same device"):
        score_forecast(predictions, truth, scales)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_scoring_stays_on_device_and_matches_cpu():
    generator = torch.Generator().manual_seed(460)
    truth = {"state": torch.randn(4, 20, 3, generator=generator)}
    predicted = {"state": truth["state"] + torch.randn(4, 20, 3, generator=generator) * 0.1}
    scales = {"state": torch.tensor([0.5, 1, 2])}
    cpu = score_forecast(predicted, truth, scales)
    mps = score_forecast(
        {name: value.to("mps") for name, value in predicted.items()},
        {name: value.to("mps") for name, value in truth.items()},
        {name: value.to("mps") for name, value in scales.items()},
    )
    assert mps.score == cpu.score
    assert mps.nrmse == pytest.approx(cpu.nrmse, rel=1e-6)
    assert mps.head_horizons == cpu.head_horizons
    assert mps.head_errors == pytest.approx(cpu.head_errors, rel=1e-6)
