"""Independent prefix scoring oracles, including device-only update checks."""

import math

import numpy as np
import pytest
import torch

from resdag.hpo._tuning_scores import StreamingForecastScore, score_forecast


@pytest.mark.parametrize("aggregation", ["weakest", "mean"])
@pytest.mark.parametrize("batch", [1, 2, 4])
def test_every_irregular_prefix_matches_numpy_contiguous_horizons_and_weighted_rms(
    aggregation, batch
):
    rng = np.random.default_rng(72)
    target = {
        "position": rng.normal(size=(batch, 17, 1)),
        "velocity": rng.normal(size=(batch, 17, 3)),
    }
    scales = {"position": np.array([2.0]), "velocity": np.array([3.0, 4.0, 5.0])}
    predicted = {
        name: value + rng.normal(scale=0.35, size=value.shape) * scales[name]
        for name, value in target.items()
    }
    # Deliberately include recovery after crossing; the prefix must not restart.
    for index in range(batch):
        for name in target:
            predicted[name][index, : index + 2] = target[name][index, : index + 2]
    scorer = StreamingForecastScore(
        {name: torch.tensor(value) for name, value in scales.items()}, aggregation=aggregation
    )
    start = 0
    for stop in [1, 4, 5, 11, 17]:
        scorer.update(
            {name: torch.tensor(value[:, start:stop]) for name, value in predicted.items()},
            {name: torch.tensor(value[:, start:stop]) for name, value in target.items()},
        )
        result = scorer.summary()
        horizons, errors = {}, {}
        for name in target:
            normalized = (predicted[name][:, :stop] - target[name][:, :stop]) / scales[name]
            accurate = np.sqrt(np.mean(normalized**2, axis=-1)) < 0.2
            horizons[name] = float(np.median(np.cumprod(accurate, axis=1).sum(axis=1)))
            errors[name] = float(np.sqrt(np.mean(normalized**2)))
        assert result.head_horizons == horizons
        assert result.score == (
            min(horizons.values()) if aggregation == "weakest" else np.mean(list(horizons.values()))
        )
        assert result.head_errors == pytest.approx(errors, rel=1e-12)
        assert result.nrmse == pytest.approx(
            (errors["position"] + 3 * errors["velocity"]) / 4, rel=1e-12
        )
        assert scorer.steps == stop
        start = stop


def test_even_median_and_weakest_head_are_exact_across_split_failure_boundaries():
    truth = {"a": torch.zeros(4, 9, 1), "b": torch.zeros(4, 9, 2)}
    predicted = {name: torch.ones_like(value) for name, value in truth.items()}
    for batch, end in enumerate([0, 1, 4, 7]):
        predicted["a"][batch, :end] = 0
        predicted["b"][batch, : end + 1] = 0
    scorer = StreamingForecastScore({"a": torch.ones(1), "b": torch.ones(2)})
    for start, stop in [(0, 1), (1, 4), (4, 5), (5, 9)]:
        scorer.update(
            {key: value[:, start:stop] for key, value in predicted.items()},
            {key: value[:, start:stop] for key, value in truth.items()},
        )
    assert scorer.summary().head_horizons == {"a": 2.5, "b": 3.5}
    assert scorer.summary().score == 2.5


def test_updates_never_read_tensor_values_on_host_and_do_not_retain_forecast_rows(monkeypatch):
    truth = {"state": torch.zeros(5, 2)}
    predicted = {"state": torch.full((5, 2), 0.125, requires_grad=True)}
    scorer = StreamingForecastScore({"state": torch.ones(2)})

    def forbidden(*args, **kwargs):
        raise AssertionError("A streaming update synchronized a tensor to the host")

    with monkeypatch.context() as patch:
        for name in ("item", "cpu", "numpy", "__bool__"):
            patch.setattr(torch.Tensor, name, forbidden)
        scorer.update(predicted, truth)
        scorer.update(predicted, truth)
    assert scorer.summary().score == 10
    assert scorer.summary().nrmse == 0.125
    assert predicted["state"].grad is None
    head = scorer._heads["state"]
    assert (
        sum(value.numel() for value in vars(head).values() if isinstance(value, torch.Tensor)) == 5
    )
    assert all(
        not value.requires_grad for value in vars(head).values() if isinstance(value, torch.Tensor)
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("magnitude", [0.0, 0.125])
def test_dtype_promotion_and_zero_error_match_complete_score(dtype, magnitude):
    predicted = {"x": torch.full((9, 2), magnitude, dtype=dtype)}
    target = {"x": torch.zeros_like(predicted["x"])}
    scales = {"x": torch.ones(2, dtype=dtype)}
    scorer = StreamingForecastScore(scales)
    for start, stop in [(0, 2), (2, 9)]:
        scorer.update({"x": predicted["x"][start:stop]}, {"x": target["x"][start:stop]})
    assert scorer.summary() == score_forecast(predicted, target, scales)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_extreme_finite_errors_use_stable_scaled_sum_of_squares(dtype):
    maximum = torch.finfo(dtype).max
    scorer = StreamingForecastScore({"x": torch.ones(1, dtype=dtype)})
    for length in (1, 3, 2):
        scorer.update(
            {"x": torch.full((length, 1), maximum, dtype=dtype)},
            {"x": torch.zeros(length, 1, dtype=dtype)},
        )
    result = scorer.summary()
    assert result.score == 0
    assert math.isfinite(result.nrmse)
    assert result.nrmse == pytest.approx(maximum)


def test_opposite_extreme_values_normalize_before_overflow_and_small_rms_remains_representable():
    for magnitude, scale, expected in [(1e308, 1e308, 2.0), (1e-300, 1.0, 2e-300)]:
        scorer = StreamingForecastScore({"x": torch.tensor([scale], dtype=torch.float64)})
        for _ in range(3):
            scorer.update(
                {"x": torch.full((2, 1), magnitude, dtype=torch.float64)},
                {"x": torch.full((2, 1), -magnitude, dtype=torch.float64)},
            )
        assert scorer.summary().nrmse == pytest.approx(expected, rel=1e-12, abs=0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_invalid_data_accumulates_on_device_and_raises_when_summarized(bad):
    scorer = StreamingForecastScore({"x": torch.ones(1)})
    scorer.update({"x": torch.tensor([[bad]])}, {"x": torch.zeros(1, 1)})
    with pytest.raises(ValueError, match="finite"):
        scorer.summary()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Native MPS is unavailable")
def test_native_mps_accumulators_remain_on_device():
    scorer = StreamingForecastScore({"x": torch.ones(2, device="mps")})
    for _ in range(3):
        scorer.update(
            {"x": torch.full((2, 4, 2), 0.125, device="mps")},
            {"x": torch.zeros(2, 4, 2, device="mps")},
        )
    assert all(
        value.device.type == "mps"
        for value in vars(scorer._heads["x"]).values()
        if isinstance(value, torch.Tensor)
    )
    assert scorer.summary().score == 12
    assert scorer.summary().nrmse == 0.125
