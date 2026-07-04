"""Tests for :mod:`resdag.metrics.forecast`.

Covers the NRMSE / contiguous-valid-horizon semantics of
:func:`valid_prediction_time`, its Lyapunov-time conversion (explicit and
dataset-inferred), and a parity check that it reproduces the golden-forecast
test's independent copy of the same logic.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from resdag.metrics import steps_to_lyapunov, valid_prediction_time


class TestValidPredictionTimeSteps:
    def test_identical_gives_full_horizon(self) -> None:
        x = torch.randn(1, 100, 3)
        assert valid_prediction_time(x, x) == 100.0

    def test_first_step_over_threshold_gives_zero(self) -> None:
        truth = torch.zeros(1, 50, 2)
        truth[..., 0] = torch.linspace(0, 1, 50)  # nonzero std for normalisation
        pred = truth + 100.0  # wildly off from step 0
        assert valid_prediction_time(pred, truth, threshold=0.3) == 0.0

    def test_contiguous_count_from_start(self) -> None:
        # Build a truth with unit per-channel std, then blow up the error after
        # a known step so the horizon is deterministic.
        rng = np.random.default_rng(0)
        truth = torch.tensor(rng.standard_normal((1, 40, 2)))
        pred = truth.clone()
        pred[:, 10:, :] += 5.0  # NRMSE jumps well above threshold from step 10
        assert valid_prediction_time(pred, truth, threshold=0.3) == 10.0

    def test_returns_float(self) -> None:
        x = torch.randn(1, 20, 3)
        assert isinstance(valid_prediction_time(x, x), float)

    def test_length_mismatch_uses_overlap(self) -> None:
        truth = torch.randn(1, 100, 3)
        pred = truth[:, :60].clone()  # shorter forecast
        assert valid_prediction_time(pred, truth) == 60.0

    def test_batch_median_of_three(self) -> None:
        # Three batch rows: two perfect, one bad from step 5. With an odd batch
        # the median is the middle (perfect) row, so the horizon spans the full
        # length -- the single bad row is outvoted.
        truth = torch.zeros(3, 30, 1)
        truth[..., 0] = torch.linspace(0, 1, 30)
        pred = truth.clone()
        pred[2, 5:, 0] += 10.0  # only row 2 goes bad
        assert valid_prediction_time(pred, truth) == 30.0

    def test_batch_median_of_two_degrades_when_either_row_fails(self) -> None:
        # With a 2-row batch the median averages both rows, so the horizon ends
        # as soon as *one* row degrades enough to lift the average over the bar.
        truth = torch.zeros(2, 30, 1)
        truth[..., 0] = torch.linspace(0, 1, 30)
        pred = truth.clone()
        pred[1, 5:, 0] += 10.0  # row 1 goes badly wrong from step 5
        assert valid_prediction_time(pred, truth) == 5.0


class TestValidPredictionTimeLyapunov:
    def test_explicit_dt_lambda(self) -> None:
        rng = np.random.default_rng(1)
        truth = torch.tensor(rng.standard_normal((1, 400, 3)))
        pred = truth.clone()
        pred[:, 357:, :] += 5.0
        vpt_steps = valid_prediction_time(pred, truth, threshold=0.3)
        assert vpt_steps == 357.0
        vpt_lyap = valid_prediction_time(
            pred, truth, threshold=0.3, units="lyapunov", dt=0.02, lambda_max=0.9056
        )
        assert vpt_lyap == pytest.approx(steps_to_lyapunov(357, 0.02, 0.9056), abs=1e-6)
        assert vpt_lyap == pytest.approx(6.466, abs=1e-3)

    def test_dataset_inference(self) -> None:
        rng = np.random.default_rng(2)
        truth = torch.tensor(rng.standard_normal((1, 200, 3)))
        pred = truth.clone()
        pred[:, 100:, :] += 5.0
        vpt_lyap = valid_prediction_time(pred, truth, units="lyapunov", dataset="lorenz")
        assert vpt_lyap == pytest.approx(steps_to_lyapunov(100, 0.02, 0.9056), abs=1e-6)

    def test_lyapunov_without_dt_raises(self) -> None:
        x = torch.randn(1, 20, 3)
        with pytest.raises(ValueError):
            valid_prediction_time(x, x, units="lyapunov", lambda_max=0.9)

    def test_lyapunov_non_chaotic_dataset_raises(self) -> None:
        x = torch.randn(1, 20, 3)
        with pytest.raises(ValueError):
            # sine has a dt but no Lyapunov exponent.
            valid_prediction_time(x, x, units="lyapunov", dataset="sine")

    def test_bad_metric_raises(self) -> None:
        x = torch.randn(1, 20, 3)
        with pytest.raises(ValueError):
            valid_prediction_time(x, x, metric="mse")

    def test_bad_units_raises(self) -> None:
        x = torch.randn(1, 20, 3)
        with pytest.raises(ValueError):
            valid_prediction_time(x, x, units="parsecs")


class TestParityWithGoldenForecast:
    def test_matches_golden_forecast_helper(self) -> None:
        # The golden-forecast test carries an independent copy of the exact same
        # NRMSE / contiguous-horizon logic (kept out of the HPO import path).
        # This module is the canonical version; assert they agree.
        import sys
        from pathlib import Path

        gf_dir = Path(__file__).resolve().parents[1] / "test_models"
        sys.path.insert(0, str(gf_dir))
        try:
            import golden_forecast as gf
        finally:
            sys.path.pop(0)

        rng = np.random.default_rng(3)
        for _ in range(5):
            truth = torch.tensor(rng.standard_normal((1, 120, 3)))
            pred = truth + torch.tensor(rng.standard_normal((1, 120, 3)) * 0.05)
            pred[:, 60:, :] += torch.tensor(rng.standard_normal((60, 3)) * 2.0)
            ref = gf.valid_prediction_time(pred, truth, threshold=0.3)
            mine = valid_prediction_time(pred, truth, threshold=0.3)
            assert mine == float(ref)
