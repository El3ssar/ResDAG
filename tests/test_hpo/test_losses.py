"""Tests for HPO loss functions."""

import warnings

import numpy as np
import pytest

from resdag.hpo.losses import (
    LOSSES,
    SCALE_FREE_METRICS,
    _aggregate_batch,
    expected_forecast_horizon,
    forecast_horizon,
    get_loss,
    lyapunov_weighted,
    soft_valid_horizon,
    standard_loss,
)


class TestLossFunctionsBasic:
    """Test basic properties of all loss functions."""

    def setup_method(self):
        """Create test data."""
        np.random.seed(42)
        # (B, T, D) = (4, 100, 3)
        self.y_true = np.random.randn(4, 100, 3)
        self.y_pred = self.y_true + np.random.randn(4, 100, 3) * 0.1

    def test_expected_forecast_horizon_returns_negative(self):
        """EFH returns negative values (to minimize)."""
        loss = expected_forecast_horizon(self.y_true, self.y_pred)
        assert isinstance(loss, float)
        assert loss < 0  # Negative horizon

    def test_expected_forecast_horizon_perfect_pred(self):
        """Perfect predictions give very negative EFH (good)."""
        loss_noisy = expected_forecast_horizon(self.y_true, self.y_pred)
        loss_perfect = expected_forecast_horizon(self.y_true, self.y_true)
        assert loss_perfect < loss_noisy  # Perfect is better (more negative)

    def test_forecast_horizon_returns_negative(self):
        """Forecast horizon returns negative value."""
        loss = forecast_horizon(self.y_true, self.y_pred)
        assert isinstance(loss, float)

    def test_lyapunov_weighted_returns_positive(self):
        """Lyapunov loss returns positive values."""
        loss = lyapunov_weighted(self.y_true, self.y_pred)
        assert isinstance(loss, float)
        assert loss > 0

    def test_standard_loss_returns_positive(self):
        """Standard loss returns positive values."""
        loss = standard_loss(self.y_true, self.y_pred)
        assert isinstance(loss, float)
        assert loss > 0

    def test_soft_valid_horizon_returns_negative(self):
        """Soft valid horizon returns negative values."""
        loss = soft_valid_horizon(self.y_true, self.y_pred)
        assert isinstance(loss, float)
        assert loss < 0  # Negative horizon

    def test_soft_valid_horizon_perfect_pred(self):
        """Perfect predictions give very negative soft horizon (good)."""
        loss_noisy = soft_valid_horizon(self.y_true, self.y_pred)
        loss_perfect = soft_valid_horizon(self.y_true, self.y_true)
        assert loss_perfect < loss_noisy

    def test_all_losses_lower_for_better_predictions(self):
        """All losses should be lower for better predictions."""
        y_pred_bad = self.y_true + np.random.randn(4, 100, 3) * 1.0

        for name, loss_fn in LOSSES.items():
            loss_good = loss_fn(self.y_true, self.y_pred)
            loss_bad = loss_fn(self.y_true, y_pred_bad)
            # For EFH/horizon, more negative is better, so loss_good < loss_bad
            # For others, lower is better
            assert loss_good < loss_bad, f"Loss {name} failed monotonicity test"


class TestLossFunctionsShapes:
    """Test loss functions handle various input shapes."""

    def test_single_batch(self):
        """Works with single batch (1, T, D)."""
        y_true = np.random.randn(1, 50, 3)
        y_pred = y_true + np.random.randn(1, 50, 3) * 0.1

        for name, loss_fn in LOSSES.items():
            loss = loss_fn(y_true, y_pred)
            assert isinstance(loss, float), f"Loss {name} failed single batch"

    def test_single_feature(self):
        """Works with single feature (B, T, 1)."""
        y_true = np.random.randn(4, 50, 1)
        y_pred = y_true + np.random.randn(4, 50, 1) * 0.1

        for name, loss_fn in LOSSES.items():
            loss = loss_fn(y_true, y_pred)
            assert isinstance(loss, float), f"Loss {name} failed single feature"

    def test_shape_mismatch_raises(self):
        """Shape mismatch raises ValueError."""
        y_true = np.random.randn(4, 50, 3)
        y_pred = np.random.randn(4, 60, 3)  # Different time steps

        for name, loss_fn in LOSSES.items():
            with pytest.raises(ValueError):
                loss_fn(y_true, y_pred)


class TestLossFunctionsParameters:
    """Test loss function parameters."""

    def setup_method(self):
        np.random.seed(42)
        self.y_true = np.random.randn(4, 100, 3)
        self.y_pred = self.y_true + np.random.randn(4, 100, 3) * 0.2

    def test_efh_threshold(self):
        """EFH with different thresholds."""
        loss_tight = expected_forecast_horizon(self.y_true, self.y_pred, threshold=0.1)
        loss_loose = expected_forecast_horizon(self.y_true, self.y_pred, threshold=0.5)
        # Looser threshold should give better (more negative) score
        assert loss_loose < loss_tight

    def test_efh_metrics(self):
        """EFH with different metrics."""
        for metric in ["rmse", "mse", "mae", "nrmse"]:
            with warnings.catch_warnings():
                # Raw-scale metrics intentionally warn (scale-safety); silence here.
                warnings.simplefilter("ignore", UserWarning)
                loss = expected_forecast_horizon(self.y_true, self.y_pred, metric=metric)
            assert isinstance(loss, float)

    def test_lyapunov_lyapunov_t(self):
        """Lyapunov with different lyapunov_t values."""
        loss_short = lyapunov_weighted(self.y_true, self.y_pred, lyapunov_t=10)
        loss_long = lyapunov_weighted(self.y_true, self.y_pred, lyapunov_t=100)
        # Different lyapunov_t should give different results
        assert loss_short != loss_long

    def test_soft_horizon_threshold(self):
        """Soft horizon with different thresholds."""
        loss_tight = soft_valid_horizon(self.y_true, self.y_pred, threshold=0.1)
        loss_loose = soft_valid_horizon(self.y_true, self.y_pred, threshold=0.5)
        # Looser threshold should give better (more negative) score
        assert loss_loose < loss_tight

    def test_soft_horizon_sharpness(self):
        """Soft horizon with different n values."""
        loss_soft = soft_valid_horizon(self.y_true, self.y_pred, n=2)
        loss_sharp = soft_valid_horizon(self.y_true, self.y_pred, n=20)
        # Different n should give different results
        assert loss_soft != loss_sharp


class TestSoftValidHorizonNumerical:
    """Test soft_valid_horizon numerical stability."""

    def test_diverged_predictions_no_nan(self):
        """Diverged predictions should not produce NaN."""
        np.random.seed(42)
        y_true = np.random.randn(4, 100, 3)
        # Huge errors — should not overflow
        y_pred = y_true + np.random.randn(4, 100, 3) * 1e10

        loss = soft_valid_horizon(y_true, y_pred)
        assert isinstance(loss, float)
        assert np.isfinite(loss)

    def test_perfect_predictions_give_max_horizon(self):
        """Perfect predictions should give horizon close to T."""
        y_true = np.random.randn(4, 100, 3)
        loss = soft_valid_horizon(y_true, y_true)
        # Perfect predictions: all good_t ≈ 1, H ≈ T = 100
        assert loss < -90  # Should be close to -100


class TestGetLoss:
    """Test the get_loss helper function."""

    def test_get_by_string(self):
        """Get loss by string key."""
        for name in LOSSES:
            loss_fn = get_loss(name)
            assert callable(loss_fn)
            assert loss_fn is LOSSES[name]

    def test_get_unknown_raises(self):
        """Unknown string raises KeyError."""
        with pytest.raises(KeyError):
            get_loss("unknown_loss")

    def test_get_callable_passthrough(self):
        """Callable is passed through."""

        def my_loss(y_true, y_pred):
            return 0.0

        result = get_loss(my_loss)
        assert result is my_loss

    def test_get_non_callable_raises(self):
        """Non-callable raises TypeError."""
        with pytest.raises(TypeError):
            get_loss(123)


class TestLossesRegistry:
    """Test the LOSSES registry."""

    def test_registry_has_all_losses(self):
        """Registry contains all expected losses."""
        expected = {"efh", "forecast_horizon", "lyapunov", "standard", "soft_horizon"}
        assert set(LOSSES.keys()) == expected

    def test_registry_values_are_callable(self):
        """All registry values are callable."""
        for name, loss_fn in LOSSES.items():
            assert callable(loss_fn), f"Loss {name} is not callable"


# Threshold-based horizon losses that must default to a scale-free metric.
HORIZON_LOSSES = {
    "efh": expected_forecast_horizon,
    "forecast_horizon": forecast_horizon,
    "soft_horizon": soft_valid_horizon,
}


class TestScaleFreeDefaults:
    """AC1: horizon-loss defaults are scale-free (nrmse), and raw-scale warns."""

    def test_horizon_losses_default_to_nrmse(self):
        """All threshold-based horizon losses default to a scale-free metric."""
        import inspect

        for name, fn in HORIZON_LOSSES.items():
            default = inspect.signature(fn).parameters["metric"].default
            assert (
                default in SCALE_FREE_METRICS
            ), f"{name} default metric {default!r} not scale-free"

    def test_lyapunov_defaults_to_nrmse(self):
        """Lyapunov-weighted loss defaults to a scale-free metric for consistency."""
        import inspect

        default = inspect.signature(lyapunov_weighted).parameters["metric"].default
        assert default in SCALE_FREE_METRICS

    def test_default_horizon_is_scale_invariant(self):
        """With nrmse defaults, scaling the data 100x leaves the horizon unchanged."""
        rng = np.random.default_rng(0)
        y_true = rng.standard_normal((4, 100, 3))
        y_pred = y_true + rng.standard_normal((4, 100, 3)) * 0.05

        for name, fn in HORIZON_LOSSES.items():
            base = fn(y_true, y_pred)
            scaled = fn(y_true * 100.0, y_pred * 100.0)
            assert np.isclose(base, scaled), f"{name} is not scale-invariant by default"

    def test_no_warning_with_default_metric(self):
        """Calling with the default (nrmse) metric does not warn."""
        rng = np.random.default_rng(1)
        y_true = rng.standard_normal((4, 50, 3))
        y_pred = y_true + rng.standard_normal((4, 50, 3)) * 0.1

        for name, fn in HORIZON_LOSSES.items():
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # any warning fails the test
                fn(y_true, y_pred)

    @pytest.mark.parametrize("metric", ["rmse", "mse", "mae"])
    def test_raw_scale_metric_warns(self, metric):
        """Threshold-based losses warn when given a raw-scale metric."""
        rng = np.random.default_rng(2)
        y_true = rng.standard_normal((4, 50, 3))
        y_pred = y_true + rng.standard_normal((4, 50, 3)) * 0.1

        for name, fn in HORIZON_LOSSES.items():
            with pytest.warns(UserWarning, match="raw-scale metric"):
                fn(y_true, y_pred, metric=metric)


class TestBatchAggregation:
    """AC2: aggregation is consistent/documented and gmean is zero-safe."""

    def test_aggregate_median_matches_numpy(self):
        """The median path matches ``np.median`` over the batch axis."""
        errors = np.array([[0.0, 0.5], [0.5, 0.5], [0.5, 0.5]])
        out = _aggregate_batch(errors, how="median")
        np.testing.assert_allclose(out, np.median(errors, axis=0))

    def test_aggregate_gmean_is_zero_safe(self):
        """A single zero-error batch element must not collapse gmean to 0."""
        # Column 0 has a perfect (0.0) element; plain gmean would return 0 there.
        errors = np.array([[0.0, 0.5], [0.5, 0.5], [0.5, 0.5]])
        out = _aggregate_batch(errors, how="gmean")
        assert np.all(out > 0.0), "gmean collapsed to zero on a perfect batch element"

    def test_aggregate_unknown_raises(self):
        """An unknown aggregation strategy raises ValueError."""
        errors = np.zeros((3, 2))
        with pytest.raises(ValueError):
            _aggregate_batch(errors, how="mean")

    def test_standard_loss_zero_safe(self):
        """standard_loss stays positive even with a perfect batch element."""
        y_true = np.zeros((3, 10, 2))
        y_true[1:] = 1.0  # non-degenerate scale for nrmse
        y_pred = y_true.copy()
        y_pred[0] = y_true[0] + 0.5  # one imperfect element
        loss = standard_loss(y_true, y_pred)
        assert np.isfinite(loss)
        assert loss > 0.0

    def test_lyapunov_zero_safe(self):
        """lyapunov_weighted stays finite/positive with a perfect batch element."""
        y_true = np.zeros((3, 10, 2))
        y_true[1:] = 1.0
        y_pred = y_true.copy()
        y_pred[0] = y_true[0] + 0.5
        loss = lyapunov_weighted(y_true, y_pred)
        assert np.isfinite(loss)
        assert loss > 0.0
