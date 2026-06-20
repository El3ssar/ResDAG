"""Contract tests for the high-level :class:`resdag.ESN` facade.

These pin down the "easiest API" headline: ``ESN(...).fit(series).forecast(N)``
returns an ``(N, D)`` forecast, builds an :class:`~resdag.core.ESNModel`
internally, drives :class:`~resdag.training.ESNTrainer`, mirrors numpy/torch
input types, and handles the stateful-reset / readout-name / slot-0 footguns.
"""

import numpy as np
import pytest
import torch

import resdag as rd
from resdag import ESN
from resdag.core import ESNModel
from resdag.layers import ReadoutLayer
from resdag.layers.reservoirs import BaseReservoirLayer


def _sine_series(timesteps: int = 1500, features: int = 3) -> np.ndarray:
    """A clean multi-feature periodic series of shape (time, features)."""
    t = np.linspace(0.0, 40.0 * np.pi, timesteps)
    cols = [np.sin(t * (1.0 + 0.1 * i) + 0.3 * i) for i in range(features)]
    return np.stack(cols, axis=-1).astype(np.float64)


class TestPublicAPI:
    """The facade is exported per project convention."""

    def test_exported_from_top_level(self) -> None:
        assert hasattr(rd, "ESN")
        assert rd.ESN is ESN

    def test_in_dunder_all(self) -> None:
        assert "ESN" in rd.__all__

    def test_repr_reflects_fitted_state(self) -> None:
        esn = ESN(reservoir_size=32, washout=20)
        assert "fitted=False" in repr(esn)
        esn.fit(_sine_series(400))
        assert "fitted=True" in repr(esn)


class TestOneLiner:
    """The headline ``fit().forecast()`` one-liner."""

    def test_chained_oneliner_returns_correct_shape(self) -> None:
        series = _sine_series(1200, features=3)
        prediction = ESN(reservoir_size=200, washout=80, seed=0).fit(series).forecast(horizon=150)
        assert prediction.shape == (150, 3)

    def test_fit_returns_self_for_chaining(self) -> None:
        esn = ESN(reservoir_size=64, washout=30)
        returned = esn.fit(_sine_series(500))
        assert returned is esn

    def test_forecast_is_accurate_on_periodic_signal(self) -> None:
        """A genuine forecast, not noise: matches the analytic continuation."""
        t = np.linspace(0.0, 60.0 * np.pi, 3000)
        dt = float(t[1] - t[0])
        series = np.stack([np.sin(t), np.cos(t)], axis=-1)

        esn = ESN(reservoir_size=300, spectral_radius=0.95, washout=100, alpha=1e-7, seed=0).fit(
            series
        )
        prediction = esn.forecast(horizon=200)

        t_cont = 60.0 * np.pi + dt * np.arange(1, 201)
        truth = np.stack([np.sin(t_cont), np.cos(t_cont)], axis=-1)
        rmse = float(np.sqrt(np.mean((prediction - truth) ** 2)))
        assert rmse < 0.1, f"forecast RMSE too high: {rmse}"


class TestComposesBuildingBlocks:
    """The facade reuses ESNModel + ESNTrainer rather than reinventing them."""

    def test_builds_esnmodel_internally(self) -> None:
        esn = ESN(reservoir_size=50, washout=30).fit(_sine_series(400))
        assert isinstance(esn.model, ESNModel)

    def test_model_is_none_before_fit(self) -> None:
        assert ESN(reservoir_size=50).model is None

    def test_model_has_reservoir_and_fitted_readout(self) -> None:
        esn = ESN(reservoir_size=50, washout=30).fit(_sine_series(400))
        assert esn.model is not None
        reservoirs = [m for m in esn.model.modules() if isinstance(m, BaseReservoirLayer)]
        readouts = [m for m in esn.model.modules() if isinstance(m, ReadoutLayer)]
        assert reservoirs, "facade model should contain a reservoir"
        assert readouts, "facade model should contain a readout"
        assert all(r.is_fitted for r in readouts), "readout(s) should be fitted after fit()"

    def test_underlying_model_stays_accessible(self) -> None:
        """Composability is not hidden: drop down to the ESNModel API."""
        esn = ESN(reservoir_size=50, washout=30).fit(_sine_series(400))
        assert esn.model is not None
        # ESNModel-specific methods remain available on the composed model.
        states = esn.model.get_reservoir_states()
        assert isinstance(states, dict)
        esn.model.reset_reservoirs()  # should not raise


class TestInputTypeMirroring:
    """numpy in -> numpy out; torch in -> torch out; batch axis preserved."""

    def test_numpy_in_numpy_out(self) -> None:
        esn = ESN(reservoir_size=64, washout=40, seed=0).fit(_sine_series(800))
        pred = esn.forecast(horizon=60)
        assert isinstance(pred, np.ndarray)
        assert pred.shape == (60, 3)

    def test_torch_in_torch_out(self) -> None:
        series = torch.as_tensor(_sine_series(800))
        esn = ESN(reservoir_size=64, washout=40, seed=0).fit(series)
        pred = esn.forecast(horizon=60)
        assert isinstance(pred, torch.Tensor)
        assert tuple(pred.shape) == (60, 3)

    def test_2d_input_treated_as_single_batch(self) -> None:
        series = torch.as_tensor(_sine_series(600))  # (T, D)
        pred = ESN(reservoir_size=48, washout=30).fit(series).forecast(horizon=40)
        assert tuple(pred.shape) == (40, 3)  # batch axis squeezed away

    def test_3d_single_batch_squeezed(self) -> None:
        series = torch.as_tensor(_sine_series(600)).unsqueeze(0)  # (1, T, D)
        pred = ESN(reservoir_size=48, washout=30).fit(series).forecast(horizon=40)
        assert tuple(pred.shape) == (40, 3)

    def test_multi_batch_preserves_batch_axis(self) -> None:
        series = torch.as_tensor(_sine_series(600)).unsqueeze(0).repeat(2, 1, 1)  # (2, T, D)
        pred = ESN(reservoir_size=48, washout=30).fit(series).forecast(horizon=40)
        assert tuple(pred.shape) == (2, 40, 3)

    def test_integer_numpy_series_is_accepted(self) -> None:
        """Integer input is cast to float rather than breaking reservoir math."""
        rng = np.random.default_rng(0)
        series = rng.integers(-5, 5, size=(500, 2))
        pred = ESN(reservoir_size=40, washout=30).fit(series).forecast(horizon=20)
        assert pred.shape == (20, 2)


class TestForecastBehaviour:
    """Footguns the facade is supposed to absorb."""

    def test_stateful_reset_makes_forecasts_independent(self) -> None:
        """Back-to-back forecasts are identical: forecast() resets state itself."""
        esn = ESN(reservoir_size=80, washout=50, seed=0).fit(_sine_series(900))
        first = esn.forecast(horizon=70)
        second = esn.forecast(horizon=70)
        np.testing.assert_allclose(first, second, rtol=0, atol=1e-6)

    def test_return_warmup_prepends_sync_window(self) -> None:
        esn = ESN(reservoir_size=64, washout=50, seed=0).fit(_sine_series(800))
        pred = esn.forecast(horizon=40, return_warmup=True)
        assert pred.shape == (50 + 40, 3)

    def test_seeded_models_are_reproducible(self) -> None:
        series = _sine_series(800)
        a = ESN(reservoir_size=64, washout=40, seed=123).fit(series).forecast(horizon=50)
        b = ESN(reservoir_size=64, washout=40, seed=123).fit(series).forecast(horizon=50)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)

    def test_refit_same_feature_dim_reuses_model(self) -> None:
        esn = ESN(reservoir_size=48, washout=30, seed=0)
        esn.fit(_sine_series(600))
        model_id = id(esn.model)
        esn.fit(_sine_series(700))  # refit on new (same-D) data
        assert id(esn.model) == model_id


class TestValidation:
    """Clear errors for misuse."""

    def test_forecast_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="before fit"):
            ESN(reservoir_size=32).forecast(horizon=10)

    def test_series_too_short_raises(self) -> None:
        with pytest.raises(ValueError, match="washout"):
            ESN(reservoir_size=32, washout=100).fit(_sine_series(50))

    def test_non_positive_horizon_raises(self) -> None:
        esn = ESN(reservoir_size=32, washout=30).fit(_sine_series(400))
        with pytest.raises(ValueError, match="horizon"):
            esn.forecast(horizon=0)

    def test_bad_ndim_raises(self) -> None:
        with pytest.raises(ValueError, match="2-D|3-D"):
            ESN(reservoir_size=32).fit(np.zeros((10,)))

    def test_unsupported_input_type_raises(self) -> None:
        with pytest.raises(TypeError, match="torch.Tensor or numpy"):
            ESN(reservoir_size=32).fit([[1.0, 2.0], [3.0, 4.0]])

    def test_changing_feature_dim_on_refit_raises(self) -> None:
        esn = ESN(reservoir_size=32, washout=30).fit(_sine_series(400, features=3))
        with pytest.raises(ValueError, match="feature"):
            esn.fit(_sine_series(400, features=2))

    def test_invalid_construction_args_raise(self) -> None:
        with pytest.raises(ValueError, match="reservoir_size"):
            ESN(reservoir_size=0)
        with pytest.raises(ValueError, match="washout"):
            ESN(reservoir_size=32, washout=0)


class TestConfigurability:
    """Hyperparameters flow through to the underlying reservoir/readout."""

    def test_hyperparameters_reach_the_reservoir(self) -> None:
        esn = ESN(
            reservoir_size=64,
            spectral_radius=0.8,
            leak_rate=0.5,
            washout=30,
            topology="erdos_renyi",
            seed=0,
        ).fit(_sine_series(500))
        assert esn.model is not None
        reservoir = next(m for m in esn.model.modules() if isinstance(m, BaseReservoirLayer))
        assert reservoir.reservoir_size == 64
        assert reservoir.leak_rate == pytest.approx(0.5)

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_runs_on_cuda(self) -> None:
        esn = ESN(reservoir_size=64, washout=30, device="cuda", seed=0)
        pred = esn.fit(_sine_series(500)).forecast(horizon=40)
        # numpy series -> numpy out even on GPU.
        assert pred.shape == (40, 3)

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_adopts_input_tensor_device(self) -> None:
        """A CUDA series with device unset 'just works' (no device mismatch)."""
        series = torch.as_tensor(_sine_series(500)).cuda()
        esn = ESN(reservoir_size=64, washout=30, seed=0).fit(series)
        pred = esn.forecast(horizon=40)
        assert isinstance(pred, torch.Tensor)
        assert pred.device.type == "cuda"
        assert tuple(pred.shape) == (40, 3)


class TestDtypeAndDataHygiene:
    """dtype preservation, NaN/inf guards, and small-shape edge cases."""

    def test_float64_input_is_not_downcast(self) -> None:
        """A float64 series keeps float64 end-to-end (precision matters for chaos)."""
        series = torch.as_tensor(_sine_series(600)).to(torch.float64)
        esn = ESN(reservoir_size=48, washout=30, seed=0).fit(series)
        assert esn.dtype == torch.float64
        pred = esn.forecast(horizon=30)
        assert pred.dtype == torch.float64

    def test_explicit_dtype_overrides_input(self) -> None:
        series = torch.as_tensor(_sine_series(600)).to(torch.float64)
        esn = ESN(reservoir_size=48, washout=30, dtype=torch.float32, seed=0).fit(series)
        pred = esn.forecast(horizon=30)
        assert pred.dtype == torch.float32

    def test_nan_input_raises(self) -> None:
        series = _sine_series(500).copy()
        series[100, 0] = np.nan
        with pytest.raises(ValueError, match="NaN or infinite"):
            ESN(reservoir_size=32, washout=30).fit(series)

    def test_inf_input_raises(self) -> None:
        series = _sine_series(500).copy()
        series[100, 0] = np.inf
        with pytest.raises(ValueError, match="NaN or infinite"):
            ESN(reservoir_size=32, washout=30).fit(series)

    def test_single_feature_series(self) -> None:
        series = _sine_series(600, features=1)
        pred = ESN(reservoir_size=64, washout=40, seed=0).fit(series).forecast(horizon=50)
        assert pred.shape == (50, 1)

    def test_horizon_longer_than_training_series(self) -> None:
        """Autoregression is unbounded: horizon may exceed the input length."""
        series = _sine_series(400)
        pred = ESN(reservoir_size=120, washout=60, seed=0).fit(series).forecast(horizon=900)
        assert pred.shape == (900, 3)
