"""Forecast pipeline tests — driver time-alignment and shape contracts.

The timing convention under test (matching how readouts are trained, where
``target = feedback shifted by 1``):

- prediction ``0`` is produced during warmup from the *last* warmup
  feedback/driver pair;
- prediction ``t`` (``t >= 1``) is produced from prediction ``t-1`` paired
  with ``forecast_inputs[:, t-1]``, the driver at that same timestep;
- forecast drivers therefore start right *after* the warmup window and only
  ``horizon - 1`` values are consumed.
"""

import pytest
import torch

from resdag.core import ESNModel, Input
from resdag.layers import CGReadoutLayer, ESNLayer


def _build_driven_model(feedback_size: int = 1, driver_size: int = 2) -> tuple[ESNModel, ESNLayer]:
    feedback = Input(shape=(20, feedback_size))
    driver = Input(shape=(20, driver_size))
    reservoir_layer = ESNLayer(
        reservoir_size=32,
        feedback_size=feedback_size,
        input_size=driver_size,
        spectral_radius=0.9,
    )
    states = reservoir_layer(feedback, driver)
    readout = CGReadoutLayer(32, feedback_size, name="output")(states)
    model = ESNModel([feedback, driver], readout)
    return model, reservoir_layer


class TestDriverAlignment:
    """The autoregressive loop must consume drivers in training-consistent order."""

    def test_drivers_consumed_in_order_starting_after_warmup(self) -> None:
        """Step t of the forecast must see forecast_inputs[:, t-1] — the
        driver series continuing exactly where the warmup drivers ended."""
        model, reservoir_layer = _build_driven_model()
        horizon = 6

        warmup_feedback = torch.randn(1, 20, 1)
        # Distinguishable driver values: warmup drivers are 0..19, forecast
        # drivers continue at 100, 101, ...
        warmup_driver = torch.arange(20, dtype=torch.float32).view(1, 20, 1).expand(1, 20, 2)
        forecast_driver = (
            (100 + torch.arange(horizon - 1, dtype=torch.float32))
            .view(1, horizon - 1, 1)
            .expand(1, horizon - 1, 2)
        )

        seen_drivers: list[torch.Tensor] = []

        def probe(module, args):
            # args = (feedback_slice, driver_slice)
            seen_drivers.append(args[1].detach().clone())

        handle = reservoir_layer.register_forward_pre_hook(probe)
        try:
            model.forecast(
                (warmup_feedback, warmup_driver),
                forecast_inputs=(forecast_driver.contiguous(),),
                horizon=horizon,
            )
        finally:
            handle.remove()

        # First captured call is the teacher-forced warmup pass.
        assert seen_drivers[0].shape[1] == 20
        # The horizon-1 autoregressive steps must consume the forecast
        # drivers in order, with no gap and no skipped first value.
        autoregressive = seen_drivers[1:]
        assert len(autoregressive) == horizon - 1
        for t, captured in enumerate(autoregressive):
            expected = forecast_driver[:, t : t + 1, :]
            assert torch.equal(captured, expected), (
                f"autoregressive step {t + 1} consumed driver {captured.flatten()[0].item()}, "
                f"expected {expected.flatten()[0].item()}"
            )

    def test_accepts_horizon_length_drivers_and_ignores_last(self) -> None:
        """A horizon-length driver tensor is accepted (same window as the
        validation targets); its last step is never consumed."""
        model, reservoir_layer = _build_driven_model()
        horizon = 5

        warmup_feedback = torch.randn(1, 20, 1)
        warmup_driver = torch.randn(1, 20, 2)
        forecast_driver = torch.randn(1, horizon, 2)

        seen_drivers: list[torch.Tensor] = []
        handle = reservoir_layer.register_forward_pre_hook(
            lambda module, args: seen_drivers.append(args[1].detach().clone())
        )
        try:
            model.forecast(
                (warmup_feedback, warmup_driver),
                forecast_inputs=(forecast_driver,),
                horizon=horizon,
            )
        finally:
            handle.remove()

        autoregressive = seen_drivers[1:]
        assert len(autoregressive) == horizon - 1
        for t, captured in enumerate(autoregressive):
            assert torch.equal(captured, forecast_driver[:, t : t + 1, :])

    def test_rejects_wrong_driver_length(self) -> None:
        model, _ = _build_driven_model()

        warmup_feedback = torch.randn(1, 20, 1)
        warmup_driver = torch.randn(1, 20, 2)

        with pytest.raises(ValueError, match="forecast_inputs\\[0\\]"):
            model.forecast(
                (warmup_feedback, warmup_driver),
                forecast_inputs=(torch.randn(1, 3, 2),),
                horizon=10,
            )

    def test_missing_drivers_raises(self) -> None:
        model, _ = _build_driven_model()

        with pytest.raises(ValueError, match="forecast_inputs must be provided"):
            model.forecast(
                (torch.randn(1, 20, 1), torch.randn(1, 20, 2)),
                horizon=10,
            )


class TestFeedbackOnlyForecast:
    """Shape contracts for the no-driver path (regression guard)."""

    def test_forecast_shapes(self) -> None:
        feedback = Input(shape=(20, 3))
        states = ESNLayer(reservoir_size=32, feedback_size=3)(feedback)
        readout = CGReadoutLayer(32, 3, name="output")(states)
        model = ESNModel(feedback, readout)

        predictions = model.forecast(torch.randn(1, 20, 3), horizon=15)
        assert predictions.shape == (1, 15, 3)

        full = model.forecast(torch.randn(1, 20, 3), horizon=15, return_warmup=True)
        assert full.shape == (1, 35, 3)
