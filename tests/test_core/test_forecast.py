"""Forecast pipeline tests — autoregressive semantics, validation, shapes.

The contract under test (see :meth:`resdag.core.ESNModel.forecast`):

- every returned slot is a *genuine* autoregressive step — the loop feeds the
  model its own previous output, seeded by ``initial_feedback`` or the last
  warmup output.  No teacher-forced frame is emitted, so slot 0 is a real
  forecast (not a copy of the warmup output) and ``return_warmup=True`` does not
  duplicate the warmup/forecast seam;
- driver alignment matches training (``target = feedback shifted by 1``): step
  ``t`` consumes ``forecast_inputs[:, t]``, so the driver series starts right
  after the warmup window and must supply at least ``horizon`` timesteps.
"""

import pytest
import torch

from resdag.core import ESNModel, Input
from resdag.layers import CGReadoutLayer, ESNLayer


def _build_driven_model(feedback_size: int = 1, driver_size: int = 2) -> tuple[ESNModel, ESNLayer]:
    torch.manual_seed(42)
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


def _build_multi_output_model(feedback_size: int = 2, aux_size: int = 3) -> ESNModel:
    """Two-output model; the first output (``main``) is used as feedback."""
    torch.manual_seed(42)
    feedback = Input(shape=(20, feedback_size))
    states = ESNLayer(reservoir_size=32, feedback_size=feedback_size, spectral_radius=0.9)(feedback)
    main = CGReadoutLayer(32, feedback_size, name="main")(states)
    aux = CGReadoutLayer(32, aux_size, name="aux")(states)
    return ESNModel(feedback, [main, aux])


def _manual_autoregressive(model: ESNModel, warmup: torch.Tensor, horizon: int) -> torch.Tensor:
    """Recompute a single-output forecast by hand from the same warmed state."""
    model.reset_reservoirs()
    warm = model.warmup(warmup, return_outputs=True)
    assert isinstance(warm, torch.Tensor)
    current = warm[:, -1:, :]
    steps = []
    with torch.no_grad():
        for _ in range(horizon):
            out = model(current)
            steps.append(out)
            current = out
    return torch.cat(steps, dim=1)


class TestAutoregressiveSlotZero:
    """Slot 0 must be a genuine autoregressive step, not the warmup output."""

    def test_horizon_one_differs_from_warmup_last_output(self, make_tiny_model) -> None:
        """``forecast(wu, horizon=1)`` is a real step, not an echo of warmup."""
        model = make_tiny_model(feedback_size=3)
        warmup = torch.randn(1, 20, 3)

        model.reset_reservoirs()
        warm = model.warmup(warmup, return_outputs=True)
        assert isinstance(warm, torch.Tensor)
        last_warmup_output = warm[:, -1:, :].clone()

        model.reset_reservoirs()
        preds = model.forecast(warmup, horizon=1)

        assert preds.shape == (1, 1, 3)
        assert not torch.allclose(preds[:, 0, :], last_warmup_output[:, 0, :], atol=1e-6)

    def test_matches_manual_autoregressive_recompute(self, make_tiny_model) -> None:
        """For ``horizon >= 2``, every slot equals a hand-rolled AR loop."""
        model = make_tiny_model(feedback_size=3)
        warmup = torch.randn(1, 20, 3)
        horizon = 8

        model.reset_reservoirs()
        preds = model.forecast(warmup, horizon=horizon)
        manual = _manual_autoregressive(model, warmup, horizon)

        assert preds.shape == (1, horizon, 3)
        assert torch.allclose(preds, manual, atol=1e-6)


class TestInitialFeedback:
    """``initial_feedback`` seeds the first step and is validated."""

    def test_initial_feedback_changes_slot_zero(self, make_tiny_model) -> None:
        model = make_tiny_model(feedback_size=3)
        warmup = torch.randn(1, 20, 3)

        model.reset_reservoirs()
        default = model.forecast(warmup, horizon=4)

        model.reset_reservoirs()
        custom = model.forecast(warmup, horizon=4, initial_feedback=torch.full((1, 1, 3), 5.0))

        assert not torch.allclose(default[:, 0, :], custom[:, 0, :], atol=1e-6)

    def test_initial_feedback_is_used_as_seed(self, make_tiny_model) -> None:
        """The first step is ``model(initial_feedback)`` from the warmed state."""
        model = make_tiny_model(feedback_size=3)
        warmup = torch.randn(1, 20, 3)
        seed = torch.randn(1, 1, 3)

        model.reset_reservoirs()
        preds = model.forecast(warmup, horizon=3, initial_feedback=seed)

        # Recompute by hand: warm the state, then drive with the custom seed.
        model.reset_reservoirs()
        model.warmup(warmup)
        with torch.no_grad():
            expected_first = model(seed)
        assert torch.allclose(preds[:, 0:1, :], expected_first, atol=1e-6)

    @pytest.mark.parametrize(
        "bad",
        [
            torch.randn(1, 3),  # missing the time axis (rank 2)
            torch.randn(2, 1, 3),  # wrong batch
            torch.randn(1, 1, 5),  # wrong feature dim
            torch.randn(1, 2, 3),  # more than one timestep
        ],
    )
    def test_invalid_initial_feedback_raises(self, make_tiny_model, bad) -> None:
        model = make_tiny_model(feedback_size=3)
        with pytest.raises(ValueError, match="initial_feedback"):
            model.forecast(torch.randn(1, 20, 3), horizon=3, initial_feedback=bad)


class TestHorizonValidation:
    """``horizon`` must be a positive integer."""

    @pytest.mark.parametrize("horizon", [0, -1, -7])
    def test_non_positive_horizon_raises(self, make_tiny_model, horizon) -> None:
        model = make_tiny_model(feedback_size=3)
        with pytest.raises(ValueError, match="horizon"):
            model.forecast(torch.randn(1, 20, 3), horizon=horizon)


class TestReturnWarmupSeam:
    """``return_warmup`` must not duplicate the warmup/forecast seam frame."""

    def test_no_duplicate_seam_and_length(self, make_tiny_model) -> None:
        model = make_tiny_model(feedback_size=3)
        warmup_steps, horizon = 20, 15
        warmup = torch.randn(1, warmup_steps, 3)

        model.reset_reservoirs()
        full = model.forecast(warmup, horizon=horizon, return_warmup=True)
        assert full.shape == (1, warmup_steps + horizon, 3)

        # The last warmup frame and the first forecast frame must differ.
        seam_last_warmup = full[:, warmup_steps - 1, :]
        seam_first_forecast = full[:, warmup_steps, :]
        assert not torch.allclose(seam_last_warmup, seam_first_forecast, atol=1e-6)

        # The forecast tail equals a plain (no-warmup) forecast.
        model.reset_reservoirs()
        preds = model.forecast(warmup, horizon=horizon)
        assert torch.allclose(full[:, warmup_steps:, :], preds, atol=1e-6)


class TestDriverAlignment:
    """The autoregressive loop consumes drivers in training-consistent order."""

    def test_drivers_consumed_in_order_starting_after_warmup(self) -> None:
        """Step ``t`` must see ``forecast_inputs[:, t]`` — the driver series
        continuing exactly where the warmup drivers ended."""
        model, reservoir_layer = _build_driven_model()
        horizon = 6

        warmup_feedback = torch.randn(1, 20, 1)
        # Distinguishable driver values: warmup drivers 0..19, forecast drivers
        # continue at 100, 101, ...
        warmup_driver = torch.arange(20, dtype=torch.float32).view(1, 20, 1).expand(1, 20, 2)
        forecast_driver = (
            (100 + torch.arange(horizon, dtype=torch.float32))
            .view(1, horizon, 1)
            .expand(1, horizon, 2)
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
        # The horizon autoregressive steps consume the forecast drivers in
        # order, with no gap and no skipped first value.
        autoregressive = seen_drivers[1:]
        assert len(autoregressive) == horizon
        for t, captured in enumerate(autoregressive):
            expected = forecast_driver[:, t : t + 1, :]
            assert torch.equal(captured, expected), (
                f"autoregressive step {t} consumed driver {captured.flatten()[0].item()}, "
                f"expected {expected.flatten()[0].item()}"
            )

    def test_accepts_extra_driver_steps_and_consumes_first_horizon(self) -> None:
        """A longer driver tensor is accepted; only the first ``horizon`` are used."""
        model, reservoir_layer = _build_driven_model()
        horizon = 5

        warmup_feedback = torch.randn(1, 20, 1)
        warmup_driver = torch.randn(1, 20, 2)
        forecast_driver = torch.randn(1, horizon + 4, 2)

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
        assert len(autoregressive) == horizon
        for t, captured in enumerate(autoregressive):
            assert torch.equal(captured, forecast_driver[:, t : t + 1, :])

    def test_rejects_too_few_driver_steps(self) -> None:
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
        torch.manual_seed(42)
        feedback = Input(shape=(20, 3))
        states = ESNLayer(reservoir_size=32, feedback_size=3)(feedback)
        readout = CGReadoutLayer(32, 3, name="output")(states)
        model = ESNModel(feedback, readout)

        predictions = model.forecast(torch.randn(1, 20, 3), horizon=15)
        assert predictions.shape == (1, 15, 3)

        full = model.forecast(torch.randn(1, 20, 3), horizon=15, return_warmup=True)
        assert full.shape == (1, 35, 3)


class TestMultiOutputForecast:
    """The multi-output path mirrors the single-output autoregressive contract."""

    def test_multi_output_shapes(self) -> None:
        model = _build_multi_output_model(feedback_size=2, aux_size=3)
        warmup = torch.randn(1, 20, 2)
        horizon = 7

        model.reset_reservoirs()
        preds = model.forecast(warmup, horizon=horizon)

        assert isinstance(preds, tuple) and len(preds) == 2
        assert preds[0].shape == (1, horizon, 2)
        assert preds[1].shape == (1, horizon, 3)

    def test_multi_output_matches_manual_recompute(self) -> None:
        model = _build_multi_output_model(feedback_size=2, aux_size=3)
        warmup = torch.randn(1, 20, 2)
        horizon = 5

        model.reset_reservoirs()
        preds = model.forecast(warmup, horizon=horizon)

        model.reset_reservoirs()
        warm = model.warmup(warmup, return_outputs=True)
        assert isinstance(warm, tuple)
        current = warm[0][:, -1:, :]
        main_steps, aux_steps = [], []
        with torch.no_grad():
            for _ in range(horizon):
                out = model(current)
                main_steps.append(out[0])
                aux_steps.append(out[1])
                current = out[0]

        assert torch.allclose(preds[0], torch.cat(main_steps, dim=1), atol=1e-6)
        assert torch.allclose(preds[1], torch.cat(aux_steps, dim=1), atol=1e-6)

    def test_multi_output_return_warmup_no_duplicate_seam(self) -> None:
        model = _build_multi_output_model(feedback_size=2, aux_size=3)
        warmup_steps, horizon = 20, 6
        warmup = torch.randn(1, warmup_steps, 2)

        model.reset_reservoirs()
        full = model.forecast(warmup, horizon=horizon, return_warmup=True)

        assert isinstance(full, tuple) and len(full) == 2
        assert full[0].shape == (1, warmup_steps + horizon, 2)
        assert full[1].shape == (1, warmup_steps + horizon, 3)
        # Feedback channel seam must not be duplicated.
        assert not torch.allclose(
            full[0][:, warmup_steps - 1, :], full[0][:, warmup_steps, :], atol=1e-6
        )
