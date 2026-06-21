"""Windowed (gap-filling) reconstruction tests.

The contract under test (see :meth:`resdag.core.ESNModel.windowed_forecast`):

- the reservoir is reset at most once (only on the first window, only when
  ``reset=True``) and its state is carried across the whole pass;
- observed (teacher-forced) segments are copied verbatim from ``series``;
- gap segments are genuine autoregressive forecasts continuing from the
  re-synchronized state — identical to chaining :meth:`forecast` by hand;
- the returned mask marks observed steps True and forecast steps False.
"""

import pytest
import torch

from resdag.core import ESNModel, Input
from resdag.layers import CGReadoutLayer, ESNLayer


def _build_driven_model(feedback_size: int = 1, driver_size: int = 2) -> ESNModel:
    torch.manual_seed(0)
    feedback = Input(shape=(20, feedback_size))
    driver = Input(shape=(20, driver_size))
    states = ESNLayer(
        reservoir_size=32, feedback_size=feedback_size, input_size=driver_size, spectral_radius=0.9
    )(feedback, driver)
    readout = CGReadoutLayer(32, feedback_size, name="output")(states)
    return ESNModel([feedback, driver], readout)


class TestShapesAndMask:
    def test_reconstruction_shape_and_mask_layout(self, make_tiny_model) -> None:
        model = make_tiny_model(feedback_size=3)
        T = 200
        series = torch.randn(1, T, 3)
        warmup_len, teacher_len, predict_len = 20, 10, 30

        recon, mask = model.windowed_forecast(
            series,
            warmup_len=warmup_len,
            teacher_len=teacher_len,
            predict_len=predict_len,
            return_mask=True,
        )

        assert recon.shape == series.shape
        assert mask.shape == (T,)
        assert mask.dtype == torch.bool

        # First ``warmup_len`` steps are observed; then a gap of ``predict_len``.
        assert mask[:warmup_len].all()
        assert not mask[warmup_len : warmup_len + predict_len].any()
        # Next re-sync window is observed again.
        nxt = warmup_len + predict_len
        assert mask[nxt : nxt + teacher_len].all()

    def test_default_warmup_len_equals_teacher_len(self, make_tiny_model) -> None:
        model = make_tiny_model(feedback_size=3)
        series = torch.randn(1, 150, 3)

        _, mask = model.windowed_forecast(series, teacher_len=15, predict_len=25, return_mask=True)
        # First observed run is teacher_len long when warmup_len is omitted.
        assert mask[:15].all()
        assert not mask[15:40].any()

    def test_return_mask_false_returns_tensor_only(self, make_tiny_model) -> None:
        model = make_tiny_model(feedback_size=3)
        out = model.windowed_forecast(torch.randn(1, 120, 3), teacher_len=10, predict_len=20)
        assert isinstance(out, torch.Tensor)


class TestObservedSegmentsAreExact:
    def test_observed_steps_copy_series_verbatim(self, make_tiny_model) -> None:
        model = make_tiny_model(feedback_size=3)
        series = torch.randn(1, 220, 3)

        recon, mask = model.windowed_forecast(
            series, warmup_len=20, teacher_len=12, predict_len=40, return_mask=True
        )
        assert torch.equal(recon[:, mask, :], series[:, mask, :])

    def test_gap_steps_differ_from_series(self, make_tiny_model) -> None:
        model = make_tiny_model(feedback_size=3)
        series = torch.randn(1, 200, 3)

        recon, mask = model.windowed_forecast(
            series, warmup_len=20, teacher_len=12, predict_len=40, return_mask=True
        )
        # Forecasts on the gaps should not coincide with the (unseen) truth.
        assert not torch.allclose(recon[:, ~mask, :], series[:, ~mask, :], atol=1e-6)


class TestMatchesManualChaining:
    """A windowed pass equals chaining forecast() by hand with carried state."""

    def test_matches_hand_rolled_chain(self, make_tiny_model) -> None:
        model = make_tiny_model(feedback_size=3)
        series = torch.randn(1, 180, 3)
        warmup_len, teacher_len, predict_len = 20, 10, 30

        recon = model.windowed_forecast(
            series, warmup_len=warmup_len, teacher_len=teacher_len, predict_len=predict_len
        )

        # Hand-rolled: reset once, then forecast cycle by cycle with reset=False.
        manual = series.clone()
        pos, window_len, first = 0, warmup_len, True
        T = series.shape[1]
        while True:
            teacher_end = pos + window_len
            if teacher_end >= T:
                break
            gap = min(predict_len, T - teacher_end)
            if gap < 1:
                break
            preds = model.forecast(series[:, pos:teacher_end, :], horizon=gap, reset=first)
            manual[:, teacher_end : teacher_end + gap, :] = preds
            first = False
            pos = teacher_end + gap
            window_len = teacher_len

        assert torch.allclose(recon, manual, atol=1e-6)

    def test_single_cycle_reduces_to_forecast(self, make_tiny_model) -> None:
        """With one gap that runs to the end, it equals a plain forecast."""
        model = make_tiny_model(feedback_size=3)
        T, warmup_len = 100, 20
        series = torch.randn(1, T, 3)
        predict_len = T - warmup_len  # exactly one gap, no room for a second window

        recon, mask = model.windowed_forecast(
            series, warmup_len=warmup_len, teacher_len=15, predict_len=predict_len, return_mask=True
        )

        model.reset_reservoirs()
        plain = model.forecast(series[:, :warmup_len, :], horizon=T - warmup_len)

        assert torch.allclose(recon[:, warmup_len:, :], plain, atol=1e-6)
        assert mask[:warmup_len].all()
        assert not mask[warmup_len:].any()


class TestResetSemantics:
    def test_reset_true_ignores_prior_state(self, make_tiny_model) -> None:
        """reset=True wipes any pre-existing state, so the result is invariant."""
        model = make_tiny_model(feedback_size=3)
        series = torch.randn(1, 150, 3)
        kwargs = dict(warmup_len=20, teacher_len=10, predict_len=30)

        model.reset_reservoirs()
        model.warmup(torch.randn(1, 40, 3))  # leave a non-trivial state
        a = model.windowed_forecast(series, reset=True, **kwargs)

        model.reset_reservoirs()
        model.set_random_reservoir_states(batch_size=1)  # a different state
        b = model.windowed_forecast(series, reset=True, **kwargs)

        assert torch.allclose(a, b, atol=1e-6)

    def test_reset_false_depends_on_prior_state(self, make_tiny_model) -> None:
        """reset=False carries the prior state in.

        A single-step warmup window is used so the Echo State Property has not
        yet washed out the carried-in state (a longer window would, by design,
        re-synchronize both runs to the same state).
        """
        model = make_tiny_model(feedback_size=3)
        series = torch.randn(1, 60, 3)
        kwargs = dict(warmup_len=1, teacher_len=1, predict_len=20)

        _, mask = model.windowed_forecast(series, return_mask=True, **kwargs)

        model.reset_reservoirs()
        model.warmup(torch.randn(1, 40, 3))
        a = model.windowed_forecast(series, reset=False, **kwargs)

        model.reset_reservoirs()
        model.set_random_reservoir_states(batch_size=1)
        b = model.windowed_forecast(series, reset=False, **kwargs)

        assert not torch.allclose(a[:, ~mask, :], b[:, ~mask, :], atol=1e-6)


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(teacher_len=10, predict_len=0), "predict_len"),
            (dict(teacher_len=0, predict_len=10), "teacher_len"),
            (dict(teacher_len=10, predict_len=10, warmup_len=0), "warmup_len"),
        ],
    )
    def test_non_positive_lengths_raise(self, make_tiny_model, kwargs, match) -> None:
        model = make_tiny_model(feedback_size=3)
        with pytest.raises(ValueError, match=match):
            model.windowed_forecast(torch.randn(1, 100, 3), **kwargs)

    def test_non_3d_series_raises(self, make_tiny_model) -> None:
        model = make_tiny_model(feedback_size=3)
        with pytest.raises(ValueError, match="series must be 3-D"):
            model.windowed_forecast(torch.randn(100, 3), teacher_len=10, predict_len=10)

    def test_series_too_short_raises(self, make_tiny_model) -> None:
        # warmup_len reaches the end of the series, leaving no room to forecast.
        model = make_tiny_model(feedback_size=3)
        with pytest.raises(ValueError, match="too short"):
            model.windowed_forecast(
                torch.randn(1, 20, 3), warmup_len=20, teacher_len=10, predict_len=30
            )


class TestDrivers:
    def test_driven_model_reconstructs_with_drivers(self) -> None:
        model = _build_driven_model(feedback_size=1, driver_size=2)
        T = 160
        feedback = torch.randn(1, T, 1)
        driver = torch.randn(1, T, 2)

        recon, mask = model.windowed_forecast(
            feedback, driver, warmup_len=20, teacher_len=10, predict_len=30, return_mask=True
        )
        assert recon.shape == feedback.shape
        assert torch.equal(recon[:, mask, :], feedback[:, mask, :])

    def test_driver_must_span_full_timeline(self) -> None:
        model = _build_driven_model()
        feedback = torch.randn(1, 160, 1)
        short_driver = torch.randn(1, 100, 2)
        with pytest.raises(ValueError, match="must cover the full timeline"):
            model.windowed_forecast(feedback, short_driver, teacher_len=10, predict_len=30)

    def test_extra_driver_for_feedback_only_model_raises(self, make_tiny_model) -> None:
        model = make_tiny_model(feedback_size=3)  # no driver inputs
        with pytest.raises(ValueError, match="driver_series were given"):
            model.windowed_forecast(
                torch.randn(1, 160, 3),
                torch.randn(1, 160, 2),
                teacher_len=10,
                predict_len=30,
            )

    def test_missing_driver_for_driven_model_raises(self) -> None:
        model = _build_driven_model()  # expects 1 driver
        with pytest.raises(ValueError, match="driver_series were given"):
            model.windowed_forecast(torch.randn(1, 160, 1), teacher_len=10, predict_len=30)


class TestMultiOutputAndBatch:
    def test_multi_output_reconstructs_feedback_channel(self) -> None:
        """For multi-output models only the fed-back (first) channel is rebuilt."""
        torch.manual_seed(0)
        feedback = Input(shape=(20, 2))
        states = ESNLayer(reservoir_size=32, feedback_size=2, spectral_radius=0.9)(feedback)
        main = CGReadoutLayer(32, 2, name="main")(states)
        aux = CGReadoutLayer(32, 3, name="aux")(states)
        model = ESNModel(feedback, [main, aux])

        series = torch.randn(1, 160, 2)
        recon, mask = model.windowed_forecast(
            series, warmup_len=20, teacher_len=10, predict_len=30, return_mask=True
        )
        # Reconstruction matches the feedback channel shape (not the aux output).
        assert recon.shape == series.shape
        assert torch.equal(recon[:, mask, :], series[:, mask, :])
        assert not torch.allclose(recon[:, ~mask, :], series[:, ~mask, :], atol=1e-6)

    def test_batch_greater_than_one(self, make_tiny_model) -> None:
        model = make_tiny_model(feedback_size=3)
        series = torch.randn(4, 150, 3)
        recon, mask = model.windowed_forecast(
            series, warmup_len=20, teacher_len=10, predict_len=30, return_mask=True
        )
        assert recon.shape == (4, 150, 3)
        assert torch.equal(recon[:, mask, :], series[:, mask, :])
