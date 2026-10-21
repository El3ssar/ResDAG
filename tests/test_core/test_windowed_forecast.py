"""Windowed (gap-filling) reconstruction tests.

The contract under test (see :meth:`resdag.core.ESNModel.windowed_forecast`):

- the reservoir is reset at most once (only on the first window, only when
  ``reset=True``) and its state is carried across the whole pass;
- observed (teacher-forced) segments are copied verbatim from ``series``;
- gap segments are genuine autoregressive forecasts continuing from the
  re-synchronized state — equal to chaining :meth:`forecast` by hand once the
  seam convention is honoured: a gap ``[teacher_end, teacher_end + gap)`` is the
  prediction of ``series[teacher_end .. teacher_end + gap - 1]``, so it is filled
  from the warmup-included forecast as ``full[:, window_len - 1 : ...]`` (the
  last warmup output followed by all but the final AR slot), not from the raw AR
  block, which is shifted one step ahead;
- the returned mask marks observed steps True and forecast steps False.
"""

import pytest
import torch

from resdag.core import ESNModel, Input
from resdag.layers import CGReadoutLayer, Concatenate, ESNLayer


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
    """A windowed pass equals chaining forecast() by hand with carried state.

    The seam convention (proven in :class:`TestGapAlignmentIsExact`): a forecast
    of horizon ``gap`` seeded on ``series[pos:teacher_end]`` predicts indices
    ``teacher_end + 1 .. teacher_end + gap`` in its autoregressive slots, while
    its *last warmup output* predicts ``series[teacher_end]``.  So the gap
    ``[teacher_end, teacher_end + gap)`` — the prediction of
    ``series[teacher_end .. teacher_end + gap - 1]`` — is filled from the
    warmup-included output as ``full[:, window_len - 1 : window_len + gap - 1]``,
    dropping the trailing AR slot (which predicts one step past the gap).
    """

    def test_matches_hand_rolled_chain(self, make_tiny_model) -> None:
        model = make_tiny_model(feedback_size=3)
        series = torch.randn(1, 180, 3)
        warmup_len, teacher_len, predict_len = 20, 10, 30

        recon = model.windowed_forecast(
            series, warmup_len=warmup_len, teacher_len=teacher_len, predict_len=predict_len
        )

        # Independent compatibility oracle: reset once, use the historical
        # shifted rollout and include its boundary prediction. This also checks
        # that state consumes the final gap input before the next teacher window.
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
            full = model.forecast(
                series[:, pos:teacher_end, :],
                horizon=gap,
                return_warmup=True,
                reset=first,
                alignment="legacy",
            )
            manual[:, teacher_end : teacher_end + gap, :] = full[
                :, window_len - 1 : window_len + gap - 1, :
            ]
            first = False
            pos = teacher_end + gap
            window_len = teacher_len

        assert torch.allclose(recon, manual, atol=1e-6)

    def test_single_cycle_equals_default_forecast(self, make_tiny_model) -> None:
        """A single gap directly equals the next-sample forecast window."""
        model = make_tiny_model(feedback_size=3)
        T, warmup_len = 100, 20
        series = torch.randn(1, T, 3)
        predict_len = T - warmup_len  # exactly one gap, no room for a second window

        recon, mask = model.windowed_forecast(
            series, warmup_len=warmup_len, teacher_len=15, predict_len=predict_len, return_mask=True
        )

        gap = T - warmup_len
        model.reset_reservoirs()
        expected_gap = model.forecast(series[:, :warmup_len, :], horizon=gap)

        assert torch.allclose(recon[:, warmup_len:, :], expected_gap, atol=1e-6)
        assert mask[:warmup_len].all()
        assert not mask[warmup_len:].any()


def _exact_linear_model(A: torch.Tensor, reservoir_size: int = 16) -> ESNModel:
    """A model whose output is exactly ``A @ feedback_t``, reservoir-independent.

    The readout reads ``[feedback, reservoir_states]`` but its reservoir block is
    zeroed and its feedback block is set to ``A``, so ``model(x_t) == x_t @ A.T``
    for every ``t`` regardless of the (still genuinely evolving) reservoir state.
    This turns the seam convention into an exact, closed-form check: for a series
    with ``series[t + 1] == series[t] @ A.T`` the correct gap reconstruction is
    bit-exact, so any one-step misalignment shows up as a hard failure rather
    than a soft chaotic-drift tolerance.
    """
    F = A.shape[0]
    inp = Input(shape=(20, F))
    states = ESNLayer(reservoir_size=reservoir_size, feedback_size=F, spectral_radius=0.9)(inp)
    cat = Concatenate()(inp, states)
    readout = CGReadoutLayer(F + reservoir_size, F, name="output", bias=False)(cat)
    model = ESNModel(inp, readout)
    ro = next(m for m in model.modules() if isinstance(m, CGReadoutLayer))
    with torch.no_grad():
        weight = torch.zeros(F, F + reservoir_size)
        weight[:, :F] = A
        ro.weight.copy_(weight)
        ro._is_fitted.fill_(True)
    return model


def _linear_series(A: torch.Tensor, total_steps: int, seed: int = 0) -> torch.Tensor:
    """Series obeying ``series[t + 1] = series[t] @ A.T`` exactly."""
    torch.manual_seed(seed)
    F = A.shape[0]
    steps = [torch.randn(1, 1, F)]
    for _ in range(total_steps - 1):
        steps.append(steps[-1] @ A.T)
    return torch.cat(steps, dim=1)


class TestGapAlignmentIsExact:
    """Gap reconstruction lands on the right timestep (regression for the
    one-dt phase shift): with an exact linear map the filled gaps equal
    ``series`` at their *own* index, not one step ahead."""

    # A rotation-plus-decay map: deterministic, invertible-ish, non-trivial.
    A = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.5]])

    def test_forecast_seam_convention(self) -> None:
        """Forecast slot t predicts series[warmup + t], with no omitted sample."""
        torch.manual_seed(0)
        model = _exact_linear_model(self.A)
        series = _linear_series(self.A, total_steps=60)
        warmup_len, horizon = 20, 15

        model.reset_reservoirs()
        preds = model.forecast(series[:, :warmup_len, :], horizon=horizon)

        shifted = series[:, warmup_len + 1 : warmup_len + 1 + horizon, :]
        same = series[:, warmup_len : warmup_len + horizon, :]
        assert torch.allclose(preds, same, atol=1e-5)
        assert not torch.allclose(preds, shifted, atol=1e-2)
        legacy = model.forecast(series[:, :warmup_len, :], horizon=horizon, alignment="legacy")
        assert torch.allclose(legacy, shifted, atol=1e-5)

    def test_windowed_gap_matches_own_index(self) -> None:
        """Every filled gap value equals ``series`` at its own index (no shift)."""
        torch.manual_seed(0)
        model = _exact_linear_model(self.A)
        series = _linear_series(self.A, total_steps=120)

        recon, mask = model.windowed_forecast(
            series, warmup_len=20, teacher_len=10, predict_len=15, return_mask=True
        )
        gap = ~mask
        # Exact match at the SAME index — the corrected placement.
        assert torch.allclose(recon[:, gap, :], series[:, gap, :], atol=1e-5)

        # And it is NOT the one-step-early alignment the bug produced
        # (recon[i] == series[i + 1]); check the interior gap indices.
        gap_idx = torch.nonzero(gap).squeeze(-1)
        interior = gap_idx[gap_idx < series.shape[1] - 1]
        assert not torch.allclose(recon[:, interior, :], series[:, interior + 1, :], atol=1e-2)

    def test_driven_windowed_gap_matches_own_index(self) -> None:
        """Driver alignment is preserved: driven exact map still fills gaps exactly."""
        torch.manual_seed(0)
        F, driver_dim, reservoir_size = 2, 2, 16
        A = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
        B = torch.tensor([[0.3, 0.0], [0.0, 0.2]])

        fb = Input(shape=(20, F))
        drv = Input(shape=(20, driver_dim))
        states = ESNLayer(
            reservoir_size=reservoir_size,
            feedback_size=F,
            input_size=driver_dim,
            spectral_radius=0.9,
        )(fb, drv)
        cat = Concatenate()(fb, drv, states)  # order: feedback, driver, states
        readout = CGReadoutLayer(F + driver_dim + reservoir_size, F, name="output", bias=False)(cat)
        model = ESNModel([fb, drv], readout)
        ro = next(m for m in model.modules() if isinstance(m, CGReadoutLayer))
        with torch.no_grad():
            weight = torch.zeros(F, F + driver_dim + reservoir_size)
            weight[:, :F] = A
            weight[:, F : F + driver_dim] = B
            ro.weight.copy_(weight)
            ro._is_fitted.fill_(True)

        total_steps = 100
        driver = torch.randn(1, total_steps, driver_dim)
        steps = [torch.randn(1, 1, F)]
        for t in range(total_steps - 1):
            steps.append(steps[-1] @ A.T + driver[:, t : t + 1, :] @ B.T)
        series = torch.cat(steps, dim=1)  # series[t + 1] = A series[t] + B driver[t]

        recon, mask = model.windowed_forecast(
            series, driver, warmup_len=20, teacher_len=10, predict_len=12, return_mask=True
        )
        gap = ~mask
        assert torch.allclose(recon[:, gap, :], series[:, gap, :], atol=1e-5)


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
            (dict(teacher_len=10, predict_len=True), "predict_len"),
            (dict(teacher_len=1.5, predict_len=10), "teacher_len"),
            (dict(teacher_len=10, predict_len=10, warmup_len=2.5), "warmup_len"),
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
