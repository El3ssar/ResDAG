"""Tests for CoupledEnsembleESNModel and its aggregators."""

import tempfile
from pathlib import Path

import pytest
import torch

import resdag as rd
from resdag.core import ESNModel, Input
from resdag.ensemble import CoupledEnsembleESNModel
from resdag.ensemble.aggregators import OutliersFilteredMean
from resdag.layers import CGReadoutLayer, ESNLayer


def _driven_submodel_factory(
    *,
    reservoir_size: int = 24,
    feedback_size: int = 2,
    driver_size: int = 2,
    output_size: int = 2,
    seed: int | None = None,
    **_: object,
) -> ESNModel:
    """Build a single driving-input ESN sub-model for the coupled ensemble.

    The first (and only) readout matches ``feedback_size`` so the aggregated
    output can be fed back autoregressively, exactly like the single-model
    driven forecast tests.
    """
    if seed is not None:
        torch.manual_seed(seed)
    feedback = Input(shape=(20, feedback_size))
    driver = Input(shape=(20, driver_size))
    states = ESNLayer(
        reservoir_size=reservoir_size,
        feedback_size=feedback_size,
        input_size=driver_size,
        spectral_radius=0.9,
    )(feedback, driver)
    readout = CGReadoutLayer(reservoir_size, output_size, name="output")(states)
    return ESNModel([feedback, driver], readout)


def _toy_data(seq_len: int = 60, feature_size: int = 2):
    """Smooth synthetic 2-D time series."""
    torch.manual_seed(0)
    raw = torch.randn(1, seq_len + 1, feature_size)
    kernel = torch.tensor([0.25, 0.5, 0.25])
    out = torch.empty_like(raw)
    for d in range(feature_size):
        out[:, :, d] = torch.nn.functional.conv1d(
            raw[:, :, d].unsqueeze(1), kernel.view(1, 1, 3), padding=1
        ).squeeze(1)
    return out[:, :seq_len, :]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_factory_builds_n_sub_models(self):
        ens = rd.coupled_ensemble_esn(n_models=4, reservoir_size=20, feedback_size=2, output_size=2)
        assert isinstance(ens, CoupledEnsembleESNModel)
        assert ens.n_models == 4
        assert len(ens.models) == 4

    def test_empty_models_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            CoupledEnsembleESNModel(models=[], aggregator="mean")

    def test_unknown_aggregator_string_raises(self):
        models = [
            rd.classic_esn(reservoir_size=10, feedback_size=2, output_size=2) for _ in range(2)
        ]
        with pytest.raises(ValueError, match="aggregator"):
            CoupledEnsembleESNModel(models=models, aggregator="argmin")

    def test_seed_reproducibility(self):
        """Two ensembles built with the same seed have identical weights."""
        e1 = rd.coupled_ensemble_esn(
            n_models=3, reservoir_size=20, feedback_size=2, output_size=2, seed=11
        )
        e2 = rd.coupled_ensemble_esn(
            n_models=3, reservoir_size=20, feedback_size=2, output_size=2, seed=11
        )
        for m1, m2 in zip(e1.models, e2.models):
            cell1 = next(
                m for m in m1.modules() if hasattr(m, "weight_hh") and hasattr(m, "reservoir_size")
            )
            cell2 = next(
                m for m in m2.modules() if hasattr(m, "weight_hh") and hasattr(m, "reservoir_size")
            )
            assert torch.equal(cell1.weight_hh.data, cell2.weight_hh.data)

    def test_seeded_build_does_not_clobber_global_rng(self):
        """Building a seeded ensemble must leave the global RNG untouched.

        Regression for the bug where the factory called
        ``torch.manual_seed(seed + i)`` inside the construction loop, mutating
        the process-global default generator so a subsequent global
        ``torch.randn`` no longer matched the pre-construction draw.  The fix
        snapshots and restores the global RNG state, so ensemble construction
        is composable inside an otherwise-reproducible pipeline.
        """
        torch.manual_seed(2024)
        expected = torch.randn(8)

        # Re-seed to the same point, build a seeded ensemble, then draw again:
        # the draw must match the pre-construction value bit-for-bit.
        torch.manual_seed(2024)
        rd.coupled_ensemble_esn(
            n_models=3, reservoir_size=25, feedback_size=2, output_size=2, seed=7
        )
        after = torch.randn(8)
        assert torch.equal(expected, after)

    def test_global_rng_restored_even_when_factory_raises(self):
        """The global RNG is restored via ``finally`` even on a factory error."""

        def _boom(**_: object) -> ESNModel:
            raise RuntimeError("factory blew up")

        torch.manual_seed(1234)
        expected = torch.randn(4)

        torch.manual_seed(1234)
        with pytest.raises(RuntimeError, match="factory blew up"):
            rd.coupled_ensemble_esn(n_models=2, model_factory=_boom, seed=5)
        after = torch.randn(4)
        assert torch.equal(expected, after)

    def test_unseeded_build_advances_global_rng(self):
        """Without ``seed`` the factory's own draws still advance the global RNG.

        The save/restore guard only applies to the seeded path; an unseeded
        build is not expected to be RNG-neutral and should consume entropy
        normally.
        """
        torch.manual_seed(99)
        before = torch.randn(4)

        torch.manual_seed(99)
        rd.coupled_ensemble_esn(n_models=2, reservoir_size=20, feedback_size=2, output_size=2)
        after = torch.randn(4)
        assert not torch.equal(before, after)

    def test_headless_sub_models_rejected(self):
        """Headless sub-models (reservoir-state output) raise a clear ValueError."""
        from resdag.models import headless_esn

        models = [headless_esn(reservoir_size=20, feedback_size=3) for _ in range(2)]
        with pytest.raises(ValueError, match="Sub-model 0 is not autoregressively coupleable"):
            CoupledEnsembleESNModel(models)

    def test_linear_sub_models_rejected(self):
        """Linear (headless, identity-activation) sub-models are rejected too."""
        from resdag.models import linear_esn

        models = [linear_esn(reservoir_size=20, feedback_size=3) for _ in range(2)]
        with pytest.raises(ValueError, match="output dimension"):
            CoupledEnsembleESNModel(models)

    def test_headless_via_factory_rejected(self):
        """The factory path surfaces the same guard for headless factories."""
        from resdag.models import headless_esn

        with pytest.raises(ValueError, match="not autoregressively coupleable"):
            rd.coupled_ensemble_esn(
                n_models=2,
                model_factory=headless_esn,
                reservoir_size=20,
                feedback_size=3,
            )

    def test_output_feedback_mismatch_names_dims(self):
        """A readout-bearing model whose output != feedback dim is also rejected."""
        from resdag.core import Input
        from resdag.layers import CGReadoutLayer, ESNLayer

        def _mismatched(**_: object) -> ESNModel:
            feedback = Input(shape=(20, 3))
            states = ESNLayer(reservoir_size=20, feedback_size=3, spectral_radius=0.9)(feedback)
            readout = CGReadoutLayer(20, 2, name="output")(states)  # out=2 != feedback=3
            return ESNModel(feedback, readout)

        models = [_mismatched()]
        with pytest.raises(ValueError) as excinfo:
            CoupledEnsembleESNModel(models)
        msg = str(excinfo.value)
        assert "(2)" in msg and "(3)" in msg  # names both mismatched dims

    def test_classic_sub_models_accepted(self):
        """Readout-bearing classic sub-models pass the guard."""
        models = [
            rd.classic_esn(reservoir_size=20, feedback_size=3, output_size=3) for _ in range(2)
        ]
        ens = CoupledEnsembleESNModel(models)
        assert ens.n_models == 2

    def test_driven_sub_models_accepted(self):
        """Multi-input (driven) sub-models with matching output dim pass the guard."""
        models = [_driven_submodel_factory(seed=i) for i in range(2)]
        ens = CoupledEnsembleESNModel(models)
        assert ens.n_models == 2


# ---------------------------------------------------------------------------
# Fit / forecast
# ---------------------------------------------------------------------------


class TestFitForecast:
    def test_fit_and_forecast_mean(self):
        x = _toy_data()
        warmup, train, f_warm = x[:, :10], x[:, 10:40], x[:, 40:60]

        ens = rd.coupled_ensemble_esn(
            n_models=3,
            reservoir_size=30,
            feedback_size=2,
            output_size=2,
            seed=0,
        )
        ens.fit((warmup,), (train,), {"output": train.clone()})
        out = ens.forecast(f_warm, horizon=15)
        assert out.shape == (1, 15, 2)

    def test_forecast_return_individuals(self):
        x = _toy_data()
        warmup, train, f_warm = x[:, :10], x[:, 10:40], x[:, 40:60]

        ens = rd.coupled_ensemble_esn(
            n_models=4, reservoir_size=20, feedback_size=2, output_size=2, seed=0
        )
        ens.fit((warmup,), (train,), {"output": train.clone()})
        agg, indiv = ens.forecast(f_warm, horizon=10, return_individuals=True)
        assert agg.shape == (1, 10, 2)
        assert len(indiv) == 4
        for buf in indiv:
            assert buf.shape == (1, 10, 2)

    def test_forecast_return_warmup(self):
        # Next-step (shifted) target so the genuine first forecast frame differs
        # from the last warmup frame, making the seam-duplication check active.
        x = _toy_data(seq_len=80)
        warmup = x[:, :10]
        train_in, train_tgt = x[:, 10:49], x[:, 11:50]
        f_warm = x[:, 50:70]

        ens = rd.coupled_ensemble_esn(
            n_models=3, reservoir_size=20, feedback_size=2, output_size=2, seed=0
        )
        ens.fit((warmup,), (train_in,), {"output": train_tgt.clone()})

        warmup_steps = f_warm.shape[1]
        horizon = 20
        forecast_only = ens.forecast(f_warm, horizon=horizon)
        full = ens.forecast(f_warm, horizon=horizon, return_warmup=True)

        # Length is exactly warmup_steps + horizon.
        assert full.shape[1] == warmup_steps + horizon
        # The forecast tail of the return_warmup output equals the standalone
        # forecast — slot 0 is a genuine step, not a teacher-forced echo.
        assert torch.allclose(full[:, warmup_steps:, :], forecast_only)
        # No duplicated seam frame: the old buggy code wrote the aggregated
        # last-warmup echo into forecast slot 0, so full[:, warmup_steps] would
        # equal full[:, warmup_steps - 1]. With the fix they differ.
        assert not torch.allclose(full[:, warmup_steps - 1, :], full[:, warmup_steps, :], atol=1e-5)

    def test_forecast_slot0_is_genuine_step_not_warmup_echo(self):
        """Slot 0 is a genuine coupled step, not the aggregated warmup echo.

        Reproduces the corrected ESNModel.forecast semantics by hand: warm every
        sub-model, aggregate the last warmup step as the seed feedback, take one
        coupled step, and check ``forecast(..., horizon=1)`` matches that — *not*
        the seed feedback itself (which is what the old off-by-one returned).

        The readout is fitted to a *next-step* (shifted) target so the genuine
        step provably differs from the seed echo, making the regression check
        below active rather than vacuous.
        """
        x = _toy_data(seq_len=80)
        warmup = x[:, :10]
        train_in, train_tgt = x[:, 10:49], x[:, 11:50]  # predict next step
        f_warm = x[:, 50:70]

        ens = rd.coupled_ensemble_esn(
            n_models=3, reservoir_size=20, feedback_size=2, output_size=2, seed=0
        )
        ens.fit((warmup,), (train_in,), {"output": train_tgt.clone()})

        # Manual one-step coupled forecast from the warmed state.
        last_steps = []
        with torch.no_grad():
            for m in ens.models:
                w = m.warmup(f_warm, return_outputs=True, reset=True)
                last_steps.append(w[:, -1:, :])
            seed_feedback = torch.stack(last_steps, dim=0).mean(dim=0)  # aggregated echo
            step_outputs = [m(seed_feedback) for m in ens.models]
            expected_step = torch.stack(step_outputs, dim=0).mean(dim=0).squeeze(1)

        out = ens.forecast(f_warm, horizon=1)
        assert out.shape == (1, 1, 2)
        # Slot 0 equals the genuine coupled step ...
        assert torch.allclose(out[:, 0, :], expected_step, atol=1e-5)
        # ... and the genuine step differs from the seed echo (sanity check that
        # this test would catch the old off-by-one) ...
        assert not torch.allclose(expected_step, seed_feedback.squeeze(1), atol=1e-5)
        # ... so slot 0 is not the old teacher-forced warmup echo.
        assert not torch.allclose(out[:, 0, :], seed_feedback.squeeze(1), atol=1e-5)

    def test_forecast_rejects_non_positive_horizon(self):
        """``horizon <= 0`` raises a ValueError naming the parameter."""
        x = _toy_data()
        warmup, train, f_warm = x[:, :10], x[:, 10:40], x[:, 40:60]

        ens = rd.coupled_ensemble_esn(
            n_models=2, reservoir_size=20, feedback_size=2, output_size=2, seed=0
        )
        ens.fit((warmup,), (train,), {"output": train.clone()})

        for bad in (0, -1):
            with pytest.raises(ValueError, match="horizon"):
                ens.forecast(f_warm, horizon=bad)

    def test_forecast_return_individuals_buffers_are_genuine_steps(self):
        """Per-model buffers have shape ``(batch, horizon, F)`` and no slot-0 echo.

        Each per-model slot 0 is that sub-model's own forecast of the seed
        feedback, *not* the aggregated warmup echo that the old code wrote in.
        """
        x = _toy_data(seq_len=80)
        warmup = x[:, :10]
        train_in, train_tgt = x[:, 10:49], x[:, 11:50]  # predict next step
        f_warm = x[:, 50:70]

        ens = rd.coupled_ensemble_esn(
            n_models=3, reservoir_size=20, feedback_size=2, output_size=2, seed=0
        )
        ens.fit((warmup,), (train_in,), {"output": train_tgt.clone()})

        # Manual seed feedback (aggregated last warmup) + per-model first steps.
        last_steps = []
        with torch.no_grad():
            for m in ens.models:
                w = m.warmup(f_warm, return_outputs=True, reset=True)
                last_steps.append(w[:, -1:, :])
            seed_feedback = torch.stack(last_steps, dim=0).mean(dim=0)
            expected_first = [m(seed_feedback).squeeze(1) for m in ens.models]

        _, indiv = ens.forecast(f_warm, horizon=8, return_individuals=True)
        assert len(indiv) == 3
        for buf, exp in zip(indiv, expected_first):
            assert buf.shape == (1, 8, 2)
            # Slot 0 is the sub-model's genuine first forecast step ...
            assert torch.allclose(buf[:, 0, :], exp, atol=1e-5)
            # ... and not the aggregated warmup echo (old slot-0 pre-fill).
            assert not torch.allclose(buf[:, 0, :], seed_feedback.squeeze(1), atol=1e-5)

    def test_fit_parallel_n_workers(self):
        x = _toy_data()
        warmup, train = x[:, :10], x[:, 10:40]

        # Same seed → fits should yield bit-identical results sequential vs
        # threaded (CG solve is deterministic for fixed inputs/seeds).
        ens_seq = rd.coupled_ensemble_esn(
            n_models=2, reservoir_size=20, feedback_size=2, output_size=2, seed=99
        )
        ens_par = rd.coupled_ensemble_esn(
            n_models=2, reservoir_size=20, feedback_size=2, output_size=2, seed=99
        )
        ens_seq.fit((warmup,), (train,), {"output": train.clone()})
        ens_par.fit((warmup,), (train,), {"output": train.clone()}, n_workers=2)

        for m1, m2 in zip(ens_seq.models, ens_par.models):
            for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
                assert n1 == n2
                assert torch.allclose(
                    p1, p2, atol=1e-6, rtol=1e-5
                ), f"{n1}: sequential vs parallel diverged"

    def test_fit_invalid_n_workers(self):
        ens = rd.coupled_ensemble_esn(n_models=2, reservoir_size=10, feedback_size=2, output_size=2)
        x = _toy_data()
        with pytest.raises(ValueError, match="n_workers"):
            ens.fit((x[:, :10],), (x[:, 10:30],), {"output": x[:, 10:30].clone()}, n_workers=0)


# ---------------------------------------------------------------------------
# Device / dtype coercion (fit + forecast)
# ---------------------------------------------------------------------------


class _DtypeChangingAggregator(torch.nn.Module):
    """Mean aggregator that returns a different dtype than its input.

    Used to exercise the deliberate-cast path in ``forecast``: a custom
    aggregator whose output dtype differs from the pre-allocated buffer must be
    cast on purpose, not silently truncated by the in-place write.
    """

    def __init__(self, out_dtype: torch.dtype) -> None:
        super().__init__()
        self.out_dtype = out_dtype

    def forward(self, stacked: torch.Tensor) -> torch.Tensor:
        return stacked.mean(dim=0).to(dtype=self.out_dtype)


class TestDeviceDtypeCoercion:
    """``fit``/``forecast`` validate or coerce device/dtype against sub-models."""

    def test_fit_mismatched_dtype_target_raises_clear_error(self):
        """A target on a different dtype raises a clear, named error.

        Sub-models are float32 (the resdag default); feeding a float64 target
        must surface a clear ValueError naming the offending tensor and both
        dtypes, not a raw low-level error from the readout solve.
        """
        x = _toy_data()
        warmup, train = x[:, :10], x[:, 10:40]

        ens = rd.coupled_ensemble_esn(
            n_models=2, reservoir_size=20, feedback_size=2, output_size=2, seed=0
        )
        bad_target = train.clone().to(dtype=torch.float64)
        with pytest.raises(ValueError, match=r"targets\['output'\].*float64.*float32"):
            ens.fit((warmup,), (train,), {"output": bad_target})

    def test_fit_mismatched_dtype_input_raises_clear_error(self):
        """A warmup/train input on a different dtype is named in the error."""
        x = _toy_data()
        warmup, train = x[:, :10], x[:, 10:40]

        ens = rd.coupled_ensemble_esn(
            n_models=2, reservoir_size=20, feedback_size=2, output_size=2, seed=0
        )
        with pytest.raises(ValueError, match=r"train_inputs\[0\].*float64.*float32"):
            ens.fit(
                (warmup,),
                (train.to(dtype=torch.float64),),
                {"output": train.clone()},
            )

    def test_fit_coerce_true_accepts_mismatched_dtype(self):
        """``coerce=True`` casts mismatched inputs/targets instead of raising."""
        x = _toy_data()
        warmup, train = x[:, :10], x[:, 10:40]

        ens = rd.coupled_ensemble_esn(
            n_models=2, reservoir_size=20, feedback_size=2, output_size=2, seed=0
        )
        # All three float64 — coerce them all back to the float32 sub-models.
        ens.fit(
            (warmup.to(dtype=torch.float64),),
            (train.to(dtype=torch.float64),),
            {"output": train.clone().to(dtype=torch.float64)},
            coerce=True,
        )
        # The readout was fitted (no error) and produces a float32 forecast.
        out = ens.forecast(x[:, 40:60], horizon=8)
        assert out.dtype == torch.float32
        assert out.shape == (1, 8, 2)

    def test_fit_matching_dtype_is_unchanged(self):
        """The common matching-dtype path is a no-op and trains normally."""
        x = _toy_data()
        warmup, train, f_warm = x[:, :10], x[:, 10:40], x[:, 40:60]

        ens = rd.coupled_ensemble_esn(
            n_models=2, reservoir_size=20, feedback_size=2, output_size=2, seed=0
        )
        ens.fit((warmup,), (train,), {"output": train.clone()})
        out = ens.forecast(f_warm, horizon=10)
        assert out.shape == (1, 10, 2)

    def test_forecast_dtype_changing_aggregator_is_cast_not_truncated(self):
        """A custom aggregator returning float64 is cast to the float32 buffer.

        The aggregated step (float64) must be deliberately cast to the buffer
        dtype (float32) on the in-place write — the result is float32 and finite,
        proving the value was cast rather than silently reinterpreted/truncated.
        """
        x = _toy_data()
        warmup, train, f_warm = x[:, :10], x[:, 10:40], x[:, 40:60]

        ens = rd.coupled_ensemble_esn(
            n_models=3,
            reservoir_size=20,
            feedback_size=2,
            output_size=2,
            aggregate=_DtypeChangingAggregator(torch.float64),
            seed=0,
        )
        ens.fit((warmup,), (train,), {"output": train.clone()})
        out = ens.forecast(f_warm, horizon=10)
        assert out.dtype == torch.float32  # cast to the buffer dtype
        assert out.shape == (1, 10, 2)
        assert torch.isfinite(out).all()

    def test_forecast_dtype_changing_aggregator_return_warmup(self):
        """The ``return_warmup`` concat also conforms a dtype-changing aggregator."""
        x = _toy_data()
        warmup, train, f_warm = x[:, :10], x[:, 10:40], x[:, 40:60]

        ens = rd.coupled_ensemble_esn(
            n_models=3,
            reservoir_size=20,
            feedback_size=2,
            output_size=2,
            aggregate=_DtypeChangingAggregator(torch.float64),
            seed=0,
        )
        ens.fit((warmup,), (train,), {"output": train.clone()})
        out = ens.forecast(f_warm, horizon=8, return_warmup=True)
        assert out.dtype == torch.float32
        assert out.shape[1] == f_warm.shape[1] + 8


# ---------------------------------------------------------------------------
# Autograd isolation (no_grad on forecast / warmup)
# ---------------------------------------------------------------------------


def _trainable_ensemble(n_models: int = 2, seed: int = 0) -> CoupledEnsembleESNModel:
    """Build a coupled ensemble whose sub-models carry trainable parameters.

    ``classic_esn(trainable=True)`` makes every sub-model's reservoir weights
    ``requires_grad=True`` — the exact condition under which an un-guarded
    autoregressive loop would build a full-horizon autograd graph.
    """
    return rd.coupled_ensemble_esn(
        n_models=n_models,
        model_factory=rd.classic_esn,
        reservoir_size=20,
        feedback_size=2,
        output_size=2,
        trainable=True,
        seed=seed,
    )


class TestForecastNoGrad:
    """``forecast``/``warmup`` are wrapped in ``@torch.no_grad()``.

    Without the guard, every sub-model call inside the autoregressive loop
    requires grad (trainable reservoirs), so the loop accumulates a
    full-horizon autograd graph — leaking memory linearly in ``horizon`` and
    risking version-counter corruption on the in-place ``torch.empty`` writes.
    These tests pin the guard via structural ``requires_grad`` / ``grad_fn``
    assertions, which hold regardless of available memory.
    """

    def test_submodels_actually_trainable(self):
        """Guard the guard: the fixture really does have trainable parameters.

        If this regressed to non-trainable sub-models the no-grad assertions
        below would pass vacuously, so assert the precondition explicitly.
        """
        ens = _trainable_ensemble()
        trainable = [p for m in ens.models for p in m.parameters() if p.requires_grad]
        assert len(trainable) > 0

    def test_forecast_output_has_no_grad(self):
        """``forecast`` output is detached even with trainable sub-models."""
        x = _toy_data()
        warmup, train, f_warm = x[:, :10], x[:, 10:40], x[:, 40:60]

        ens = _trainable_ensemble()
        ens.fit((warmup,), (train,), {"output": train.clone()})

        out = ens.forecast(f_warm, horizon=15)
        assert out.requires_grad is False
        assert out.grad_fn is None

    def test_forecast_individuals_have_no_grad(self):
        """Per-model trajectories from ``return_individuals`` are also detached."""
        x = _toy_data()
        warmup, train, f_warm = x[:, :10], x[:, 10:40], x[:, 40:60]

        ens = _trainable_ensemble(n_models=3)
        ens.fit((warmup,), (train,), {"output": train.clone()})

        agg, indiv = ens.forecast(f_warm, horizon=10, return_individuals=True)
        assert agg.requires_grad is False and agg.grad_fn is None
        for buf in indiv:
            assert buf.requires_grad is False
            assert buf.grad_fn is None

    def test_forecast_return_warmup_has_no_grad(self):
        """The warmup-prepended forecast output is detached too."""
        x = _toy_data()
        warmup, train, f_warm = x[:, :10], x[:, 10:40], x[:, 40:60]

        ens = _trainable_ensemble()
        ens.fit((warmup,), (train,), {"output": train.clone()})

        out = ens.forecast(f_warm, horizon=12, return_warmup=True)
        assert out.requires_grad is False
        assert out.grad_fn is None

    def test_warmup_does_not_accumulate_graph(self):
        """``warmup`` advances reservoir states without retaining a graph.

        After a teacher-forced warmup with trainable sub-models, the stored
        reservoir states must be detached — a retained ``grad_fn`` here would
        be the leaked graph the guard is meant to prevent.
        """
        x = _toy_data()
        warmup, train, f_warm = x[:, :10], x[:, 10:40], x[:, 40:60]

        ens = _trainable_ensemble()
        ens.fit((warmup,), (train,), {"output": train.clone()})

        ens.warmup(f_warm)
        for state_dict in ens.get_reservoir_states():
            for state in state_dict.values():
                assert state.requires_grad is False
                assert state.grad_fn is None

    def test_long_horizon_forecast_retains_no_graph(self):
        """A long-horizon forecast retains no autograd graph (no per-step growth).

        A structural ``grad_fn``/``requires_grad`` check is sufficient: with the
        guard the loop never links steps into a graph, so memory cannot grow
        with ``horizon``. Without it, the output would carry a ``grad_fn``
        chaining all ``horizon`` steps.
        """
        x = _toy_data()
        warmup, train, f_warm = x[:, :10], x[:, 10:40], x[:, 40:60]

        ens = _trainable_ensemble()
        ens.fit((warmup,), (train,), {"output": train.clone()})

        out = ens.forecast(f_warm, horizon=200)
        assert out.shape == (1, 200, 2)
        assert out.requires_grad is False
        assert out.grad_fn is None

    def test_forecast_does_not_disable_grad_globally(self):
        """The guard is scoped: grad tracking is restored after ``forecast``."""
        x = _toy_data()
        warmup, train, f_warm = x[:, :10], x[:, 10:40], x[:, 40:60]

        ens = _trainable_ensemble()
        ens.fit((warmup,), (train,), {"output": train.clone()})

        assert torch.is_grad_enabled()
        ens.forecast(f_warm, horizon=5)
        assert torch.is_grad_enabled()


# ---------------------------------------------------------------------------
# Driven (input-driven) forecast
# ---------------------------------------------------------------------------


def _reservoir_layer_of(model: ESNModel) -> ESNLayer:
    """Return the (single) ESN reservoir layer inside a sub-model."""
    return next(m for m in model.modules() if isinstance(m, ESNLayer))  # type: ignore[return-value]


class TestDrivenForecast:
    """Driver alignment in the coupled loop matches ESNModel.forecast."""

    def _build_driven_ensemble(self, n_models: int = 3) -> CoupledEnsembleESNModel:
        ens = rd.coupled_ensemble_esn(
            n_models=n_models,
            model_factory=_driven_submodel_factory,
            seed=0,
        )
        x = _toy_data()
        warmup, train = x[:, :10], x[:, 10:40]
        driver_w = torch.randn(1, warmup.shape[1], 2)
        driver_t = torch.randn(1, train.shape[1], 2)
        ens.fit((warmup, driver_w), (train, driver_t), {"output": train.clone()})
        return ens

    def test_driven_forecast_shape(self):
        ens = self._build_driven_ensemble()
        x = _toy_data()
        f_warm = x[:, 40:60]
        warm_driver = torch.randn(1, f_warm.shape[1], 2)
        horizon = 12
        forecast_driver = torch.randn(1, horizon, 2)
        out = ens.forecast(
            (f_warm, warm_driver),
            forecast_inputs=(forecast_driver,),
            horizon=horizon,
        )
        assert out.shape == (1, horizon, 2)

    def test_driven_step_consumes_forecast_inputs_t(self):
        """Step ``t`` consumes ``forecast_inputs[:, t]`` for every sub-model."""
        ens = self._build_driven_ensemble(n_models=2)
        x = _toy_data()
        f_warm = x[:, 40:60]
        warm_driver = (
            torch.arange(f_warm.shape[1], dtype=torch.float32)
            .view(1, -1, 1)
            .expand(1, f_warm.shape[1], 2)
        )
        horizon = 6
        forecast_driver = (
            (100 + torch.arange(horizon, dtype=torch.float32))
            .view(1, horizon, 1)
            .expand(1, horizon, 2)
        )

        # Probe the first sub-model's reservoir layer.
        reservoir_layer = _reservoir_layer_of(ens.models[0])
        seen: list[torch.Tensor] = []
        handle = reservoir_layer.register_forward_pre_hook(
            lambda module, args: seen.append(args[1].detach().clone())
        )
        try:
            ens.forecast(
                (f_warm, warm_driver.contiguous()),
                forecast_inputs=(forecast_driver.contiguous(),),
                horizon=horizon,
            )
        finally:
            handle.remove()

        # First captured call is the teacher-forced warmup pass.
        assert seen[0].shape[1] == f_warm.shape[1]
        autoregressive = seen[1:]
        assert len(autoregressive) == horizon
        for t, captured in enumerate(autoregressive):
            expected = forecast_driver[:, t : t + 1, :]
            assert torch.equal(captured, expected), (
                f"step {t} consumed driver {captured.flatten()[0].item()}, "
                f"expected {expected.flatten()[0].item()}"
            )

    def test_driven_accepts_extra_driver_steps(self):
        """A longer driver tensor is accepted; only the first ``horizon`` used."""
        ens = self._build_driven_ensemble(n_models=2)
        x = _toy_data()
        f_warm = x[:, 40:60]
        warm_driver = torch.randn(1, f_warm.shape[1], 2)
        horizon = 5
        forecast_driver = torch.randn(1, horizon + 4, 2)

        reservoir_layer = _reservoir_layer_of(ens.models[0])
        seen: list[torch.Tensor] = []
        handle = reservoir_layer.register_forward_pre_hook(
            lambda module, args: seen.append(args[1].detach().clone())
        )
        try:
            ens.forecast(
                (f_warm, warm_driver),
                forecast_inputs=(forecast_driver,),
                horizon=horizon,
            )
        finally:
            handle.remove()

        autoregressive = seen[1:]
        assert len(autoregressive) == horizon
        for t, captured in enumerate(autoregressive):
            assert torch.equal(captured, forecast_driver[:, t : t + 1, :])

    def test_driven_rejects_too_few_driver_steps(self):
        ens = self._build_driven_ensemble(n_models=2)
        x = _toy_data()
        f_warm = x[:, 40:60]
        warm_driver = torch.randn(1, f_warm.shape[1], 2)
        with pytest.raises(ValueError, match="forecast_inputs\\[0\\]"):
            ens.forecast(
                (f_warm, warm_driver),
                forecast_inputs=(torch.randn(1, 3, 2),),
                horizon=10,
            )

    def test_driven_missing_drivers_raises(self):
        ens = self._build_driven_ensemble(n_models=2)
        x = _toy_data()
        f_warm = x[:, 40:60]
        warm_driver = torch.randn(1, f_warm.shape[1], 2)
        with pytest.raises(ValueError, match="forecast_inputs must be provided"):
            ens.forecast((f_warm, warm_driver), horizon=10)


# ---------------------------------------------------------------------------
# Single-model (n_models=1) fast path
# ---------------------------------------------------------------------------


class TestSingleModel:
    """``n_models=1`` collapses to a plain single-model forecast.

    With one sub-model, aggregation (mean/median of a singleton stack) is the
    identity, so the coupled ensemble's two-phase forecast must reproduce the
    sole sub-model's own :meth:`resdag.core.ESNModel.forecast` step-for-step.
    These tests pin that equivalence for both the feedback-only and the driven
    paths, and exercise the ``n_models == 1`` short-circuit in
    :meth:`CoupledEnsembleESNModel.fit`.
    """

    def test_n_models_1_constructs(self):
        ens = rd.coupled_ensemble_esn(
            n_models=1, reservoir_size=20, feedback_size=2, output_size=2, seed=0
        )
        assert ens.n_models == 1
        assert len(ens.models) == 1

    def test_n_models_1_forecast_matches_single_submodel(self):
        """The ensemble forecast equals the sole sub-model's own forecast.

        Fit the singleton ensemble, then run :meth:`ESNModel.forecast` directly
        on its only sub-model with the same warmup/horizon. Both run under
        ``no_grad`` from the warmed-and-reset state, so the aggregated ensemble
        output must match the bare sub-model forecast.
        """
        x = _toy_data()
        warmup, train, f_warm = x[:, :10], x[:, 10:40], x[:, 40:60]

        ens = rd.coupled_ensemble_esn(
            n_models=1, reservoir_size=30, feedback_size=2, output_size=2, seed=0
        )
        ens.fit((warmup,), (train,), {"output": train.clone()})

        horizon = 20
        ens_out = ens.forecast(f_warm, horizon=horizon)

        sub_model = next(ens._iter_models())
        sub_out = sub_model.forecast(f_warm, horizon=horizon)
        assert isinstance(sub_out, torch.Tensor)

        assert ens_out.shape == (1, horizon, 2)
        assert ens_out.shape == sub_out.shape
        assert torch.allclose(ens_out, sub_out, atol=1e-5)

    def test_n_models_1_individual_equals_aggregate(self):
        """With one model the aggregated and per-model trajectories coincide."""
        x = _toy_data()
        warmup, train, f_warm = x[:, :10], x[:, 10:40], x[:, 40:60]

        ens = rd.coupled_ensemble_esn(
            n_models=1, reservoir_size=25, feedback_size=2, output_size=2, seed=1
        )
        ens.fit((warmup,), (train,), {"output": train.clone()})

        agg, indiv = ens.forecast(f_warm, horizon=12, return_individuals=True)
        assert len(indiv) == 1
        assert indiv[0].shape == (1, 12, 2)
        # mean of a singleton stack is the identity → aggregate == the one member.
        assert torch.allclose(agg, indiv[0], atol=1e-6)

    def test_n_models_1_driven_forecast_matches_single_submodel(self):
        """The driven ``n_models=1`` forecast matches the sub-model's driven one.

        Same equivalence as the feedback-only case, but with an exogenous driver
        threaded through both the ensemble loop and :meth:`ESNModel.forecast`.
        """
        ens = rd.coupled_ensemble_esn(n_models=1, model_factory=_driven_submodel_factory, seed=0)
        x = _toy_data()
        warmup, train = x[:, :10], x[:, 10:40]
        driver_w = torch.randn(1, warmup.shape[1], 2)
        driver_t = torch.randn(1, train.shape[1], 2)
        ens.fit((warmup, driver_w), (train, driver_t), {"output": train.clone()})

        f_warm = x[:, 40:60]
        warm_driver = torch.randn(1, f_warm.shape[1], 2)
        horizon = 12
        forecast_driver = torch.randn(1, horizon, 2)

        ens_out = ens.forecast(
            (f_warm, warm_driver),
            forecast_inputs=(forecast_driver,),
            horizon=horizon,
        )

        sub_model = next(ens._iter_models())
        sub_out = sub_model.forecast(
            (f_warm, warm_driver),
            forecast_inputs=(forecast_driver,),
            horizon=horizon,
        )
        assert isinstance(sub_out, torch.Tensor)

        assert ens_out.shape == (1, horizon, 2)
        assert torch.allclose(ens_out, sub_out, atol=1e-5)

    def test_n_models_1_fit_trains_the_single_readout(self):
        """The ``n_models == 1`` fit branch actually fits the lone readout.

        Before fitting, the readout weight is the (untrained) default; after a
        single-model fit it must have changed, proving the short-circuit path in
        ``fit`` ran the trainer rather than no-opping.
        """
        x = _toy_data()
        warmup, train, f_warm = x[:, :10], x[:, 10:40], x[:, 40:60]

        ens = rd.coupled_ensemble_esn(
            n_models=1, reservoir_size=20, feedback_size=2, output_size=2, seed=0
        )
        sub_model = next(ens._iter_models())
        readout = next(m for m in sub_model.modules() if isinstance(m, CGReadoutLayer))
        before = readout.weight.detach().clone()

        ens.fit((warmup,), (train,), {"output": train.clone()})

        after = readout.weight.detach()
        assert not torch.allclose(before, after)
        # And the fitted single-model ensemble forecasts a finite trajectory.
        out = ens.forecast(f_warm, horizon=10)
        assert out.shape == (1, 10, 2)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Aggregators
# ---------------------------------------------------------------------------


class TestAggregators:
    def test_custom_module_aggregator(self):
        """A user-supplied nn.Module aggregator is invoked at each step."""
        x = _toy_data()
        warmup, train, f_warm = x[:, :10], x[:, 10:40], x[:, 40:60]

        agg = OutliersFilteredMean(method="z_score", threshold=2.0)
        ens = rd.coupled_ensemble_esn(
            n_models=4,
            reservoir_size=20,
            feedback_size=2,
            output_size=2,
            aggregate=agg,
            seed=0,
        )
        ens.fit((warmup,), (train,), {"output": train.clone()})
        out = ens.forecast(f_warm, horizon=10)
        assert out.shape == (1, 10, 2)

    def test_median_even_n_is_interpolated(self):
        """An even-sized ensemble's ``"median"`` aggregation averages the two
        central members (interpolated median), not the lower one.

        Regression for the bias where ``Tensor.median`` returned the lower of
        the two middle values for even N, distorting the coupled feedback at
        every autoregressive step.
        """
        ens = rd.coupled_ensemble_esn(
            n_models=4,
            reservoir_size=10,
            feedback_size=2,
            output_size=2,
            aggregate="median",
            seed=0,
        )
        # Synthetic stacked tensor (N=4, batch=1, T=1, F=1): members 1, 2, 3, 4.
        # Lower median would give 2.0; the interpolated median is 2.5.
        stacked = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(4, 1, 1, 1)
        result = ens._aggregate_stacked(stacked)
        assert torch.allclose(
            result, torch.tensor(2.5)
        ), f"even-N median returned {result.item()}, expected interpolated 2.5"

    def test_median_odd_n_unchanged(self):
        """Odd-sized ensembles keep the exact central member as the median."""
        ens = rd.coupled_ensemble_esn(
            n_models=3,
            reservoir_size=10,
            feedback_size=2,
            output_size=2,
            aggregate="median",
            seed=0,
        )
        # Members 1, 5, 9 (N=3, batch=1, T=1, F=1): the median is the middle 5.
        stacked = torch.tensor([1.0, 5.0, 9.0]).view(3, 1, 1, 1)
        result = ens._aggregate_stacked(stacked)
        assert torch.allclose(
            result, torch.tensor(5.0)
        ), f"odd-N median returned {result.item()}, expected 5.0"

    def test_outliers_filtered_mean_all_outliers_fallback(self):
        """Regression for the bug fixed in commit f0bd4a7: at a position
        where *every* sample is classed as an outlier, the layer must
        return the plain mean (not zero)."""
        layer = OutliersFilteredMean(method="z_score", threshold=0.0)
        # threshold=0 means *nothing* is within tolerance — every sample
        # at every (batch, timestep) is an outlier.
        samples = torch.tensor(
            [
                [[[1.0, 2.0]]],  # sample 0
                [[[3.0, 4.0]]],  # sample 1
                [[[5.0, 6.0]]],  # sample 2
            ]
        )  # shape (samples=3, batch=1, T=1, F=2)
        result = layer(samples)
        expected = samples.mean(dim=0)
        assert torch.allclose(
            result, expected
        ), f"all-outlier fallback returned {result}, expected plain mean {expected}"


# ---------------------------------------------------------------------------
# State handling + persistence
# ---------------------------------------------------------------------------


class TestStateAndPersistence:
    def test_get_set_reservoir_states_roundtrip(self):
        x = _toy_data()
        warmup, train, f_warm = x[:, :10], x[:, 10:40], x[:, 40:60]

        ens = rd.coupled_ensemble_esn(
            n_models=2, reservoir_size=20, feedback_size=2, output_size=2, seed=0
        )
        ens.fit((warmup,), (train,), {"output": train.clone()})
        ens.warmup(f_warm)
        saved = ens.get_reservoir_states()
        # Mutate then restore
        ens.reset_reservoirs()
        ens.set_reservoir_states(saved)
        restored = ens.get_reservoir_states()
        for s_old, s_new in zip(saved, restored):
            for key in s_old:
                assert torch.equal(s_old[key], s_new[key])

    def test_set_reservoir_states_wrong_length_raises(self):
        ens = rd.coupled_ensemble_esn(n_models=2, reservoir_size=20, feedback_size=2, output_size=2)
        with pytest.raises(ValueError, match="state dict"):
            ens.set_reservoir_states([{}])  # only 1 dict for 2 sub-models

    def test_save_load_roundtrip(self):
        x = _toy_data()
        warmup, train, f_warm = x[:, :10], x[:, 10:40], x[:, 40:60]

        ens = rd.coupled_ensemble_esn(
            n_models=2, reservoir_size=20, feedback_size=2, output_size=2, seed=0
        )
        ens.fit((warmup,), (train,), {"output": train.clone()})
        # Run a warmup to populate states
        ens.warmup(f_warm)
        saved_states = ens.get_reservoir_states()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ensemble.pt"
            ens.save(str(path), include_states=True, epoch=1)

            # Construct a fresh ensemble with the same architecture and load
            ens2 = rd.coupled_ensemble_esn(
                n_models=2, reservoir_size=20, feedback_size=2, output_size=2, seed=999
            )
            ens2.load(str(path), load_states=True)

            # Weights match
            for m1, m2 in zip(ens.models, ens2.models):
                for (_, p1), (_, p2) in zip(m1.named_parameters(), m2.named_parameters()):
                    assert torch.equal(p1, p2)

            # States restored
            for s1, s2 in zip(saved_states, ens2.get_reservoir_states()):
                for key in s1:
                    assert torch.equal(s1[key], s2[key])

    def test_save_full_load_full_roundtrip_no_rebuild(self):
        """save_full / load_full reconstruct the whole ensemble without rebuilding."""
        x = _toy_data()
        warmup, train, f_warm = x[:, :10], x[:, 10:40], x[:, 40:60]

        ens = rd.coupled_ensemble_esn(
            n_models=2, reservoir_size=20, feedback_size=2, output_size=2, seed=0
        )
        ens.fit((warmup,), (train,), {"output": train.clone()})
        ens.warmup(f_warm)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ensemble_full.pt"
            ens.save_full(str(path), epoch=3)

            restored, meta = CoupledEnsembleESNModel.load_full(str(path), return_metadata=True)
            assert isinstance(restored, CoupledEnsembleESNModel)
            assert len(restored.models) == len(ens.models)
            assert meta == {"epoch": 3}

            for m1, m2 in zip(ens.models, restored.models):
                for (_, p1), (_, p2) in zip(m1.named_parameters(), m2.named_parameters()):
                    assert torch.equal(p1, p2)

    def test_load_full_rejects_state_dict_checkpoint(self):
        ens = rd.coupled_ensemble_esn(n_models=2, reservoir_size=15, feedback_size=2, output_size=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ens.pt"
            ens.save(str(path))  # state-dict format, not full
            with pytest.raises(ValueError, match="does not contain a full ensemble"):
                CoupledEnsembleESNModel.load_full(str(path))

    def test_load_rejects_full_checkpoint(self):
        ens = rd.coupled_ensemble_esn(n_models=2, reservoir_size=15, feedback_size=2, output_size=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ens_full.pt"
            ens.save_full(str(path))
            with pytest.raises(ValueError, match="load_full"):
                rd.coupled_ensemble_esn(
                    n_models=2, reservoir_size=15, feedback_size=2, output_size=2
                ).load(str(path))

    def test_load_size_mismatch_raises(self):
        ens_small = rd.coupled_ensemble_esn(
            n_models=2, reservoir_size=15, feedback_size=2, output_size=2
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ens.pt"
            ens_small.save(str(path))
            ens_big = rd.coupled_ensemble_esn(
                n_models=3, reservoir_size=15, feedback_size=2, output_size=2
            )
            with pytest.raises(ValueError, match="sub-model"):
                ens_big.load(str(path))
