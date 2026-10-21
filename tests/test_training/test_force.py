"""FORCE learning contracts: online closed-loop pattern generation.

Pins down :meth:`ESNTrainer.fit_force`, the FORCE trainer (Sussillo & Abbott
2009) that trains a reservoir to *autonomously generate* a target pattern by
running the reservoir closed-loop — driven by its own :class:`RLSReadout`
output — and RLS-updating the readout toward the target at every step.

Coverage:

- Validation: readout must be an ``RLSReadout``, single-readout only,
  output/feedback dimension match, driver arity, warmup presence, target rank,
  seed-shape, and hyperparameter positivity.
- Generation quality: after FORCE the *free-running* network reproduces a clean
  periodic (sum-of-sines) target to a small fraction of its scale on every tested
  seed. A strong open-loop teacher-forced-ridge baseline can be much more
  accurate on this clean periodic target; no universal relative ranking is
  asserted. See :class:`TestGenerationQuality` for the measured scope.
- Stability: closed-loop training produces no NaN/Inf and the readout's
  inverse-correlation matrix ``P`` stays symmetric positive-definite.
- Hyperparameter overrides (``delta`` / ``forgetting``) and reset semantics.
- Driven-model path (autonomous generation with a held driver).

Every test seeds ``torch.manual_seed`` so the reservoir topology, input/feedback
weights, and the (noise-free) target are reproducible.
"""

import pytest
import pytorch_symbolic as ps
import torch

from resdag import RLSReadout
from resdag.core import ESNModel
from resdag.layers import Concatenate, ESNLayer, RidgeReadoutLayer
from resdag.models import classic_esn, headless_esn
from resdag.training import ESNTrainer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sum_of_sines(t0: int, n: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """A clean, reproducible periodic target ``(1, n, 1)`` starting at index ``t0``.

    A two-frequency sum of sines with a fixed period — smooth, noise-free, and
    exactly continuable, so a free-running generator can be scored against the
    genuine continuation ``_sum_of_sines(T, horizon)``.
    """
    t = torch.arange(t0, t0 + n, dtype=dtype)
    y = 0.7 * torch.sin(2 * torch.pi * 4 * t / 200) + 0.3 * torch.sin(
        2 * torch.pi * 9 * t / 200 + 1.0
    )
    return y.reshape(1, n, 1)


def _force_model(
    reservoir_size: int = 600,
    spectral_radius: float = 1.5,
    bias: bool = False,
    seed: int = 0,
) -> tuple[torch.nn.Module, RLSReadout]:
    """Build a feedback-only ``classic_esn`` with an ``RLSReadout`` head (float64).

    Returns the model and its (single) ``RLSReadout``.  A chaotic reservoir
    (``spectral_radius > 1``) is the FORCE regime — the online readout tames the
    chaos into the target pattern.
    """
    model = classic_esn(
        reservoir_size,
        feedback_size=1,
        output_size=1,
        spectral_radius=spectral_radius,
        readout=RLSReadout,
        readout_name="output",
        readout_bias=bias,
        seed=seed,
    ).double()
    readout = next(m for _, m in model.named_modules() if isinstance(m, RLSReadout))
    return model, readout


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    """``fit_force`` rejects malformed setups with clear, specific errors."""

    def test_requires_rls_readout(self) -> None:
        """A non-RLS readout (the ridge default) is rejected with a TypeError."""
        torch.manual_seed(0)
        model = classic_esn(50, 1, 1, seed=0).double()  # default ridge readout
        trainer = ESNTrainer(model)
        target = _sum_of_sines(0, 300)
        with pytest.raises(TypeError, match="requires the readout .* to be an RLSReadout"):
            trainer.fit_force(warmup_inputs=(target[:, :50, :],), targets=target[:, 50:, :])

    def test_multi_readout_rejected(self) -> None:
        """A model with two readouts is rejected (FORCE trains a single generator)."""
        torch.manual_seed(0)
        inp = ps.Input((300, 1))
        res = ESNLayer(60, feedback_size=1, spectral_radius=1.2, seed=0)(inp)
        cat = Concatenate()(inp, res)
        out1 = RLSReadout(61, 1, name="output")(cat)
        out2 = RLSReadout(61, 1, name="aux")(cat)
        model = ESNModel(inp, [out1, out2]).double()
        trainer = ESNTrainer(model)
        target = _sum_of_sines(0, 300)
        with pytest.raises(ValueError, match="trains a single self-generating readout"):
            trainer.fit_force(warmup_inputs=(target[:, :50, :],), targets=target[:, 50:, :])

    def test_output_feedback_dim_mismatch(self) -> None:
        """A readout whose out_features != feedback dim is rejected."""
        torch.manual_seed(0)
        # 3-feature reservoir feedback but the target is 1-D: build a model whose
        # readout emits 3 features, then feed a 1-D target.
        model = classic_esn(
            60,
            feedback_size=3,
            output_size=3,
            spectral_radius=1.2,
            readout=RLSReadout,
            readout_name="output",
            seed=0,
        ).double()
        trainer = ESNTrainer(model)
        target = _sum_of_sines(0, 300)  # (1, 300, 1)
        with pytest.raises(ValueError, match="out_features.*must equal the feedback"):
            trainer.fit_force(warmup_inputs=(target[:, :50, :],), targets=target[:, 50:, :])

    def test_empty_warmup_inputs_rejected(self) -> None:
        """An empty warmup_inputs tuple raises (feedback tensor is required)."""
        model, _ = _force_model(reservoir_size=60, spectral_radius=1.2)
        trainer = ESNTrainer(model)
        target = _sum_of_sines(0, 300)
        with pytest.raises(ValueError, match="at least one warmup input"):
            trainer.fit_force(warmup_inputs=(), targets=target)

    def test_non_3d_targets_rejected(self) -> None:
        """A 2-D target raises (targets must be (batch, T, feedback_dim))."""
        model, _ = _force_model(reservoir_size=60, spectral_radius=1.2)
        trainer = ESNTrainer(model)
        target = _sum_of_sines(0, 300)
        with pytest.raises(ValueError, match="targets must be 3-D"):
            trainer.fit_force(warmup_inputs=(target[:, :50, :],), targets=target.squeeze(-1))

    def test_driver_arity_mismatch_rejected(self) -> None:
        """Passing a driver to a feedback-only model raises."""
        model, _ = _force_model(reservoir_size=60, spectral_radius=1.2)
        trainer = ESNTrainer(model)
        target = _sum_of_sines(0, 300)
        driver = torch.randn(1, 50, 2, dtype=torch.float64)
        with pytest.raises(ValueError, match="driver input"):
            trainer.fit_force(warmup_inputs=(target[:, :50, :], driver), targets=target[:, 50:, :])

    def test_bad_initial_feedback_shape_rejected(self) -> None:
        """A wrongly-shaped initial_feedback seed raises."""
        model, _ = _force_model(reservoir_size=60, spectral_radius=1.2)
        trainer = ESNTrainer(model)
        target = _sum_of_sines(0, 300)
        with pytest.raises(ValueError, match="initial_feedback must have shape"):
            trainer.fit_force(
                warmup_inputs=(target[:, :50, :],),
                targets=target[:, 50:, :],
                initial_feedback=torch.zeros(1, 2, 1, dtype=torch.float64),
            )

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_bad_forgetting_rejected(self, bad: float) -> None:
        """A forgetting override outside (0, 1] raises."""
        model, _ = _force_model(reservoir_size=60, spectral_radius=1.2)
        trainer = ESNTrainer(model)
        target = _sum_of_sines(0, 300)
        with pytest.raises(ValueError, match="forgetting"):
            trainer.fit_force(
                warmup_inputs=(target[:, :50, :],), targets=target[:, 50:, :], forgetting=bad
            )

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_bad_delta_rejected(self, bad: float) -> None:
        """A non-positive delta override raises."""
        model, _ = _force_model(reservoir_size=60, spectral_radius=1.2)
        trainer = ESNTrainer(model)
        target = _sum_of_sines(0, 300)
        with pytest.raises(ValueError, match="delta must be positive"):
            trainer.fit_force(
                warmup_inputs=(target[:, :50, :],), targets=target[:, 50:, :], delta=bad
            )


# ---------------------------------------------------------------------------
# Generation quality vs. baseline (the headline capability)
# ---------------------------------------------------------------------------


def _force_and_ridge_generation(
    seed: int,
    *,
    reservoir_size: int = 600,
    spectral_radius: float = 1.5,
    total_steps: int = 5000,
    warmup: int = 200,
    horizon: int = 1000,
    delta: float = 1.0,
) -> tuple[float, float, float, float, float, bool]:
    """Run FORCE and a teacher-forced-ridge baseline on one seed; return metrics.

    Both generators are scored under the *same* protocol — re-warm on the real
    series via :meth:`ESNModel.forecast`, then free-run ``horizon`` steps — so the
    comparison is apples-to-apples. Returns ``(train_tail, force_rmse,
    force_rmse_100, base_rmse, target_std, force_finite)``.
    """
    target = _sum_of_sines(0, total_steps)
    continuation = _sum_of_sines(total_steps, horizon)
    target_std = float(continuation.std().item())

    # --- FORCE (closed-loop online RLS) ---
    model, _ = _force_model(reservoir_size, spectral_radius, seed=seed)
    trainer = ESNTrainer(model)
    train_out = trainer.fit_force(
        warmup_inputs=(target[:, :warmup, :],),
        targets=target[:, warmup:, :],
        delta=delta,
        forgetting=1.0,
    )
    tail = ((train_out[:, -500:] - target[:, warmup:, :][:, -500:]) ** 2).mean().sqrt()
    force_preds = model.forecast(target, horizon=horizon)
    force_rmse = ((force_preds - continuation) ** 2).mean().sqrt()
    force_rmse_100 = ((force_preds[:, :100] - continuation[:, :100]) ** 2).mean().sqrt()

    # --- Baseline: open-loop teacher-forced ridge, then free-run ---
    torch.manual_seed(seed)
    baseline = classic_esn(
        reservoir_size,
        feedback_size=1,
        output_size=1,
        spectral_radius=spectral_radius,
        readout=RidgeReadoutLayer,
        readout_alpha=1e-6,
        readout_name="output",
        readout_bias=False,
        seed=seed,
    ).double()
    base_trainer = ESNTrainer(baseline)
    base_trainer.fit(
        warmup_inputs=(target[:, :warmup, :],),
        train_inputs=(target[:, warmup : total_steps - 1, :],),
        targets={"output": target[:, warmup + 1 : total_steps, :]},
    )
    base_preds = baseline.forecast(target, horizon=horizon)
    base_rmse = ((base_preds - continuation) ** 2).mean().sqrt()

    return (
        float(tail.item()),
        float(force_rmse.item()),
        float(force_rmse_100.item()),
        float(base_rmse.item()),
        target_std,
        bool(torch.isfinite(force_preds).all().item()),
    )


class TestGenerationQuality:
    """FORCE reproduces the continuation within fixed absolute quality bounds.

    Correct next-sample timing reveals that ridge is nearly exact for some
    seeds on this clean periodic target, while FORCE has finite approximation
    error. The old forecast skipped one sample; the common phase error masked
    that difference and made an unsupported relative-ratio assertion pass.
    Absolute FORCE bounds below are unchanged. Chaotic-reservoir stability is
    tested independently in TestClosedLoopStability.
    """

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_force_robustly_generates_continuation(self, seed: int) -> None:
        """Training converges and short/long free runs stay near the target."""
        tail, force_rmse, force_rmse_100, base_rmse, target_std, finite = (
            _force_and_ridge_generation(seed)
        )
        # 1. Converges: the closed loop is on-target by the end of training.
        assert tail < 0.1
        # 2. Generates: small relative error at short and long horizon (worst
        #    measured over seeds 0-3: rmse_100/std ~= 0.18, rmse/std ~= 0.23).
        assert finite
        assert force_rmse_100 < 0.25 * target_std
        assert force_rmse < 0.35 * target_std
        # The baseline is measured on the identical next-sample timeline, but
        # its error can be orders of magnitude smaller on this periodic signal.
        assert torch.isfinite(torch.tensor(base_rmse))

    def test_ridge_baseline_predicts_the_immediate_unseen_sample(self) -> None:
        """An exact next-step ramp distinguishes aligned scoring from a shift."""
        model = classic_esn(
            20,
            feedback_size=1,
            output_size=1,
            readout=RidgeReadoutLayer,
            readout_alpha=1e-8,
            readout_bias=True,
            seed=1,
        ).double()
        # Keep only the direct feedback and intercept features, so this is an
        # exactly representable ridge baseline rather than an extrapolation test.
        with torch.no_grad():
            for layer in model.modules():
                if isinstance(layer, ESNLayer):
                    layer.cell.weight_feedback.zero_()
                    layer.cell.weight_hh.zero_()
                    if layer.cell.bias_h is not None:
                        layer.cell.bias_h.zero_()
        data = torch.arange(36, dtype=torch.float64).view(1, 36, 1) / 100
        ESNTrainer(model).fit((data[:, :5],), (data[:, 5:30],), {"output": data[:, 6:31]})
        pred = model.forecast(data[:, :31], horizon=5)
        assert torch.allclose(pred, data[:, 31:], atol=1e-5, rtol=0)
        assert not torch.allclose(pred[:, :1], data[:, 32:33], atol=1e-3, rtol=0)

    def test_returns_closed_loop_training_outputs(self) -> None:
        """fit_force returns the per-step closed-loop outputs of the right shape."""
        model, _ = _force_model(reservoir_size=300, spectral_radius=1.5)
        trainer = ESNTrainer(model)
        target = _sum_of_sines(0, 2000)
        out = trainer.fit_force(
            warmup_inputs=(target[:, :200, :],), targets=target[:, 200:, :], delta=1.0
        )
        assert out.shape == (1, 1800, 1)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------


class TestStability:
    """Closed-loop FORCE training stays numerically stable."""

    def test_no_nan_and_p_stays_pd(self) -> None:
        """No NaN/Inf during training and P remains symmetric positive-definite."""
        model, readout = _force_model(reservoir_size=400, spectral_radius=1.5)
        trainer = ESNTrainer(model)
        target = _sum_of_sines(0, 3000)
        out = trainer.fit_force(
            warmup_inputs=(target[:, :200, :],), targets=target[:, 200:, :], delta=1.0
        )
        assert torch.isfinite(out).all()
        assert torch.isfinite(readout.P).all()
        # Exactly symmetric (re-symmetrised each RLS step).
        assert (readout.P - readout.P.T).abs().max().item() == 0.0
        # Strictly positive-definite.
        evals = torch.linalg.eigvalsh(readout.P)
        assert float(evals.min().item()) > 0.0

    def test_readout_is_fitted_after_force(self) -> None:
        """The readout reports fitted and has non-trivial weights after FORCE."""
        model, readout = _force_model(reservoir_size=200, spectral_radius=1.5)
        trainer = ESNTrainer(model)
        target = _sum_of_sines(0, 1500)
        trainer.fit_force(
            warmup_inputs=(target[:, :150, :],), targets=target[:, 150:, :], delta=1.0
        )
        assert readout.is_fitted
        assert readout.n_seen == 1350  # one RLS step per closed-loop timestep
        assert float(readout.weight.detach().abs().max().item()) > 0.0


# ---------------------------------------------------------------------------
# Hyperparameter overrides and semantics
# ---------------------------------------------------------------------------


class TestOverridesAndSemantics:
    """delta / forgetting overrides and reset semantics behave as documented."""

    def test_delta_override_applied(self) -> None:
        """A delta override is written onto the readout before the recursion."""
        model, readout = _force_model(reservoir_size=100, spectral_radius=1.5)
        trainer = ESNTrainer(model)
        target = _sum_of_sines(0, 800)
        trainer.fit_force(
            warmup_inputs=(target[:, :100, :],), targets=target[:, 100:, :], delta=2.5
        )
        assert readout.delta == 2.5

    def test_forgetting_override_applied(self) -> None:
        """A forgetting override is written onto the readout before the recursion."""
        model, readout = _force_model(reservoir_size=100, spectral_radius=1.5)
        trainer = ESNTrainer(model)
        target = _sum_of_sines(0, 800)
        trainer.fit_force(
            warmup_inputs=(target[:, :100, :],),
            targets=target[:, 100:, :],
            forgetting=0.999,
            delta=1.0,
        )
        assert readout.forgetting == 0.999

    def test_rerun_trains_from_scratch(self) -> None:
        """A second fit_force resets the recursion (n_seen reflects only the last run)."""
        model, readout = _force_model(reservoir_size=100, spectral_radius=1.5)
        trainer = ESNTrainer(model)
        target = _sum_of_sines(0, 800)
        trainer.fit_force(
            warmup_inputs=(target[:, :100, :],), targets=target[:, 100:, :], delta=1.0
        )
        first = readout.n_seen
        trainer.fit_force(
            warmup_inputs=(target[:, :100, :],), targets=target[:, 100:400, :], delta=1.0
        )
        assert readout.n_seen == 300  # only the second run's steps
        assert first == 700

    def test_zero_length_warmup_runs(self) -> None:
        """A length-0 warmup starts FORCE from the reset reservoir state."""
        model, readout = _force_model(reservoir_size=150, spectral_radius=1.5)
        trainer = ESNTrainer(model)
        target = _sum_of_sines(0, 1000)
        empty_warmup = target[:, :0, :]  # (1, 0, 1)
        out = trainer.fit_force(warmup_inputs=(empty_warmup,), targets=target, delta=1.0)
        assert out.shape == (1, 1000, 1)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Driven-model path
# ---------------------------------------------------------------------------


class TestDrivenModel:
    """FORCE on a model with an exogenous driver (held during generation)."""

    def test_driven_force_runs_and_is_stable(self) -> None:
        """A driven model FORCE-trains without error, drivers held at last warmup."""
        torch.manual_seed(0)
        model = classic_esn(
            300,
            feedback_size=1,
            output_size=1,
            input_size=2,
            spectral_radius=1.5,
            readout=RLSReadout,
            readout_name="output",
            readout_bias=False,
            seed=0,
        ).double()
        readout = next(m for _, m in model.named_modules() if isinstance(m, RLSReadout))
        trainer = ESNTrainer(model)
        target = _sum_of_sines(0, 2000)
        driver = torch.randn(1, 200, 2, dtype=torch.float64) * 0.1
        out = trainer.fit_force(
            warmup_inputs=(target[:, :200, :], driver),
            targets=target[:, 200:, :],
            delta=1.0,
        )
        assert out.shape == (1, 1800, 1)
        assert torch.isfinite(out).all()
        assert torch.isfinite(readout.P).all()
        evals = torch.linalg.eigvalsh(readout.P)
        assert float(evals.min().item()) > 0.0

    def test_driven_empty_warmup_rejected(self) -> None:
        """A driven model with a length-0 warmup raises (no driver value to hold)."""
        torch.manual_seed(0)
        model = classic_esn(
            100,
            feedback_size=1,
            output_size=1,
            input_size=2,
            spectral_radius=1.5,
            readout=RLSReadout,
            readout_name="output",
            readout_bias=False,
            seed=0,
        ).double()
        trainer = ESNTrainer(model)
        target = _sum_of_sines(0, 500)
        empty_driver = torch.zeros(1, 0, 2, dtype=torch.float64)
        with pytest.raises(ValueError, match="driven model needs a non-empty warmup"):
            trainer.fit_force(warmup_inputs=(target[:, :0, :], empty_driver), targets=target)


# ---------------------------------------------------------------------------
# Discovery guard
# ---------------------------------------------------------------------------


def test_headless_model_has_no_readout() -> None:
    """A headless model (no readout) is rejected by fit_force's single-readout guard."""
    torch.manual_seed(0)
    model = headless_esn(60, feedback_size=1, spectral_radius=1.2, seed=0).double()
    trainer = ESNTrainer(model)
    target = _sum_of_sines(0, 300)
    with pytest.raises(ValueError, match="trains a single self-generating readout"):
        trainer.fit_force(warmup_inputs=(target[:, :50, :],), targets=target[:, 50:, :])
