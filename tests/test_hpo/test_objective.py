"""Tests for the HPO objective built by :func:`build_objective`.

Two behaviours are pinned here:

* **Non-finite loss handling.**  A non-finite (NaN/inf) loss is a *value*, not an
  exception, so it slips past the ``catch_exceptions`` / ``penalty_value``
  machinery in :class:`resdag.hpo.runner.TrialRunner` (built by
  :func:`resdag.hpo.objective.build_objective`).  Optuna then rejects the value
  and records the trial as ``FAIL`` — starving the sampler of the signal that the
  region is bad and excluding the trial from the study summary.  Diverged
  forecasts are the primary failure mode of the chaotic-system use case this
  module targets, so non-finite losses are common, not rare.  These tests pin the
  fix: a non-finite loss must yield a ``COMPLETE`` trial scored at
  ``penalty_value`` (direction-aware), with the original non-finiteness recorded
  as the ``nonfinite_loss`` user attribute.

* **Multi-output forecasts.**  :meth:`ESNModel.forecast` returns a *tuple* of
  tensors for multi-output models (the first output being the autoregression
  feedback).  The objective must normalize that tuple to the first output before
  indexing it (``preds.shape``), otherwise every multi-output trial raises
  ``AttributeError: 'tuple' object has no attribute 'shape'`` and — under the
  default ``catch_exceptions=True`` — silently completes at ``penalty_value``,
  turning the whole study into garbage.  These tests pin the guard: a 2-output
  model produces a ``COMPLETE`` trial with a finite, non-penalty loss scored on
  the first output.
"""

import numpy as np
import pytest
import torch

optuna = pytest.importorskip("optuna")

from resdag.core import ESNModel, reservoir_input  # noqa: E402
from resdag.hpo.objective import build_objective  # noqa: E402
from resdag.layers import CGReadoutLayer, Concatenate, ESNLayer, Power  # noqa: E402
from resdag.models import ott_esn  # noqa: E402


# ── Module-level callbacks (top-level so the runner stays picklable) ──────────
def model_creator(reservoir_size: int = 30, spectral_radius: float = 0.9):
    """Create a small Ott ESN for fast trials."""
    return ott_esn(
        reservoir_size=reservoir_size,
        feedback_size=3,
        output_size=3,
        spectral_radius=spectral_radius,
    )


def search_space(trial):
    """Fixed search space so trials are cheap and deterministic."""
    return {"reservoir_size": 30, "spectral_radius": 0.9}


def data_loader(trial):
    """Deterministic synthetic data so trials are reproducible."""
    torch.manual_seed(0)
    data = torch.randn(1, 200, 3)
    return {
        "warmup": data[:, :20, :],
        "train": data[:, 20:120, :],
        "target": data[:, 21:121, :],
        "f_warmup": data[:, 120:140, :],
        "val": data[:, 140:180, :],  # 40-step horizon
    }


def nan_loss(y_true, y_pred, /, **kwargs):
    """Loss that always returns NaN (simulates a diverged forecast)."""
    return float("nan")


def inf_loss(y_true, y_pred, /, **kwargs):
    """Loss that always returns +inf (simulates a diverged forecast)."""
    return float("inf")


def _run_single_trial(loss_fn, *, direction: str = "minimize", penalty_value: float = 1e10):
    """Run a one-trial study with ``loss_fn`` and return the (study, trial)."""
    runner = build_objective(
        model_creator=model_creator,
        search_space=search_space,
        data_loader=data_loader,
        loss_fn=loss_fn,
        seed=1,
        penalty_value=penalty_value,
    )
    study = optuna.create_study(direction=direction)
    # ``catch=()`` ensures the trial is NOT rescued by Optuna's own exception
    # handling — any FAIL would be a genuine non-finite rejection.
    study.optimize(runner, n_trials=1, catch=())
    return study, study.trials[0]


class TestNonFiniteLossPenalty:
    """A non-finite loss completes with ``penalty_value`` instead of failing."""

    @pytest.mark.parametrize("loss_fn", [nan_loss, inf_loss], ids=["nan", "inf"])
    def test_nonfinite_loss_completes_with_penalty(self, loss_fn):
        """NaN/inf losses yield a COMPLETE trial valued at ``penalty_value``."""
        penalty = 1e10
        study, trial = _run_single_trial(loss_fn, penalty_value=penalty)

        assert trial.state == optuna.trial.TrialState.COMPLETE
        assert trial.value == penalty
        assert study.best_value == penalty

    @pytest.mark.parametrize("loss_fn", [nan_loss, inf_loss], ids=["nan", "inf"])
    def test_nonfinite_loss_records_user_attr(self, loss_fn):
        """The original non-finiteness is recorded as ``nonfinite_loss``."""
        _, trial = _run_single_trial(loss_fn)
        assert trial.user_attrs.get("nonfinite_loss") is True

    def test_maximize_direction_penalizes_negatively(self):
        """For a maximize study the penalty is sign-flipped to ``-penalty``."""
        penalty = 1e10
        study, trial = _run_single_trial(nan_loss, direction="maximize", penalty_value=penalty)

        assert trial.state == optuna.trial.TrialState.COMPLETE
        assert trial.value == -penalty
        assert study.best_value == -penalty

    def test_finite_loss_is_unchanged(self):
        """A finite loss passes through untouched and sets no penalty attr."""

        def finite_loss(y_true, y_pred, /, **kwargs):
            if isinstance(y_true, torch.Tensor):
                return float(torch.mean(torch.abs(y_true - y_pred)))
            return float(np.mean(np.abs(y_true - y_pred)))

        _, trial = _run_single_trial(finite_loss)

        assert trial.state == optuna.trial.TrialState.COMPLETE
        assert trial.value is not None
        assert np.isfinite(trial.value)
        assert "nonfinite_loss" not in trial.user_attrs


# ── Multi-output model callbacks (top-level so the runner stays picklable) ─────
def multi_output_model_creator(reservoir_size: int = 30, spectral_radius: float = 0.9):
    """Create a 2-output ESN whose ``forecast`` returns a tuple of tensors.

    The first output is the trained feedback readout (``name="output"``, the
    autoregression target the loss scores); the second is a parameterless
    ``Power`` branch derived from it, so the model needs only the single
    ``"output"`` target the runner supplies while still being genuinely
    multi-output (``forecast`` returns a 2-tuple).
    """
    inp = reservoir_input(3)
    res = ESNLayer(reservoir_size, feedback_size=3, spectral_radius=spectral_radius)(inp)
    cat = Concatenate()(inp, res)
    primary = CGReadoutLayer(3 + reservoir_size, 3, name="output")(cat)
    secondary = Power(2.0)(primary)
    return ESNModel(inp, [primary, secondary])


def mae_loss(y_true, y_pred, /, **kwargs):
    """Finite mean-absolute-error loss supporting tensor or NumPy inputs."""
    if isinstance(y_true, torch.Tensor):
        return float(torch.mean(torch.abs(y_true - y_pred)))
    return float(np.mean(np.abs(y_true - y_pred)))


class TestMultiOutputForecast:
    """Multi-output models complete with a finite loss scored on output 0."""

    def test_multi_output_forecast_is_a_tuple(self):
        """Sanity: the 2-output model's ``forecast`` returns a tuple of tensors.

        This guards the test's own premise — if the model ever became
        single-output, the regression below would pass vacuously.
        """
        model = multi_output_model_creator()
        data = data_loader(None)
        from resdag.training import ESNTrainer

        ESNTrainer(model).fit(
            warmup_inputs=(data["warmup"],),
            train_inputs=(data["train"],),
            targets={"output": data["target"]},
        )
        preds = model.forecast((data["f_warmup"],), horizon=data["val"].shape[1])

        assert isinstance(preds, tuple)
        assert len(preds) == 2
        assert all(isinstance(p, torch.Tensor) for p in preds)
        # First output is the feedback dimension (matched against ``val``).
        assert preds[0].shape[-1] == data["val"].shape[-1]

    def test_multi_output_trial_completes_with_finite_loss(self):
        """A 2-output model yields a COMPLETE trial with a finite, non-penalty loss.

        Without the tuple guard, ``forecast`` returns a tuple and the objective's
        ``preds.shape`` raises ``AttributeError``; under the default
        ``catch_exceptions=True`` the trial would silently complete at
        ``penalty_value``.  This asserts the loss is finite and *not* the penalty.
        """
        penalty = 1e10
        runner = build_objective(
            model_creator=multi_output_model_creator,
            search_space=search_space,
            data_loader=data_loader,
            loss_fn=mae_loss,
            seed=1,
            penalty_value=penalty,
        )
        study = optuna.create_study(direction="minimize")
        # ``catch=()`` so any genuine AttributeError surfaces as a FAIL rather
        # than being rescued by Optuna's own exception handling.
        study.optimize(runner, n_trials=1, catch=())
        trial = study.trials[0]

        assert trial.state == optuna.trial.TrialState.COMPLETE
        assert trial.value is not None
        assert np.isfinite(trial.value)
        assert trial.value != penalty
        # No exception was caught and stashed by the runner.
        assert "error" not in trial.user_attrs

    def test_multi_output_does_not_raise_with_catch_disabled(self):
        """With ``catch_exceptions=False`` the tuple return must not raise.

        This is the strongest form of the guard check: it removes *both* safety
        nets (Optuna's ``catch`` and the runner's ``catch_exceptions``) so a
        tuple-indexing crash would propagate out of ``optimize`` as a hard error.
        """
        runner = build_objective(
            model_creator=multi_output_model_creator,
            search_space=search_space,
            data_loader=data_loader,
            loss_fn=mae_loss,
            seed=1,
            catch_exceptions=False,
        )
        study = optuna.create_study(direction="minimize")
        study.optimize(runner, n_trials=1, catch=())  # must not raise

        assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
