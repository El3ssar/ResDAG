"""Tests for non-finite loss handling in the HPO objective.

A non-finite (NaN/inf) loss is a *value*, not an exception, so it slips past the
``catch_exceptions`` / ``penalty_value`` machinery in
:class:`resdag.hpo.runner.TrialRunner` (built by
:func:`resdag.hpo.objective.build_objective`).  Optuna then rejects the value and
records the trial as ``FAIL`` — starving the sampler of the signal that the
region is bad and excluding the trial from the study summary.  Diverged forecasts
are the primary failure mode of the chaotic-system use case this module targets,
so non-finite losses are common, not rare.

These tests pin the fix: a non-finite loss must yield a ``COMPLETE`` trial scored
at ``penalty_value`` (direction-aware), with the original non-finiteness recorded
as the ``nonfinite_loss`` user attribute.
"""

import numpy as np
import pytest
import torch

optuna = pytest.importorskip("optuna")

from resdag.hpo.objective import build_objective  # noqa: E402
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
