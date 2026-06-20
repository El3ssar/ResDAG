"""Tests for the picklable :class:`resdag.hpo.runner.TrialRunner`.

These tests exercise the four acceptance criteria of the HPO-unblocker ticket:

1. ``TrialRunner`` is picklable and parity-matches the legacy closure when no
   pruner is set.
2. ``trial.report`` fires at >= 2 horizon checkpoints and a ``MedianPruner``
   prunes >= 1 divergent trial.
3. CUDA tensors are scored on-device until the final scalar (covered by the
   torch-scoring path; verified on CPU and, when available, on CUDA).
4. ``trial_callbacks`` never fail the trial.
"""

import pickle

import numpy as np
import pytest
import torch

optuna = pytest.importorskip("optuna")

from resdag.hpo.objective import build_objective  # noqa: E402
from resdag.hpo.runner import TrialRunner, _checkpoint_horizons  # noqa: E402
from resdag.models import ott_esn  # noqa: E402


# ── Module-level callbacks (must be top-level so the runner stays picklable) ──
def model_creator(reservoir_size: int = 40, spectral_radius: float = 0.9):
    """Create a small Ott ESN for fast trials."""
    return ott_esn(
        reservoir_size=reservoir_size,
        feedback_size=3,
        output_size=3,
        spectral_radius=spectral_radius,
    )


def search_space(trial):
    """Tiny search space over reservoir size and spectral radius."""
    return {
        "reservoir_size": trial.suggest_int("reservoir_size", 20, 40, step=10),
        "spectral_radius": trial.suggest_float("spectral_radius", 0.5, 1.1),
    }


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


def mae_loss(y_true, y_pred, /, **kwargs):
    """MAE loss that works on both NumPy arrays and torch tensors.

    Used to confirm on-device (torch) scoring: with torch tensors the
    arithmetic and reductions stay on the input device.
    """
    if isinstance(y_true, torch.Tensor):
        return float(torch.mean(torch.abs(y_true - y_pred)))
    return float(np.mean(np.abs(y_true - y_pred)))


def numpy_only_loss(y_true, y_pred, /, **kwargs):
    """Loss that only accepts NumPy arrays (rejects torch tensors)."""
    if isinstance(y_true, torch.Tensor):
        raise TypeError("this loss only accepts numpy arrays")
    return float(np.mean(np.abs(y_true - y_pred)))


# ── _checkpoint_horizons ─────────────────────────────────────────────────────
class TestCheckpointHorizons:
    """Unit tests for the checkpoint schedule helper."""

    def test_multiple_checkpoints_end_at_horizon(self):
        cps = _checkpoint_horizons(40, 5)
        assert cps[-1] == 40
        assert len(cps) >= 2
        assert all(b > a for a, b in zip(cps, cps[1:]))  # strictly increasing
        assert all(1 <= c <= 40 for c in cps)

    def test_single_checkpoint_when_disabled(self):
        assert _checkpoint_horizons(40, 1) == [40]
        assert _checkpoint_horizons(40, 0) == [40]

    def test_dedup_when_horizon_small(self):
        cps = _checkpoint_horizons(3, 5)
        assert cps[-1] == 3
        assert cps == sorted(set(cps))


# ── Acceptance criterion 1: picklable + parity ───────────────────────────────
class TestPicklableAndParity:
    """TrialRunner is picklable and matches the legacy single-shot behaviour."""

    def test_runner_is_picklable(self):
        runner = build_objective(
            model_creator=model_creator,
            search_space=search_space,
            data_loader=data_loader,
            loss_fn=mae_loss,
            seed=123,
        )
        assert isinstance(runner, TrialRunner)
        restored = pickle.loads(pickle.dumps(runner))
        assert isinstance(restored, TrialRunner)
        # Attributes survive the round-trip.
        assert restored.seed == 123
        assert restored.n_checkpoints == runner.n_checkpoints

    def test_pickled_runner_optimizes(self):
        """A round-tripped runner can still run a study end-to-end."""
        runner = build_objective(
            model_creator=model_creator,
            search_space=search_space,
            data_loader=data_loader,
            loss_fn=mae_loss,
            seed=7,
        )
        restored = pickle.loads(pickle.dumps(runner))
        study = optuna.create_study(direction="minimize")
        study.optimize(restored, n_trials=2)
        assert len(study.trials) == 2
        assert study.best_value is not None

    def test_parity_no_pruner(self):
        """With no pruner the checkpointed runner equals a single-shot run.

        The legacy objective forecasted once and returned a single full-horizon
        loss.  With ``NopPruner`` the checkpointed runner must return the same
        value as a runner configured with a single checkpoint.
        """
        common = dict(
            model_creator=model_creator,
            search_space=search_space,
            data_loader=data_loader,
            loss_fn=mae_loss,
            seed=42,
        )
        checkpointed = build_objective(n_checkpoints=5, **common)
        single_shot = build_objective(n_checkpoints=1, **common)

        study_a = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.NopPruner(),
            sampler=optuna.samplers.RandomSampler(seed=1),
        )
        study_b = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.NopPruner(),
            sampler=optuna.samplers.RandomSampler(seed=1),
        )
        study_a.optimize(checkpointed, n_trials=3)
        study_b.optimize(single_shot, n_trials=3)

        vals_a = [t.value for t in study_a.trials]
        vals_b = [t.value for t in study_b.trials]
        assert vals_a == pytest.approx(vals_b, rel=1e-9, abs=1e-9)

    def test_parity_matches_legacy_numpy_computation(self):
        """The returned loss matches the legacy ``loss_fn(val_np, preds_np)``.

        Uses the real ``efh`` loss (nrmse-based, so argument order matters) to
        lock the ``loss_fn(y_true=val, y_pred=preds)`` convention and confirm
        the single-forecast checkpointed value equals a manual computation.
        """
        import numpy as np_

        from resdag.hpo import get_loss
        from resdag.training import ESNTrainer

        efh = get_loss("efh")

        def fixed_space(trial):
            return {"reservoir_size": 30, "spectral_radius": 0.9}

        # Reference: mirror the runner's exact RNG sequence
        # (seed → search_space → data_loader → model_creator → train → forecast).
        torch.manual_seed(99)
        np_.random.seed(99)
        params = fixed_space(None)
        data = data_loader(None)
        model = model_creator(**params)
        trainer = ESNTrainer(model)
        trainer.fit(
            warmup_inputs=(data["warmup"],),
            train_inputs=(data["train"],),
            targets={"output": data["target"]},
        )
        preds = model.forecast((data["f_warmup"],), horizon=data["val"].shape[1])
        if isinstance(preds, tuple):
            preds = preds[0]
        steps = min(preds.shape[1], data["val"].shape[1])
        expected = float(
            efh(
                data["val"][:, :steps, :].detach().cpu().numpy(),
                preds[:, :steps, :].detach().cpu().numpy(),
            )
        )

        runner = build_objective(
            model_creator=model_creator,
            search_space=fixed_space,
            data_loader=data_loader,
            loss_fn=efh,
            seed=99,
            n_checkpoints=5,
        )
        study = optuna.create_study(direction="minimize", pruner=optuna.pruners.NopPruner())
        study.optimize(runner, n_trials=1)
        assert study.trials[0].value == pytest.approx(expected, rel=1e-6, abs=1e-6)
        # Final reported intermediate value equals the returned objective.
        intermediate = study.trials[0].intermediate_values
        assert intermediate[max(intermediate)] == pytest.approx(expected, rel=1e-6, abs=1e-6)


# ── Acceptance criterion 2: reporting + pruning ──────────────────────────────
class TestReportingAndPruning:
    """trial.report fires at >= 2 checkpoints; pruner prunes divergent trials."""

    def test_reports_at_multiple_checkpoints(self):
        runner = build_objective(
            model_creator=model_creator,
            search_space=search_space,
            data_loader=data_loader,
            loss_fn=mae_loss,
            seed=5,
            n_checkpoints=5,
        )
        study = optuna.create_study(direction="minimize", pruner=optuna.pruners.NopPruner())
        study.optimize(runner, n_trials=1)
        # intermediate_values is keyed by checkpoint step.
        intermediate = study.trials[0].intermediate_values
        assert len(intermediate) >= 2

    def test_median_pruner_prunes_divergent_trial(self):
        """A diverging trial is pruned by MedianPruner at an early checkpoint.

        We construct a deterministic synthetic loss: the first trial is good
        (small reported losses), all later trials diverge badly.  Once the first
        trial establishes a baseline, the MedianPruner prunes the diverging
        trials at an early horizon checkpoint.  The loss depends only on the
        slice length, so the per-checkpoint report sequence is well-defined.
        """
        trial_counter = {"completed": 0}

        def diverging_loss(y_true, y_pred, /, **kwargs):
            # Loss grows with slice length so later checkpoints report worse.
            steps = y_true.shape[1]
            if trial_counter["completed"] == 0:
                return 0.01 * steps  # first trial: good baseline
            return 100.0 * steps  # subsequent trials: diverge

        def _count_finished(study, trial):
            trial_counter["completed"] += 1

        runner = build_objective(
            model_creator=model_creator,
            search_space=search_space,
            data_loader=data_loader,
            loss_fn=diverging_loss,
            seed=11,
            n_checkpoints=5,
        )

        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=0),
            sampler=optuna.samplers.RandomSampler(seed=3),
        )
        study.optimize(runner, n_trials=3, callbacks=[_count_finished])

        pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        assert len(pruned) >= 1
        # And reporting happened at >= 2 checkpoints for the baseline trial.
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        assert len(completed[0].intermediate_values) >= 2


# ── Acceptance criterion 3: on-device (torch) scoring ────────────────────────
class TestTorchScoring:
    """CUDA tensors stay on-device until the final scalar; NumPy fallback works."""

    def test_torch_scoring_keeps_device(self):
        """The loss receives tensors on the same device as the forecast."""
        seen = {"is_tensor": False, "device_type": None}

        def device_probe_loss(y_true, y_pred, /, **kwargs):
            if isinstance(y_true, torch.Tensor):
                seen["is_tensor"] = True
                seen["device_type"] = y_true.device.type
                return float(torch.mean(torch.abs(y_true - y_pred)))
            return float(np.mean(np.abs(y_true - y_pred)))

        runner = build_objective(
            model_creator=model_creator,
            search_space=search_space,
            data_loader=data_loader,
            loss_fn=device_probe_loss,
            seed=9,
            torch_scoring=True,
        )
        study = optuna.create_study(direction="minimize")
        study.optimize(runner, n_trials=1)
        assert seen["is_tensor"] is True
        assert seen["device_type"] == "cpu"  # no CUDA in CI → CPU device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_torch_scoring_on_cuda(self):
        """On CUDA, the loss is invoked with CUDA tensors (scored on-device)."""
        seen = {"device_type": None}

        def cuda_probe_loss(y_true, y_pred, /, **kwargs):
            if isinstance(y_true, torch.Tensor):
                seen["device_type"] = y_true.device.type
                return float(torch.mean(torch.abs(y_true - y_pred)))
            return float(np.mean(np.abs(y_true - y_pred)))

        runner = build_objective(
            model_creator=model_creator,
            search_space=search_space,
            data_loader=data_loader,
            loss_fn=cuda_probe_loss,
            seed=9,
            torch_scoring=True,
            device=torch.device("cuda"),
        )
        study = optuna.create_study(direction="minimize")
        study.optimize(runner, n_trials=1)
        assert seen["device_type"] == "cuda"

    def test_numpy_fallback_for_numpy_only_loss(self):
        """A loss that rejects tensors transparently falls back to NumPy."""
        runner = build_objective(
            model_creator=model_creator,
            search_space=search_space,
            data_loader=data_loader,
            loss_fn=numpy_only_loss,
            seed=9,
            torch_scoring=True,
        )
        study = optuna.create_study(direction="minimize")
        study.optimize(runner, n_trials=1)
        # Did not crash → fallback worked; a finite loss was produced.
        assert study.trials[0].value is not None
        assert np.isfinite(study.trials[0].value)

    def test_torch_scoring_disabled_uses_numpy(self):
        """With torch_scoring=False the NumPy-only loss is invoked directly."""
        runner = build_objective(
            model_creator=model_creator,
            search_space=search_space,
            data_loader=data_loader,
            loss_fn=numpy_only_loss,
            seed=9,
            torch_scoring=False,
        )
        study = optuna.create_study(direction="minimize")
        study.optimize(runner, n_trials=1)
        assert np.isfinite(study.trials[0].value)


# ── Acceptance criterion 4: trial_callbacks never fail the trial ─────────────
class TestTrialCallbacks:
    """trial_callbacks run but can never fail the trial."""

    def test_failing_callback_does_not_fail_trial(self):
        called = {"count": 0}

        def boom(trial, context):
            called["count"] += 1
            raise RuntimeError("callback intentionally explodes")

        runner = build_objective(
            model_creator=model_creator,
            search_space=search_space,
            data_loader=data_loader,
            loss_fn=mae_loss,
            seed=2,
            catch_exceptions=False,  # ensure the callback error is NOT masked
            trial_callbacks=[boom],
        )
        study = optuna.create_study(direction="minimize")
        study.optimize(runner, n_trials=1)
        assert called["count"] == 1
        assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
        assert study.trials[0].value is not None

    def test_callback_receives_context(self):
        received = {}

        def capture(trial, context):
            received.update(context)

        runner = build_objective(
            model_creator=model_creator,
            search_space=search_space,
            data_loader=data_loader,
            loss_fn=mae_loss,
            seed=2,
            trial_callbacks=[capture],
        )
        study = optuna.create_study(direction="minimize")
        study.optimize(runner, n_trials=1)
        assert "params" in received
        assert "raw_loss" in received
        assert "loss" in received

    def test_callback_can_signal_pruning(self):
        """A callback may raise TrialPruned to act as a pruning signal."""

        def prune_signal(trial, context):
            raise optuna.TrialPruned("monitor-as-pruning")

        runner = build_objective(
            model_creator=model_creator,
            search_space=search_space,
            data_loader=data_loader,
            loss_fn=mae_loss,
            seed=2,
            trial_callbacks=[prune_signal],
        )
        study = optuna.create_study(direction="minimize")
        study.optimize(runner, n_trials=1)
        assert study.trials[0].state == optuna.trial.TrialState.PRUNED
