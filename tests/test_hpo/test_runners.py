"""Tests for the HPO trial-execution backends (``resdag.hpo.runners``).

These cover the three correctness properties of the multi-process backend
guaranteed by issue #276:

- **Portability** — :func:`select_start_method` prefers ``fork`` but falls back
  to ``spawn`` on fork-less platforms (and errors actionably when neither is
  available), so ``n_workers > 1`` works everywhere.
- **Bounded budget** — :func:`worker_budget` and an end-to-end run keep the
  completed-trial count within ``[n_trials, n_trials + n_workers - 1]``.
- **Crash-safe interruption** — a cooperatively interrupted parallel study
  leaves the storage backend readable (not corrupted).

The model / search-space / data-loader callables are defined at *module level*
so they are picklable — required for the ``spawn`` start method.
"""

import multiprocessing as mp
from pathlib import Path

import optuna
import pytest
import torch

from resdag.hpo import run_hpo
from resdag.hpo.losses import get_loss
from resdag.hpo.objective import build_objective
from resdag.hpo.runners import _worker_process, select_start_method, worker_budget
from resdag.hpo.storage import resolve_storage
from resdag.models import ott_esn


# ── Picklable top-level fixtures (needed for the spawn start method) ──────────
def _model_creator(reservoir_size: int = 30, spectral_radius: float = 0.9):
    """Tiny model creator (small reservoir → fast trials)."""
    return ott_esn(
        reservoir_size=reservoir_size,
        feedback_size=3,
        output_size=3,
        spectral_radius=spectral_radius,
    )


def _search_space(trial):
    """Minimal search space."""
    return {
        "reservoir_size": trial.suggest_int("reservoir_size", 20, 40, step=10),
        "spectral_radius": trial.suggest_float("spectral_radius", 0.7, 1.1),
    }


def _data_loader(trial):
    """Small synthetic dataset."""
    torch.manual_seed(0)
    data = torch.randn(1, 150, 3)
    return {
        "warmup": data[:, :20, :],
        "train": data[:, 20:70, :],
        "target": data[:, 21:71, :],
        "f_warmup": data[:, 70:90, :],
        "val": data[:, 90:100, :],
    }


# ── select_start_method ──────────────────────────────────────────────────────
class TestSelectStartMethod:
    """Portability: a usable start method is always chosen (or errors cleanly)."""

    def test_prefers_available_method(self, monkeypatch):
        """The preferred method is returned verbatim when supported."""
        monkeypatch.setattr(mp, "get_all_start_methods", lambda: ["fork", "spawn"])
        assert select_start_method("fork") == "fork"

    def test_falls_back_to_spawn_when_fork_absent(self, monkeypatch):
        """Fork-less platforms (Windows / hardened macOS) fall back to spawn."""
        monkeypatch.setattr(mp, "get_all_start_methods", lambda: ["spawn"])
        assert select_start_method("fork") == "spawn"

    def test_raises_when_no_method_available(self, monkeypatch):
        """An empty platform raises an actionable error mentioning n_workers=1."""
        monkeypatch.setattr(mp, "get_all_start_methods", lambda: [])
        with pytest.raises(RuntimeError, match="n_workers=1"):
            select_start_method("fork")

    def test_real_platform_returns_usable_method(self):
        """On the real platform a context can actually be built from the result."""
        method = select_start_method("fork")
        assert method in mp.get_all_start_methods()
        # Must be constructable — this is what run_multiprocess does.
        mp.get_context(method)


# ── worker_budget ────────────────────────────────────────────────────────────
class TestWorkerBudget:
    """Bounded budget: each worker gets ceil(remaining / W) + slack trials."""

    def test_even_split_plus_slack(self):
        """10 trials / 2 workers → 5 + slack per worker."""
        assert worker_budget(10, 2, slack=2) == 7

    def test_rounds_up(self):
        """Uneven splits round up so the study can still reach `remaining`."""
        assert worker_budget(10, 3, slack=0) == 4  # ceil(10/3) == 4

    def test_single_worker_gets_full_budget(self):
        """With one worker the budget covers all remaining trials (+ slack)."""
        assert worker_budget(10, 1, slack=0) == 10

    def test_clamps_nonpositive_workers(self):
        """`n_workers <= 0` is clamped to 1 rather than dividing by zero."""
        assert worker_budget(5, 0, slack=0) == 5

    def test_sum_of_budgets_covers_remaining(self):
        """Sum of per-worker budgets is always >= remaining (no under-run)."""
        for remaining in (1, 5, 7, 13, 100):
            for n_workers in (1, 2, 3, 4, 8):
                budget = worker_budget(remaining, n_workers, slack=0)
                assert budget * n_workers >= remaining

    def test_no_single_worker_runs_whole_budget(self):
        """For W > 1 the per-worker budget is strictly below `remaining`.

        This is the core overshoot fix: previously each worker received the
        global ``n_trials`` as its budget.
        """
        remaining, n_workers = 20, 4
        assert worker_budget(remaining, n_workers, slack=2) < remaining


# ── End-to-end multiprocess runs ─────────────────────────────────────────────
@pytest.mark.slow
class TestMultiprocessEndToEnd:
    """Integration tests that actually spawn workers via ``run_hpo``."""

    def test_n_workers_gt_1_runs(self, tmp_path: Path):
        """`n_workers > 1` completes a study and produces a best value."""
        study = run_hpo(
            model_creator=_model_creator,
            search_space=_search_space,
            data_loader=_data_loader,
            n_trials=6,
            n_workers=2,
            storage=str(tmp_path / "study.log"),
            study_name="mp_runs",
            verbosity=0,
        )
        assert study.best_value is not None

    def test_completed_count_within_overshoot_bound(self, tmp_path: Path):
        """AC: completed trials land in [n_trials, n_trials + n_workers - 1]."""
        n_trials, n_workers = 6, 3
        study = run_hpo(
            model_creator=_model_creator,
            search_space=_search_space,
            data_loader=_data_loader,
            n_trials=n_trials,
            n_workers=n_workers,
            storage=str(tmp_path / "bound.log"),
            study_name="mp_bound",
            verbosity=0,
        )
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        assert n_trials <= completed <= n_trials + n_workers - 1

    def test_sqlite_backend_readable_after_run(self, tmp_path: Path):
        """A multiprocess SQLite study reloads cleanly (no corruption)."""
        db = tmp_path / "study.db"
        run_hpo(
            model_creator=_model_creator,
            search_space=_search_space,
            data_loader=_data_loader,
            n_trials=4,
            n_workers=2,
            storage=f"sqlite:///{db}",
            study_name="mp_sqlite",
            verbosity=0,
        )
        # Reopening must not raise — corruption would surface here.
        reloaded = optuna.load_study(
            study_name="mp_sqlite",
            storage=f"sqlite:///{db}",
        )
        completed = len([t for t in reloaded.trials if t.state == optuna.trial.TrialState.COMPLETE])
        assert completed >= 4


@pytest.mark.slow
class TestCooperativeInterruption:
    """Crash-safe interruption: a pre-signalled stop leaves storage readable."""

    def test_preset_stop_event_exits_cleanly_without_corruption(self, tmp_path: Path):
        """A worker started with the stop event already set exits gracefully.

        This exercises the cooperative-stop path (the same one taken on
        ``KeyboardInterrupt``): instead of being ``terminate()``-d mid-write, the
        worker observes the event and returns from ``study.optimize`` after at
        most one trial, leaving the journal/SQLite backend readable.
        """
        storage_path = str(tmp_path / "interrupt.log")

        # Create the study (and storage schema) up front.
        store = resolve_storage(storage_path, n_workers=1)
        assert store is not None
        optuna.create_study(
            direction="minimize",
            study_name="mp_interrupt",
            storage=store,
        )
        del store

        objective = build_objective(
            model_creator=_model_creator,
            search_space=_search_space,
            data_loader=_data_loader,
            loss_fn=get_loss("efh"),
        )

        ctx = mp.get_context(select_start_method("fork"))
        stop_event = ctx.Event()
        stop_event.set()  # Signal stop before the worker runs any trial.

        proc = ctx.Process(
            target=_worker_process,
            # Signature: (study_name, storage, objective, n_trials, local_budget,
            #             worker_seed, pruner, stop_event). ``None`` pruner =
            #             Optuna default (no pruning); this test exercises the
            #             cooperative-stop path, not pruning.
            args=("mp_interrupt", storage_path, objective, 100, 100, 0, None, stop_event),
            daemon=False,
        )
        proc.start()
        proc.join(timeout=60)

        assert not proc.is_alive(), "worker did not stop cooperatively"
        assert proc.exitcode == 0, "worker crashed instead of exiting cleanly"

        # Storage must still be readable (uncorrupted) after the interruption.
        reopened = resolve_storage(storage_path, n_workers=1)
        assert reopened is not None
        reloaded = optuna.load_study(study_name="mp_interrupt", storage=reopened)
        # At most one trial may have started before the event was observed.
        completed = len([t for t in reloaded.trials if t.state == optuna.trial.TrialState.COMPLETE])
        assert completed <= 1
