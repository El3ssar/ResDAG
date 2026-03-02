"""Trial execution backends for HPO studies.

This module provides single-process and multi-process execution strategies for
running Optuna trials.  The multi-process backend uses ``fork``-based
multiprocessing so that the objective closure built in the parent is inherited
by workers without pickling.

See Also
--------
resdag.hpo.run : High-level orchestrator that selects and invokes a runner.
resdag.hpo.storage : Storage backend resolution used by workers.
"""

import logging
import multiprocessing as mp
import time
from typing import Callable

import optuna
import torch
from optuna.samplers import TPESampler
from tqdm.auto import tqdm

from .storage import resolve_storage

__all__ = ["run_single", "run_multiprocess"]

logger = logging.getLogger(__name__)


class _TrialProgressCallback:
    """Thread-safe tqdm progress bar callback for Optuna studies.

    Provides real-time feedback during single-process optimization by updating
    a ``tqdm`` bar after every completed trial and displaying the current best
    objective value.

    Parameters
    ----------
    n_trials : int
        Total number of trials expected (used as the progress bar total).
    """

    def __init__(self, n_trials: int) -> None:
        self._bar = tqdm(total=n_trials, desc="HPO Trials", unit="trial")

    def __call__(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
    ) -> None:
        """Update the progress bar after a trial completes.

        Parameters
        ----------
        study : optuna.Study
            The running study (used to read the current best value).
        trial : optuna.trial.FrozenTrial
            The trial that just finished.
        """
        self._bar.update(1)
        if trial.state == optuna.trial.TrialState.COMPLETE:
            try:
                self._bar.set_postfix(
                    best=f"{study.best_value:.4f}",
                    trial=trial.number,
                )
            except ValueError:
                pass

    def close(self) -> None:
        """Close the underlying tqdm bar."""
        self._bar.close()


def _worker_process(
    study_name: str,
    storage: str,
    objective: Callable[[optuna.Trial], float],
    target_total: int,
    worker_seed: int | None,
) -> None:
    """Execute trials in a forked worker process.

    Each worker opens its own storage connection, loads the shared study, and
    runs trials until the global completed-trial count reaches *target_total*.

    Parameters
    ----------
    study_name : str
        Name of the shared Optuna study.
    storage : str
        Raw storage path (journal file or SQLite URL) that the worker will
        independently open.
    objective : Callable[[Trial], float]
        Objective function inherited from the parent via ``fork``.
    target_total : int
        Stop once this many trials have been completed across all workers.
    worker_seed : int or None
        Seed for the worker's ``TPESampler``.  ``None`` for unseeded sampling.

    Notes
    -----
    Thread counts are pinned to 1 per worker to avoid oversubscription when
    many workers share the same CPU.
    """
    # Prevent thread oversubscription: each worker gets 1 thread.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Reconstruct the storage object in the worker process
    worker_storage = resolve_storage(storage, n_workers=1)

    sampler = TPESampler(
        multivariate=True,
        warn_independent_sampling=False,
        seed=worker_seed,
    )
    study = optuna.load_study(
        study_name=study_name,
        storage=worker_storage,
        sampler=sampler,
    )

    def _stop_when_done(
        study: optuna.Study,
        _trial: optuna.trial.FrozenTrial,
    ) -> None:
        done = len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        )
        if done >= target_total:
            study.stop()

    study.optimize(
        objective,
        n_trials=target_total,
        callbacks=[_stop_when_done],
        show_progress_bar=False,
        n_jobs=1,
    )


def run_single(
    study: optuna.Study,
    objective: Callable[[optuna.Trial], float],
    remaining: int,
    n_workers: int,
    verbosity: int,
) -> None:
    """Run optimization in a single process.

    Parameters
    ----------
    study : optuna.Study
        The study to optimize.
    objective : Callable[[Trial], float]
        Objective function to evaluate each trial.
    remaining : int
        Number of trials still to run.
    n_workers : int
        Number of Optuna ``n_jobs`` (thread-level parallelism within this
        process).
    verbosity : int
        Logging verbosity: ``0`` = silent, ``>= 1`` = show progress bar.

    Raises
    ------
    Exception
        Re-raises any non-``KeyboardInterrupt`` exception after logging.
    """
    progress_cb = _TrialProgressCallback(remaining) if verbosity >= 1 else None

    try:
        study.optimize(
            objective,
            n_trials=remaining,
            n_jobs=n_workers,
            show_progress_bar=False,
            callbacks=[progress_cb] if progress_cb else None,
        )
    except KeyboardInterrupt:
        logger.warning("Optimization interrupted by user")
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise
    finally:
        if progress_cb is not None:
            progress_cb.close()


def run_multiprocess(
    study_name: str,
    storage: str,
    objective: Callable[[optuna.Trial], float],
    n_trials: int,
    remaining: int,
    completed_trials: int,
    n_workers: int,
    seed: int | None,
    verbosity: int,
) -> None:
    """Run optimization with multiple OS processes sharing storage.

    Uses ``fork``-based multiprocessing so that the objective closure built
    in the parent process is available in workers without pickling.

    Parameters
    ----------
    study_name : str
        Name of the shared study in storage.
    storage : str
        Raw storage path that each worker will independently open.
    objective : Callable[[Trial], float]
        Objective function (inherited via ``fork``).
    n_trials : int
        Total target number of completed trials (including previously completed).
    remaining : int
        Number of new trials to run.
    completed_trials : int
        Number of trials already completed before this call.
    n_workers : int
        Number of parallel worker processes to spawn.
    seed : int or None
        Base random seed.  Each worker receives ``seed + i * 7919``.
    verbosity : int
        Logging verbosity: ``0`` = silent, ``>= 1`` = show progress bar.

    Notes
    -----
    The caller **must** release all storage references before calling this
    function to avoid inheriting file locks into forked children.  After this
    function returns the caller should reopen the storage to build the final
    ``Study`` object.
    """
    logger.info(f"Spawning {n_workers} worker processes")

    # Use fork context so the objective closure is inherited without pickling.
    ctx = mp.get_context("fork")

    processes: list[mp.process.BaseProcess] = []
    for i in range(n_workers):
        worker_seed = (seed + i * 7919) if seed is not None else None
        p = ctx.Process(
            target=_worker_process,
            args=(study_name, storage, objective, n_trials, worker_seed),
            daemon=True,
        )
        p.start()
        processes.append(p)

    # Reopen storage in the main process for progress monitoring.
    # This is safe because forked children have their own copies.
    monitor_storage = resolve_storage(storage, n_workers=1)
    monitor_study = optuna.load_study(
        study_name=study_name,
        storage=monitor_storage,
    )

    bar = tqdm(total=remaining, desc="HPO Trials", unit="trial") if verbosity >= 1 else None
    prev_count = completed_trials

    try:
        while any(p.is_alive() for p in processes):
            time.sleep(0.5)
            current = len(
                [t for t in monitor_study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            )
            delta = current - prev_count
            if delta > 0 and bar is not None:
                bar.update(delta)
                prev_count = current
                try:
                    bar.set_postfix(best=f"{monitor_study.best_value:.4f}")
                except ValueError:
                    pass
    except KeyboardInterrupt:
        logger.warning("Optimization interrupted by user — terminating workers")
        for p in processes:
            p.terminate()
    finally:
        for p in processes:
            p.join(timeout=5)
        if bar is not None:
            # Final sync in case we missed any
            current = len(
                [t for t in monitor_study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            )
            delta = current - prev_count
            if delta > 0:
                bar.update(delta)
                try:
                    bar.set_postfix(best=f"{monitor_study.best_value:.4f}")
                except ValueError:
                    pass
            bar.close()
