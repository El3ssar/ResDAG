"""Trial execution backends for HPO studies.

This module provides single-process and multi-process execution strategies for
running Optuna trials.  The multi-process backend is **start-method agnostic**:
it prefers ``fork`` (cheapest — no pickling) when the platform supports it, and
transparently falls back to ``spawn`` elsewhere (Windows, hardened macOS /
CPython 3.14).  Spawn works because the objective is a top-level *picklable*
:class:`resdag.hpo.runner.TrialRunner`, not a closure, so it can be shipped to
freshly-interpreted workers along with the other (already picklable) arguments.

Three correctness properties the multi-process backend guarantees:

- **Portability.** A usable start method is selected automatically via
  :func:`select_start_method`; if none is available an actionable
  :class:`RuntimeError` is raised early.
- **Bounded budget.** Each worker is given a small *local* trial budget
  (:func:`worker_budget`, ``ceil(remaining / n_workers) + slack``) rather than
  the global target, with a global stop callback as a safety net.  The
  completed-trial count therefore lands in
  ``[n_trials, n_trials + n_workers - 1]`` instead of overshooting by a full
  ``n_trials`` per worker.
- **Crash-safe interruption.** Workers are **non-daemon** and stop
  *cooperatively* via a shared :class:`multiprocessing.Event`; on
  ``KeyboardInterrupt`` the event is set and the workers are *joined* (allowed
  to flush their current write) instead of being ``terminate()``-d mid-write,
  which could corrupt the journal / SQLite backend.

See Also
--------
resdag.hpo.run : High-level orchestrator that selects and invokes a runner.
resdag.hpo.storage : Storage backend resolution used by workers.
"""

import logging
import math
import multiprocessing as mp
import time
from multiprocessing.context import ForkContext, SpawnContext
from typing import Callable, cast

import optuna
import torch
from optuna.samplers import TPESampler
from tqdm.auto import tqdm

from .storage import resolve_storage

__all__ = ["run_single", "run_multiprocess", "select_start_method", "worker_budget"]

logger = logging.getLogger(__name__)

# Worker-local budgets are padded by this many extra trials so that, if some
# trials are pruned/fail (and therefore do not count towards COMPLETE), workers
# still collectively reach ``remaining`` completed trials before exhausting
# their local budgets.  The global stop callback caps the true total.
_BUDGET_SLACK = 2


def select_start_method(preferred: str = "fork") -> str:
    """Pick a usable :mod:`multiprocessing` start method for the platform.

    Prefers *preferred* (``"fork"`` by default — the cheapest, since children
    inherit the parent's memory without pickling), falling back to ``"spawn"``
    when the preferred method is unavailable (e.g. Windows, or hardened
    macOS / CPython 3.14 where ``fork`` alongside threads is unsafe).  Raises if
    neither is available.

    Parameters
    ----------
    preferred : str, default="fork"
        The start method to use when the platform supports it.

    Returns
    -------
    str
        A start method name accepted by :func:`multiprocessing.get_context`
        (``"fork"`` or ``"spawn"``).

    Raises
    ------
    RuntimeError
        If neither *preferred* nor ``"spawn"`` is supported on this platform.
    """
    available = mp.get_all_start_methods()
    if preferred in available:
        return preferred
    if "spawn" in available:
        logger.info(
            "multiprocessing start method '%s' is unavailable on this platform; "
            "falling back to 'spawn'. The objective is picklable, so this is safe.",
            preferred,
        )
        return "spawn"
    raise RuntimeError(
        f"No usable multiprocessing start method found (wanted '{preferred}' or "
        f"'spawn'; available: {available}). Run with n_workers=1 for "
        "single-process optimization."
    )


def worker_budget(remaining: int, n_workers: int, slack: int = _BUDGET_SLACK) -> int:
    """Compute the bounded per-worker trial budget.

    Each worker runs at most ``ceil(remaining / n_workers) + slack`` trials so
    that no single worker can consume the entire global budget (which would let
    the study overshoot ``n_trials`` by up to ``n_trials`` trials).  The small
    *slack* keeps throughput high when some workers prune/fail trials; the
    global stop callback caps the true completed total.

    Parameters
    ----------
    remaining : int
        Total number of new trials still to run across all workers.
    n_workers : int
        Number of parallel worker processes (clamped to ``>= 1``).
    slack : int, default=2
        Extra per-worker headroom above the even split.

    Returns
    -------
    int
        The maximum number of trials a single worker should attempt.
    """
    n_workers = max(1, n_workers)
    return math.ceil(remaining / n_workers) + slack


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
    local_budget: int,
    worker_seed: int | None,
    pruner: "optuna.pruners.BasePruner | None",
    stop_event: "mp.synchronize.Event",
) -> None:
    """Execute trials in a worker process (``fork`` *or* ``spawn``).

    Each worker opens its own storage connection, loads the shared study, and
    runs **at most** *local_budget* trials.  It stops early when either the
    global completed-trial count reaches *target_total* (safety net against
    overshoot) or a cooperative *stop_event* is set by the parent (graceful
    interruption).  The local budget — not *target_total* — is the hard cap, so
    no single worker can run the whole global budget.

    Parameters
    ----------
    study_name : str
        Name of the shared Optuna study.
    storage : str
        Raw storage path (journal file or SQLite URL) that the worker will
        independently open.
    objective : Callable[[Trial], float]
        Objective function.  Inherited via ``fork`` or unpickled under
        ``spawn`` — it is a top-level picklable
        :class:`resdag.hpo.runner.TrialRunner`, so both work.
    target_total : int
        Global stop threshold: once this many trials are ``COMPLETE`` across all
        workers, the worker stops.  This is a *safety net*; the per-worker hard
        cap is *local_budget*.
    local_budget : int
        Maximum number of trials this worker may attempt (the bounded budget
        from :func:`worker_budget`).
    worker_seed : int or None
        Seed for the worker's ``TPESampler``.  ``None`` for unseeded sampling.
    pruner : optuna.pruners.BasePruner or None
        Pruner to attach to the worker's loaded study so it honors the same
        early-stopping policy as the parent process.  ``None`` leaves Optuna's
        default (no pruning) in place.
    stop_event : multiprocessing.Event
        Shared event; when set by the parent (e.g. on ``KeyboardInterrupt``),
        the worker stops cooperatively after its current trial rather than being
        forcibly terminated mid-write.

    Notes
    -----
    Thread counts are pinned to 1 per worker to avoid oversubscription when
    many workers share the same CPU.
    """
    # Prevent thread oversubscription: each worker gets 1 thread.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Reconstruct the storage object in the worker process. ``storage`` is a
    # non-None path string here, so ``resolve_storage`` always returns a
    # concrete backend (it only returns None for in-memory single-worker runs).
    worker_storage = resolve_storage(storage, n_workers=1)
    assert worker_storage is not None

    sampler = TPESampler(
        multivariate=True,
        warn_independent_sampling=False,
        seed=worker_seed,
    )
    study = optuna.load_study(
        study_name=study_name,
        storage=worker_storage,
        sampler=sampler,
        pruner=pruner,
    )

    def _stop_callback(
        study: optuna.Study,
        _trial: optuna.trial.FrozenTrial,
    ) -> None:
        # Cooperative interruption: flush-on-exit by returning from optimize()
        # rather than terminating, so the just-finished trial's write completes.
        if stop_event.is_set():
            study.stop()
            return
        # Global safety net against overshoot beyond the bounded local budget.
        done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        if done >= target_total:
            study.stop()

    study.optimize(
        objective,
        n_trials=local_budget,
        callbacks=[_stop_callback],
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
    pruner: "optuna.pruners.BasePruner | None" = None,
    verbosity: int = 1,
) -> None:
    """Run optimization with multiple OS processes sharing storage.

    Selects a portable start method (``fork`` when available, else ``spawn``),
    gives each worker a **bounded** local budget so the study cannot overshoot
    *n_trials* by a full budget per worker, and stops workers **cooperatively**
    via a shared :class:`multiprocessing.Event` so an interruption never
    ``terminate()``-s a worker mid-write.

    Parameters
    ----------
    study_name : str
        Name of the shared study in storage.
    storage : str
        Raw storage path that each worker will independently open.
    objective : Callable[[Trial], float]
        Objective function.  Inherited via ``fork`` or pickled to workers under
        ``spawn`` (it is a picklable
        :class:`resdag.hpo.runner.TrialRunner`, so both paths work).
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
    pruner : optuna.pruners.BasePruner or None, optional
        Early-stopping pruner attached to each worker's loaded study so the
        multi-process run honors the same policy as the parent.  ``None``
        (default) leaves Optuna's default (no pruning) in place.  Must be
        picklable for the ``spawn`` start method; Optuna's built-in pruners are.
    verbosity : int, default=1
        Logging verbosity: ``0`` = silent, ``>= 1`` = show progress bar.

    Raises
    ------
    RuntimeError
        If the platform supports no usable start method (neither ``fork`` nor
        ``spawn``); see :func:`select_start_method`.

    Notes
    -----
    The caller **must** release all storage references before calling this
    function to avoid inheriting file locks into forked children.  After this
    function returns the caller should reopen the storage to build the final
    ``Study`` object.

    Under ``spawn`` the *objective* and every other argument must be picklable.
    The objective is a top-level :class:`~resdag.hpo.runner.TrialRunner`; the
    user-supplied ``model_creator`` / ``search_space`` / ``data_loader`` it
    holds must likewise be picklable (top-level functions, not lambdas or local
    closures) for fork-less platforms.
    """
    # Portable start method: fork if available (no pickling), else spawn. Raises
    # an actionable RuntimeError when neither is supported. The concrete
    # ForkContext/SpawnContext both expose ``Process`` / ``Event``; the typeshed
    # ``BaseContext`` does not, so narrow to the union for the type checker.
    start_method = select_start_method("fork")
    ctx: ForkContext | SpawnContext = cast(
        "ForkContext | SpawnContext", mp.get_context(start_method)
    )
    logger.info(f"Spawning {n_workers} worker processes (start method: {start_method})")

    # Bounded per-worker budget so no single worker can run the whole global
    # budget; the global stop callback inside the worker remains a safety net.
    local_budget = worker_budget(remaining, n_workers)

    # Cooperative stop signal — set on KeyboardInterrupt so workers flush their
    # current write and exit cleanly instead of being terminated mid-write.
    stop_event = ctx.Event()

    processes: list[mp.process.BaseProcess] = []
    for i in range(n_workers):
        worker_seed = (seed + i * 7919) if seed is not None else None
        # Annotate as BaseProcess so the later monitor loops (which type ``p`` as
        # BaseProcess) agree with this context's Process instance. Non-daemon so
        # workers can finish their in-flight storage write on shutdown.
        p: mp.process.BaseProcess = ctx.Process(
            target=_worker_process,
            args=(
                study_name,
                storage,
                objective,
                n_trials,
                local_budget,
                worker_seed,
                pruner,
                stop_event,
            ),
            daemon=False,
        )
        p.start()
        processes.append(p)

    # Reopen storage in the main process for progress monitoring.
    # This is safe because forked children have their own copies. ``storage`` is
    # a non-None path string, so the resolved backend is never None here.
    monitor_storage = resolve_storage(storage, n_workers=1)
    assert monitor_storage is not None
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
        logger.warning(
            "Optimization interrupted by user — signalling workers to stop "
            "cooperatively (current trial writes will be flushed)"
        )
        stop_event.set()
    finally:
        # Always signal stop so non-daemon workers terminate even on normal exit
        # or an unexpected error; already-finished workers are unaffected.
        stop_event.set()
        # Join without a timeout so a worker is allowed to finish its in-flight
        # storage write rather than being abandoned/terminated mid-write.  The
        # cooperative stop_event guarantees each worker exits after at most one
        # more trial.
        for p in processes:
            p.join()
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
