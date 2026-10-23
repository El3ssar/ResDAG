"""Parent-owned Optuna bookkeeping with the shared managed joblib executor."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any

import optuna
from tqdm import tqdm

from ._parallel import iter_jobs
from ._trial_progress import TrialProgress
from .resources import (
    MemoryBudget,
    MemoryContext,
    MemoryEstimate,
    estimate_memory,
    raw_memory_context,
)
from .trial_runner import PreparedTrial, TrialEvaluationError, TrialRunner

__all__ = ["run_trials"]


def _fail_active(
    study: optuna.Study, trials: dict[int, optuna.Trial], error: BaseException | None
) -> None:
    """Release only this call's reservations, preserving the primary exception."""
    first_error = error
    for number in tuple(trials):
        try:
            study.tell(number, state=optuna.trial.TrialState.FAIL, skip_if_finished=True)
        except BaseException as cleanup_error:
            if first_error is None:
                first_error = cleanup_error
            else:
                first_error.add_note(f"Trial {number} cleanup also failed: {cleanup_error}")
    if error is None and first_error is not None:
        raise first_error


def run_trials(
    study: optuna.Study,
    runner: TrialRunner,
    *,
    n_trials: int,
    n_workers: int = 1,
    verbosity: int = 1,
    inner_max_num_threads: int = 1,
    memory_budget: MemoryBudget | None = None,
    memory_estimator: Callable[[dict[str, Any], MemoryContext], MemoryEstimate | int] | None = None,
) -> None:
    """Run finite new attempts with sampling/storage in the caller only.

    Managed loky workers receive numerical jobs without Trial/storage objects.
    Failed/pruned configurations consume attempts. Finite submission windows
    prevent background consumption of future attempts. Errors and interrupts
    propagate after this call's RUNNING reservations are marked failed.
    """
    from joblib import effective_n_jobs

    window = effective_n_jobs(n_workers)
    active: dict[int, optuna.Trial] = {}
    worker = runner.for_worker()
    pruning = not isinstance(study.pruner, optuna.pruners.NopPruner)
    bar = tqdm(total=n_trials, desc="HPO attempts", unit="trial") if verbosity else None
    primary_error: BaseException | None = None
    attempts = 0
    pending: tuple[PreparedTrial, MemoryEstimate] | None = None
    try:
        with TrialProgress(parallel=n_workers != 1 and pruning) as progress:
            while attempts < n_trials or pending is not None:
                callbacks = []
                window_error: BaseException | None = None
                try:
                    jobs: list[tuple[PreparedTrial, Any]] = []
                    estimates: list[MemoryEstimate] = []
                    while len(jobs) < window and (attempts < n_trials or pending is not None):
                        if pending is not None:
                            prepared, estimate = pending
                            pending = None
                            trial = active[prepared.number]
                        else:
                            trial = study.ask()
                            active[trial.number] = trial
                            attempts += 1
                            try:
                                prepared = runner.prepare(trial)
                            except optuna.TrialPruned:
                                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                                active.pop(trial.number)
                                if bar:
                                    bar.update()
                                continue
                            except Exception as error:
                                if not runner.catch_exceptions:
                                    raise
                                trial.set_user_attr("error", str(error))
                                if runner.error_policy == "penalty":
                                    penalty = abs(runner.penalty_value)
                                    study.tell(
                                        trial,
                                        penalty if study.direction.name == "MINIMIZE" else -penalty,
                                    )
                                else:
                                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                                active.pop(trial.number)
                                if bar:
                                    bar.update()
                                continue
                            estimate = None
                            if memory_budget is not None:
                                horizon = (
                                    int(prepared.data[runner.horizon_key])
                                    if runner.horizon_key and runner.horizon_key in prepared.data
                                    else int(prepared.data["val"].shape[1])
                                )
                                context = raw_memory_context(
                                    prepared.data,
                                    device=runner.device,
                                    dtype=prepared.data["train"].dtype,
                                    horizon=horizon,
                                )
                                estimate = estimate_memory(
                                    prepared.params,
                                    context,
                                    custom_factory=True,
                                    estimator=memory_estimator,
                                )
                                metadata = {
                                    "host_bytes": estimate.host_bytes,
                                    "device_bytes": estimate.device_bytes,
                                    "details": json.loads(
                                        json.dumps(
                                            dict(estimate.details), default=str, allow_nan=False
                                        )
                                    ),
                                }
                                trial.set_user_attr("memory_estimate", metadata)
                                prepared.attrs["memory_estimate"] = metadata
                                if not memory_budget.fits(estimate):
                                    trial.set_user_attr(
                                        "pruned_reason",
                                        "memory_budget: estimated candidate exceeds the planning limit",
                                    )
                                    study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                                    active.pop(trial.number)
                                    if bar:
                                        bar.update()
                                    continue
                        if memory_budget is not None and estimate is not None:
                            if jobs and not memory_budget.fits(estimate, estimates):
                                pending = prepared, estimate
                                break
                            estimates.append(estimate)
                        callback = progress.callback(trial) if pruning else None
                        if callback is not None:
                            callbacks.append(callback)
                        jobs.append((prepared, callback))
                    if jobs:
                        with iter_jobs(
                            worker.evaluate_job,
                            jobs,
                            n_jobs=n_workers,
                            inner_max_num_threads=inner_max_num_threads,
                        ) as results:
                            for result in results:
                                progress.check()
                                trial = active[result.number]
                                try:
                                    value = runner.finalize(trial, result)
                                except optuna.TrialPruned:
                                    study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                                except TrialEvaluationError:
                                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                                else:
                                    if not math.isfinite(value):
                                        raise ValueError(
                                            "An evaluation returned a non-finite objective"
                                        )
                                    study.tell(trial, value)
                                active.pop(result.number)
                                if bar:
                                    bar.update()
                    progress.check()
                except BaseException as error:
                    window_error = error
                    raise
                finally:
                    release_error = window_error
                    for callback in callbacks:
                        try:
                            progress.release(callback)
                        except BaseException as cleanup_error:
                            if release_error is None:
                                release_error = cleanup_error
                            else:
                                release_error.add_note(
                                    f"Progress cleanup also failed: {cleanup_error}"
                                )
                    if window_error is None and release_error is not None:
                        raise release_error
    except BaseException as error:
        primary_error = error
        raise
    finally:
        cleanup_primary = primary_error
        try:
            _fail_active(study, active, primary_error)
        except BaseException as cleanup_error:
            cleanup_primary = cleanup_error
        if bar is not None:
            try:
                bar.close()
            except BaseException as cleanup_error:
                if cleanup_primary is None:
                    cleanup_primary = cleanup_error
                else:
                    cleanup_primary.add_note(f"Progress bar cleanup also failed: {cleanup_error}")
        if primary_error is None and cleanup_primary is not None:
            raise cleanup_primary
