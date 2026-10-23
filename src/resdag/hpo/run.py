"""Advanced raw-model tuning with parent-owned Optuna and managed joblib jobs."""

from __future__ import annotations

import logging
import math
import platform
import warnings
from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, Literal

import optuna
import torch
from optuna.samplers import TPESampler

from resdag.core import ESNModel

from ._provenance import PROTOCOL_KEY, bind_protocol, callable_identity, fingerprint, trial_settings
from .dispatch import run_trials
from .losses import LossProtocol, get_loss
from .objective import build_objective
from .pruners import resolve_pruner
from .resources import MemoryBudget, MemoryContext, MemoryEstimate
from .storage import _close_storage, resolve_storage
from .transfer import _source_study, apply_warm_start, transfer_trials
from .trial_runner import TrialCallback, TrialRunner, _rng_scope, _validate_data_keys
from .utils import make_study_name

__all__ = ["run_hpo"]
logger = logging.getLogger(__name__)


def _configure_logging(verbosity: int) -> None:
    level = logging.ERROR if verbosity == 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.getLogger("resdag").setLevel(level)
    optuna.logging.set_verbosity(level)


def _validate_previous_data(
    study: optuna.Study, runner: TrialRunner, task_context: Mapping[str, Any] | str | None
) -> None:
    """Re-materialize prior configurations before their old score can win.

    FixedTrial is Optuna's standard replay interface; no models are fitted and
    no historical trial is mutated. A callback that needs a live study rather
    than replayable suggestions requires an explicit trusted task context.
    """
    for previous in study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)):
        if "error" in previous.user_attrs or previous.user_attrs.get("nonfinite_loss", False):
            continue
        expected = previous.user_attrs.get("resdag_raw_data_hash")
        if expected is None:
            raise ValueError(
                "Old raw scores have no recorded data identity; re-evaluate their settings"
            )
        source = previous.user_attrs.get("resdag_transfer", {})
        original_number = previous.user_attrs.get(
            "resdag_raw_evaluation_number", source.get("trial_number", previous.number)
        )
        settings = trial_settings(previous)
        replay = optuna.trial.FixedTrial({**previous.params, **settings}, number=original_number)
        replay_trial: Any = replay  # Optuna's public duck-typed replay interface.
        for key, value in previous.user_attrs.items():
            replay.set_user_attr(key, value)
        try:
            with _rng_scope(runner.device):
                runner._seed_trial(replay_trial)
                runner.search_space(replay_trial)
                data = runner.data_loader(replay_trial)
                _validate_data_keys(data, drivers_keys=runner.drivers_keys)
                current = fingerprint(data)
        except (AttributeError, TypeError) as error:
            if task_context is None:
                raise ValueError(
                    "Cannot verify historical data with Optuna FixedTrial. Supply a trusted "
                    "task_context identifying external data/resources, or re-evaluate in a fresh study"
                ) from error
            warnings.warn(
                "Historical data callback is not replayable; trusting the explicit "
                "task_context for its external resources",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        if current != expected:
            raise ValueError(
                "Study data changed for an evaluated configuration; reuse its settings "
                "in a fresh study and re-evaluate them"
            )


def run_hpo(
    model_creator: Callable[..., ESNModel],
    search_space: Callable[[optuna.Trial], dict[str, Any]],
    data_loader: Callable[[optuna.Trial], dict[str, Any]],
    n_trials: int,
    loss: str | LossProtocol = "efh",
    loss_params: dict[str, Any] | None = None,
    targets_key: str = "output",
    drivers_keys: list[str] | None = None,
    monitor_losses: list[str | LossProtocol] | None = None,
    monitor_params: dict[str, dict[str, Any]] | None = None,
    study_name: str | None = None,
    storage: str | None = None,
    warm_start: list[dict[str, Any]] | None = None,
    transfer_from: optuna.Study | str | None = None,
    sampler: optuna.samplers.BaseSampler | None = None,
    pruner: str | optuna.pruners.BasePruner | None = None,
    seed: int | None = None,
    device: str | torch.device | None = None,
    n_workers: int = 1,
    verbosity: int = 1,
    catch_exceptions: bool = True,
    penalty_value: float = 1e10,
    clip_value: float | None = None,
    prune_on_clip: bool = False,
    horizon_key: str | None = None,
    n_checkpoints: int = 5,
    torch_scoring: bool = True,
    trial_callbacks: list[TrialCallback] | None = None,
    error_policy: Literal["fail", "penalty"] = "fail",
    *,
    transfer_mode: Literal["reevaluate", "scores"] = "reevaluate",
    task_context: Mapping[str, Any] | str | None = None,
    inner_max_num_threads: int = 1,
    memory_budget: MemoryBudget | int | str | None = None,
    memory_estimator: Callable[[dict[str, Any], MemoryContext], MemoryEstimate | int] | None = None,
) -> optuna.Study:
    """Tune raw ESNModel factories with explicit five-array data preparation.

    The usual ``Model.tune``/``ESN.tune`` workflow prepares chronological views
    and installs its winner. This advanced entry point returns an Optuna study;
    callers own split alignment, scaling and winner reconstruction.

    ``n_trials`` counts NEW attempts in this call, including failures/pruning.
    Single-worker jobs run in the caller; multiple workers use the shared
    managed joblib/loky executor, including notebook closures. Search-space and
    data callbacks run once per new attempt in the parent with a real Trial.
    Numerical workers receive no study/storage connection. Trial callbacks and
    all Optuna bookkeeping are parent-owned; errors/interrupts propagate after
    cleanup without leaving this call's reservations RUNNING.

    Parameters
    ----------
    model_creator : callable
        ``model_creator(**resolved_settings) -> ESNModel``. Fixed settings as
        well as suggested parameters are recorded. Capture custom Python
        modules/initializers in this callable rather than in JSON settings.
    search_space : callable
        ``search_space(trial) -> mapping`` of finite JSON/literal settings.
    data_loader : callable
        Returns 3D tensors keyed ``warmup``, ``train``, ``target``, ``f_warmup``
        and ``val``. Each driver requires warmup/train/f_warmup/forecast entries
        with its name suffix. This function may also make Optuna suggestions.
    n_trials : int
        Positive count of NEW attempted configurations; transferred scores do
        not consume this budget. Resume adds this many attempts.
    loss, loss_params : str or callable, dict, optional
        Raw forecast objective and options. Built-in/default losses minimize.
    targets_key, drivers_keys : str, list[str], optional
        Named target readout and declared driver data suffixes.
    monitor_losses, monitor_params : list, dict, optional
        Additional final-horizon losses, recorded without changing the objective.
    study_name, storage : str, optional
        In-memory by default, or an explicit journal/SQLite study. Only the
        parent opens storage, so in-memory studies work with multiple workers.
    warm_start : list[dict], optional
        Enqueue fixed settings for evaluation, preserving unsuggested values.
    transfer_from : optuna.Study or str, optional
        Prior study or a path containing exactly one study. Full configurations
        are re-evaluated by default; source provenance is recorded.
    sampler, pruner : Optuna objects or registered pruner name, optional
        Parent-owned sampling/pruning. Pruning at streamed horizon checkpoints
        stops future forecast generation; raw prefix scoring can retain history.
    seed : int, optional
        Trial initialization seed is seed + trial.number. Caller RNG streams
        are restored; the post-data-loading RNG sequence reaches the worker.
    device : str or torch.device, optional
        Numerical model/data device. Parameter/scoring operations stay there
        when supported, with the existing NumPy custom-loss fallback.
    n_workers : int, default=1
        Positive joblib worker count; one runs in the caller.
    verbosity : int, default=1
        Zero is quiet. Caller logging levels are restored after this call.
    catch_exceptions : bool, default=True
        Record invalid evaluation attempts rather than aborting on their error.
    penalty_value, clip_value, prune_on_clip : optional
        Explicit numeric failure penalty or clipping/pruning controls.
        ``error_policy="fail"`` prevents invalid evaluations becoming winners.
    horizon_key : str, optional
        Data key specifying the forecast length; cannot exceed validation truth.
    n_checkpoints : int, default=5
        Growing forecast prefixes; reports use 1-based checkpoint indices.
    torch_scoring : bool, default=True
        Prefer on-device numeric scoring until the final scalar.
    trial_callbacks : list[callable], optional
        Parent callbacks receive the real Trial and params/raw_loss/loss.
        TrialPruned propagates; other callback errors are logged/ignored.
    error_policy : {"fail", "penalty"}, default="fail"
        Invalid evaluation representation. Penalty results are never valid winners.
    transfer_mode : {"reevaluate", "scores"}, default="reevaluate"
        Copy evaluated scores only for identical bound task/objective protocols.
    task_context : mapping or str, optional
        Explicit trusted version/content identity for external data/resources.
        Prior materialized data is checked before old scores can win. Callbacks
        requiring a live study on replay need this explicit assertion.
    inner_max_num_threads : int, default=1
        joblib worker thread limit; caller thread settings remain unchanged.
    memory_budget, memory_estimator : optional
        Estimated-memory admission and reduced numerical concurrency. An
        arbitrary raw factory requires a custom estimator. Oversized candidates
        are PRUNED before model construction; raw prefix storage is included in
        the known minimum. These estimates do not impose a process/RAM limit.
    """
    if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials < 1:
        raise ValueError("n_trials must be a positive integer")
    if isinstance(n_workers, bool) or not isinstance(n_workers, int) or n_workers < 1:
        raise ValueError("n_workers must be positive")
    if (
        isinstance(inner_max_num_threads, bool)
        or not isinstance(inner_max_num_threads, int)
        or inner_max_num_threads < 1
    ):
        raise ValueError("inner_max_num_threads must be a positive integer")
    for callback, name in (
        (model_creator, "model_creator"),
        (search_space, "search_space"),
        (data_loader, "data_loader"),
    ):
        if not callable(callback):
            raise TypeError(f"{name} must be callable")
    if error_policy not in ("fail", "penalty"):
        raise ValueError("error_policy must be 'fail' or 'penalty'")
    if transfer_mode not in ("reevaluate", "scores"):
        raise ValueError("transfer_mode must be 'reevaluate' or 'scores'")
    if not math.isfinite(penalty_value):
        raise ValueError("penalty_value must be finite")
    budget = (
        memory_budget
        if isinstance(memory_budget, MemoryBudget) or memory_budget is None
        else MemoryBudget(memory_budget)
    )
    if budget is not None and memory_estimator is None:
        raise ValueError("Raw model factories require memory_estimator with memory_budget")
    if memory_estimator is not None and (budget is None or not callable(memory_estimator)):
        raise ValueError("memory_estimator must be callable and requires memory_budget")
    resolved_device = None if device is None else torch.device(device)
    levels = logging.getLogger("resdag").level, optuna.logging.get_verbosity()
    _configure_logging(verbosity)
    resolved_storage = None
    primary_error: BaseException | None = None
    try:
        # No worker connects to storage: a memory study is safe even in parallel.
        resolved_storage, _ = resolve_storage(storage, n_workers=1, return_path=True)
        base_loss = get_loss(loss)
        resolved_loss = partial(base_loss, **loss_params) if loss_params else base_loss
        monitors = [
            get_loss(item) if isinstance(item, str) else item for item in monitor_losses or ()
        ]
        mapped_monitor_params = dict(monitor_params or {})
        for original, resolved in zip(monitor_losses or (), monitors, strict=True):
            if isinstance(original, str) and original in mapped_monitor_params:
                mapped_monitor_params[getattr(resolved, "__name__", original)] = (
                    mapped_monitor_params.pop(original)
                )
        runner = build_objective(
            model_creator=model_creator,
            search_space=search_space,
            data_loader=data_loader,
            loss_fn=resolved_loss,
            targets_key=targets_key,
            drivers_keys=drivers_keys,
            horizon_key=horizon_key,
            catch_exceptions=catch_exceptions,
            penalty_value=penalty_value,
            monitor_losses=monitors,
            monitor_params=mapped_monitor_params,
            device=resolved_device,
            seed=seed,
            clip_value=clip_value,
            prune_on_clip=prune_on_clip,
            n_checkpoints=n_checkpoints,
            torch_scoring=torch_scoring,
            trial_callbacks=trial_callbacks,
            error_policy=error_policy,
        )
        payload = {
            "version": 2,
            "kind": "raw",
            "model": callable_identity(model_creator),
            "space": callable_identity(search_space),
            "data": callable_identity(data_loader),
            "loss": callable_identity(resolved_loss),
            "targets_key": targets_key,
            "drivers_keys": drivers_keys,
            "horizon_key": horizon_key,
            "seed": seed,
            "clip_value": clip_value,
            "prune_on_clip": prune_on_clip,
            "n_checkpoints": n_checkpoints,
            "torch_scoring": torch_scoring,
            "task_context": task_context,
            "error_policy": error_policy,
            "device": resolved_device,
            "environment": (platform.python_version(), torch.__version__, optuna.__version__),
        }
        protocol = fingerprint(payload)
        study = optuna.create_study(
            direction="minimize",
            study_name=study_name or make_study_name(model_creator),
            storage=resolved_storage,
            load_if_exists=True,
            sampler=sampler
            or TPESampler(multivariate=True, warn_independent_sampling=False, seed=seed),
            pruner=resolve_pruner(pruner),
        )
        bind_protocol(study, protocol)
        _validate_previous_data(study, runner, task_context)
        if transfer_from is not None:
            if transfer_mode == "scores":
                with _source_study(transfer_from) as source:
                    if source.user_attrs.get(PROTOCOL_KEY) != protocol:
                        raise ValueError(
                            "Score transfer requires identical task protocols; re-evaluate settings"
                        )
                    _validate_previous_data(source, runner, task_context)
                    transfer_trials(study, source, mode="scores", protocol_hash=protocol)
            else:
                transfer_trials(study, transfer_from)
        if warm_start:
            apply_warm_start(study, warm_start)
        run_trials(
            study,
            runner,
            n_trials=n_trials,
            n_workers=n_workers,
            verbosity=verbosity,
            inner_max_num_threads=inner_max_num_threads,
            memory_budget=budget,
            memory_estimator=memory_estimator,
        )
        completed = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
        valid = [
            trial
            for trial in completed
            if "error" not in trial.user_attrs
            and not trial.user_attrs.get("nonfinite_loss", False)
            and trial.value is not None
            and math.isfinite(trial.value)
        ]
        failed = study.get_trials(states=(optuna.trial.TrialState.FAIL,))
        if not valid and (completed or failed):
            detail = (failed or completed)[0].user_attrs.get(
                "error", "Invalid/non-finite evaluation"
            )
            raise RuntimeError(f"No valid HPO trials completed. First evaluation error: {detail}")
        if valid and study.best_trial not in valid:
            raise RuntimeError(
                "The best score belongs to an invalid penalty evaluation; use error_policy='fail'"
            )
        return study
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            _close_storage(
                resolved_storage,
                primary_error=primary_error,
                keep_study_usable=primary_error is None,
            )
        finally:
            logging.getLogger("resdag").setLevel(levels[0])
            optuna.logging.set_verbosity(levels[1])
