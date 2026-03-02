"""High-level HPO orchestrator for reservoir computing models.

This module exposes a single public function, :func:`run_hpo`, which wires
together storage resolution, study creation, objective construction, and trial
execution into a complete hyperparameter optimization pipeline.

See Also
--------
resdag.hpo.storage : Storage backend resolution.
resdag.hpo.runners : Single- and multi-process trial execution.
resdag.hpo.objective : Objective function builder.
resdag.hpo.losses : Available loss functions.
"""

import logging
import os
from functools import partial
from typing import Any, Callable

import optuna
import torch
from optuna.samplers import TPESampler

from resdag.composition import ESNModel

from .losses import LossProtocol, get_loss
from .objective import build_objective
from .runners import run_multiprocess, run_single
from .storage import enable_sqlite_wal, resolve_storage
from .utils import make_study_name

__all__ = ["run_hpo"]

logger = logging.getLogger(__name__)


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
    sampler: optuna.samplers.BaseSampler | None = None,
    seed: int | None = None,
    device: str | torch.device | None = None,
    n_workers: int = 1,
    verbosity: int = 1,
    catch_exceptions: bool = True,
    penalty_value: float = 1e10,
    clip_value: float | None = None,
    prune_on_clip: bool = False,
) -> optuna.Study:
    """Run an Optuna hyperparameter optimization study for ESN models.

    This function provides a complete HPO pipeline that handles model creation,
    training, forecasting, and evaluation with robust error handling.

    When ``n_workers > 1``, optimization is parallelized using real OS processes
    (not threads) that coordinate via shared file storage. By default, uses
    Optuna's ``JournalFileStorage`` which is designed for multi-process
    coordination. Pass a ``.db`` path or ``sqlite:///`` URL to use SQLite
    instead (with WAL mode for concurrency).

    Parameters
    ----------
    model_creator : Callable[..., ESNModel]
        Function that creates a fresh model for each trial. Must accept all
        hyperparameters from ``search_space`` as keyword arguments.
    search_space : Callable[[Trial], dict[str, Any]]
        Function that defines the hyperparameter search space. Uses Optuna's
        ``trial.suggest_*`` methods and returns a dictionary of parameters.
    data_loader : Callable[[Trial], dict[str, Any]]
        Function that loads and returns data. Must return a dictionary with:

        - ``"warmup"``: Warmup data (B, warmup_steps, D)
        - ``"train"``: Training input (B, train_steps, D)
        - ``"target"``: Training targets (B, train_steps, D)
        - ``"f_warmup"``: Forecast warmup (B, warmup_steps, D)
        - ``"val"``: Validation data (B, val_steps, D)

    n_trials : int
        Total number of trials to run.
    loss : str or LossProtocol, default="efh"
        Loss function to optimize. Can be:

        - ``"efh"``: Expected Forecast Horizon (default, recommended)
        - ``"horizon"``: Forecast Horizon Loss
        - ``"lyap"``: Lyapunov-weighted Loss
        - ``"standard"``: Standard Loss
        - ``"discounted"``: Discounted RMSE
        - A custom callable following LossProtocol

    loss_params : dict, optional
        Additional keyword arguments for the loss function.
    targets_key : str, default="output"
        Name of the readout layer for training targets.
    drivers_keys : list[str], optional
        List of driver input keys in data dict for input-driven models.
    monitor_losses : list[str | LossProtocol], optional
        Additional loss functions to compute and log (but not optimize on).
        Can be loss names (e.g., ``"standard"``, ``"lyap"``) or callables.
        Results are stored as trial user attributes with prefix ``monitor_``.
    monitor_params : dict[str, dict[str, Any]], optional
        Keyword arguments for each monitor loss. Keys are loss function names
        (e.g., ``"lyapunov_weighted_loss"``), values are kwargs dicts.
    study_name : str, optional
        Name for the study. If ``None``, auto-generated from *model_creator*.
    storage : str, optional
        Storage path or URL. Behaviour depends on the value:

        - ``None``: In-memory (single worker) or temp journal file (multi-worker).
        - ``"study.log"``: Journal file storage (recommended for multi-worker).
        - ``"study.db"`` or ``"sqlite:///study.db"``: SQLite with WAL mode.

    sampler : BaseSampler, optional
        Optuna sampler. Defaults to ``TPESampler`` with ``multivariate=True``.
    seed : int, optional
        Random seed for reproducibility. Seeds both the Optuna sampler and
        per-trial PyTorch / numpy random state (``seed + trial.number``).
    device : str or torch.device, optional
        Device to place models and data on (e.g., ``"cuda"`` or
        ``torch.device("cpu")``). If ``None``, uses default device.
    n_workers : int, default=1
        Number of parallel workers. When ``> 1``, uses real OS processes
        (multiprocessing) that coordinate via shared file storage.
    verbosity : int, default=1
        Logging verbosity: ``0`` = silent, ``1`` = normal, ``2`` = verbose.
    catch_exceptions : bool, default=True
        If ``True``, catch exceptions and return *penalty_value*.
    penalty_value : float, default=1e10
        Value returned for failed trials.
    clip_value : float, optional
        Upper bound for the objective value.  When set and the raw loss
        exceeds this threshold, the value returned to Optuna is clamped to
        *clip_value* (or the trial is pruned, see *prune_on_clip*).  The
        unclipped value is always stored as the ``"raw_loss"`` trial user
        attribute.  When ``None`` (default), no clipping is applied and
        ``raw_loss == loss``.
    prune_on_clip : bool, default=False
        If ``True`` **and** *clip_value* is set, trials whose raw loss exceeds
        *clip_value* are pruned instead of returning the clipped value.

    Returns
    -------
    optuna.Study
        The completed study with all trial results.

    Raises
    ------
    ValueError
        If *n_trials* is not positive.
    TypeError
        If *model_creator*, *search_space*, or *data_loader* are not callable.

    Examples
    --------
    Basic usage:

    >>> from resdag.hpo import run_hpo
    >>> from resdag.models import ott_esn
    >>>
    >>> def model_creator(reservoir_size, spectral_radius):
    ...     return ott_esn(
    ...         reservoir_size=reservoir_size,
    ...         feedback_size=3,
    ...         output_size=3,
    ...         spectral_radius=spectral_radius,
    ...     )
    >>>
    >>> def search_space(trial):
    ...     return {
    ...         "reservoir_size": trial.suggest_int("reservoir_size", 100, 500, step=50),
    ...         "spectral_radius": trial.suggest_float("spectral_radius", 0.5, 1.5),
    ...     }
    >>>
    >>> def data_loader(trial):
    ...     # Load your data here
    ...     return {
    ...         "warmup": warmup, "train": train, "target": target,
    ...         "f_warmup": f_warmup, "val": val,
    ...     }
    >>>
    >>> study = run_hpo(
    ...     model_creator=model_creator,
    ...     search_space=search_space,
    ...     data_loader=data_loader,
    ...     n_trials=50,
    ...     loss="efh",
    ... )
    >>> print(f"Best params: {study.best_params}")
    >>> print(f"Best value: {study.best_value}")

    Parallel with 4 workers:

    >>> study = run_hpo(
    ...     model_creator=model_creator,
    ...     search_space=search_space,
    ...     data_loader=data_loader,
    ...     n_trials=100,
    ...     n_workers=4,
    ...     storage="study.log",
    ... )

    See Also
    --------
    LOSSES : Available loss functions.
    get_study_summary : Generate study summary.
    ESNTrainer : Training interface.
    """
    # ── Configure logging ────────────────────────────────────────────
    if verbosity == 0:
        logging.basicConfig(level=logging.ERROR)
        optuna.logging.set_verbosity(optuna.logging.ERROR)
    elif verbosity == 1:
        logging.basicConfig(level=logging.INFO)
        optuna.logging.set_verbosity(optuna.logging.INFO)
    else:
        logging.basicConfig(level=logging.DEBUG)
        optuna.logging.set_verbosity(optuna.logging.DEBUG)

    # ── Validate inputs ──────────────────────────────────────────────
    if n_trials <= 0:
        raise ValueError(f"n_trials must be positive, got {n_trials}")
    if not callable(model_creator):
        raise TypeError("model_creator must be callable")
    if not callable(search_space):
        raise TypeError("search_space must be callable")
    if not callable(data_loader):
        raise TypeError("data_loader must be callable")

    # ── Resolve device ───────────────────────────────────────────────
    if device is not None and not isinstance(device, torch.device):
        device = torch.device(device)

    # ── Resolve storage ──────────────────────────────────────────────
    resolved_storage = resolve_storage(storage, n_workers)

    # ── Resolve loss function ────────────────────────────────────────
    loss_params = loss_params or {}
    base_loss = get_loss(loss)
    resolved_loss = partial(base_loss, **loss_params) if loss_params else base_loss

    loss_name = loss if isinstance(loss, str) else getattr(loss, "__name__", "custom")
    logger.info(f"Using loss function: {loss_name}")
    if loss_params:
        logger.info(f"Loss parameters: {loss_params}")

    # ── Resolve monitor losses ───────────────────────────────────────
    resolved_monitor_losses = None
    resolved_monitor_params = monitor_params
    if monitor_losses:
        resolved_monitor_losses = [get_loss(m) if isinstance(m, str) else m for m in monitor_losses]
        monitor_names = [
            m if isinstance(m, str) else getattr(m, "__name__", "custom") for m in monitor_losses
        ]
        logger.info(f"Monitoring additional losses: {monitor_names}")

        # Remap monitor_params keys from registry keys (e.g. "efh") to
        # function names (e.g. "expected_forecast_horizon") so the objective
        # can look them up by monitor_fn.__name__.
        if monitor_params:
            resolved_monitor_params = dict(monitor_params)
            for user_key, fn in zip(monitor_losses, resolved_monitor_losses):
                if isinstance(user_key, str):
                    func_name = getattr(fn, "__name__", user_key)
                    if user_key != func_name and user_key in resolved_monitor_params:
                        resolved_monitor_params[func_name] = resolved_monitor_params.pop(user_key)

    # ── Configure sampler ────────────────────────────────────────────
    if sampler is None:
        sampler = TPESampler(
            multivariate=True,
            warn_independent_sampling=False,
            seed=seed,
        )
        logger.info("Using TPESampler with multivariate optimization")

    # ── Create or load study ─────────────────────────────────────────
    if study_name is None:
        study_name = make_study_name(model_creator)
        logger.info(f"Auto-generated study name: {study_name}")

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=resolved_storage,
        load_if_exists=True,
        sampler=sampler,
    )

    completed_trials = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )
    if completed_trials > 0:
        logger.info(f"Loaded existing study with {completed_trials} completed trials")
        logger.info(f"Best value so far: {study.best_value:.6f}")

    # ── Build objective function ─────────────────────────────────────
    objective = build_objective(
        model_creator=model_creator,
        search_space=search_space,
        data_loader=data_loader,
        loss_fn=resolved_loss,
        targets_key=targets_key,
        drivers_keys=drivers_keys,
        catch_exceptions=catch_exceptions,
        penalty_value=penalty_value,
        monitor_losses=resolved_monitor_losses,
        monitor_params=resolved_monitor_params,
        device=device,
        seed=seed,
        clip_value=clip_value,
        prune_on_clip=prune_on_clip,
    )

    # ── Run optimization ─────────────────────────────────────────────
    remaining = max(0, n_trials - completed_trials)
    if remaining > 0:
        logger.info(f"Starting optimization: {remaining} trials remaining")

        if n_workers > 1:
            study, resolved_storage = _dispatch_multiprocess(
                study=study,
                resolved_storage=resolved_storage,
                storage=storage,
                study_name=study_name,
                objective=objective,
                n_trials=n_trials,
                remaining=remaining,
                completed_trials=completed_trials,
                n_workers=n_workers,
                seed=seed,
                verbosity=verbosity,
            )
        else:
            run_single(
                study=study,
                objective=objective,
                remaining=remaining,
                n_workers=1,
                verbosity=verbosity,
            )

        # Final summary
        done = len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        )
        logger.info(f"Optimization completed: {len(study.trials)} total trials")
        if done > 0:
            logger.info(f"Best value: {study.best_value:.6f}")
            logger.info(f"Best parameters: {study.best_params}")
    else:
        logger.info(f"All {n_trials} trials already completed")

    return study


def _dispatch_multiprocess(
    study: optuna.Study,
    resolved_storage: optuna.storages.BaseStorage | None,
    storage: str | None,
    study_name: str,
    objective: Callable[[optuna.Trial], float],
    n_trials: int,
    remaining: int,
    completed_trials: int,
    n_workers: int,
    seed: int | None,
    verbosity: int,
) -> tuple[optuna.Study, optuna.storages.BaseStorage | None]:
    """Prepare the environment and delegate to :func:`run_multiprocess`.

    Throttles BLAS/OpenMP threads, releases parent storage references to avoid
    fork-inherited file locks, enables WAL mode for SQLite, runs the workers,
    and reopens the study for the caller.

    Parameters
    ----------
    study : optuna.Study
        The study created in the parent process (will be deleted before fork).
    resolved_storage : BaseStorage or None
        The storage object created in the parent (will be disposed).
    storage : str or None
        Original user-provided storage specifier.
    study_name : str
        Study name for reconnection after workers finish.
    objective : Callable[[Trial], float]
        Objective function to pass to workers.
    n_trials : int
        Total target completed trials.
    remaining : int
        Number of new trials to run.
    completed_trials : int
        Trials already completed before this call.
    n_workers : int
        Number of worker processes.
    seed : int or None
        Base random seed.
    verbosity : int
        Logging verbosity.

    Returns
    -------
    tuple[optuna.Study, BaseStorage or None]
        Reopened study and storage objects after workers finish.
    """
    # Throttle BLAS/OpenMP threads BEFORE fork so children inherit it.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    # Determine the raw storage path for workers to reconnect.
    storage_path = storage  # user-provided path
    if storage_path is None:
        # Auto-created temp journal — extract the file path
        journal_backend = resolved_storage._backend  # type: ignore[union-attr]
        storage_path = journal_backend._file_path

    # Force schema creation + commit before releasing
    _ = study.get_trials(deepcopy=False)

    # Dispose engine cleanly if using SQLite
    if isinstance(resolved_storage, optuna.storages.RDBStorage):
        resolved_storage.engine.dispose()

    # Drop the parent's storage references before forking
    del study, resolved_storage

    # Enable WAL mode for SQLite (better concurrent reads/writes)
    enable_sqlite_wal(storage_path)

    run_multiprocess(
        study_name=study_name,
        storage=storage_path,
        objective=objective,
        n_trials=n_trials,
        remaining=remaining,
        completed_trials=completed_trials,
        n_workers=n_workers,
        seed=seed,
        verbosity=verbosity,
    )

    # Reopen storage to build the return study object
    new_storage = resolve_storage(storage_path, n_workers=1)
    new_study = optuna.load_study(
        study_name=study_name,
        storage=new_storage,
    )
    return new_study, new_storage
