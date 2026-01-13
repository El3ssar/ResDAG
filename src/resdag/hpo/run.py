"""Main HPO function for reservoir computing models.

This module provides the high-level interface for running hyperparameter
optimization studies on ESN models using Optuna.
"""

import logging
from functools import partial
from typing import Any, Callable

import optuna
from optuna.samplers import TPESampler

from ..composition.symbolic import ESNModel
from .losses import LossProtocol, get_loss
from .objective import build_objective
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
    n_workers: int = 1,
    verbosity: int = 1,
    catch_exceptions: bool = True,
    penalty_value: float = 1e10,
) -> optuna.Study:
    """Run an Optuna hyperparameter optimization study for ESN models.

    This function provides a complete HPO pipeline that handles model creation,
    training, forecasting, and evaluation with robust error handling.

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
        Can be loss names (e.g., "standard", "lyap") or callables.
        Results are stored as trial user attributes with prefix "monitor_".
    monitor_params : dict[str, dict[str, Any]], optional
        Keyword arguments for each monitor loss. Keys are loss function names
        (e.g., "expected_forecast_horizon_loss"), values are kwargs dicts.
        Example: ``{"lyapunov_weighted_loss": {"lyapunov_time": 50}}``.
    study_name : str, optional
        Name for the study. If None, auto-generated from model_creator.
    storage : str, optional
        Optuna storage URL (e.g., "sqlite:///study.db").
    sampler : BaseSampler, optional
        Optuna sampler. Defaults to TPESampler with multivariate=True.
    seed : int, optional
        Random seed for reproducibility.
    n_workers : int, default=1
        Number of parallel workers (uses Optuna's n_jobs).
    verbosity : int, default=1
        Logging verbosity: 0=silent, 1=normal, 2=verbose.
    catch_exceptions : bool, default=True
        If True, catch exceptions and return penalty_value.
    penalty_value : float, default=1e10
        Value returned for failed trials.

    Returns
    -------
    optuna.Study
        The completed study with all trial results.

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

    With persistence:

    >>> study = run_hpo(
    ...     model_creator=model_creator,
    ...     search_space=search_space,
    ...     data_loader=data_loader,
    ...     n_trials=100,
    ...     storage="sqlite:///my_study.db",
    ...     study_name="lorenz_optimization",
    ... )

    See Also
    --------
    LOSSES : Available loss functions
    get_study_summary : Generate study summary
    ESNTrainer : Training interface
    """
    # Configure logging
    if verbosity == 0:
        logging.basicConfig(level=logging.ERROR)
        optuna.logging.set_verbosity(optuna.logging.ERROR)
    elif verbosity == 1:
        logging.basicConfig(level=logging.INFO)
        optuna.logging.set_verbosity(optuna.logging.INFO)
    else:
        logging.basicConfig(level=logging.DEBUG)
        optuna.logging.set_verbosity(optuna.logging.DEBUG)

    # Validate inputs
    if n_trials <= 0:
        raise ValueError(f"n_trials must be positive, got {n_trials}")
    if not callable(model_creator):
        raise TypeError("model_creator must be callable")
    if not callable(search_space):
        raise TypeError("search_space must be callable")
    if not callable(data_loader):
        raise TypeError("data_loader must be callable")

    # Resolve loss function
    loss_params = loss_params or {}
    base_loss = get_loss(loss)
    resolved_loss = partial(base_loss, **loss_params) if loss_params else base_loss

    loss_name = loss if isinstance(loss, str) else getattr(loss, "__name__", "custom")
    logger.info(f"Using loss function: {loss_name}")
    if loss_params:
        logger.info(f"Loss parameters: {loss_params}")

    # Resolve monitor losses
    resolved_monitor_losses = None
    if monitor_losses:
        resolved_monitor_losses = [get_loss(m) if isinstance(m, str) else m for m in monitor_losses]
        monitor_names = [
            m if isinstance(m, str) else getattr(m, "__name__", "custom") for m in monitor_losses
        ]
        logger.info(f"Monitoring additional losses: {monitor_names}")

    # Configure sampler
    if sampler is None:
        sampler = TPESampler(
            multivariate=True,
            warn_independent_sampling=False,
            seed=seed,
        )
        logger.info("Using TPESampler with multivariate optimization")

    # Generate study name
    if study_name is None:
        study_name = make_study_name(model_creator)
        logger.info(f"Auto-generated study name: {study_name}")

    # Create or load study
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
    )

    # Check for existing trials
    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if completed_trials > 0:
        logger.info(f"Loaded existing study with {completed_trials} completed trials")
        logger.info(f"Best value so far: {study.best_value:.6f}")

    # Build objective function
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
        monitor_params=monitor_params,
    )

    # Run optimization
    remaining = max(0, n_trials - completed_trials)
    if remaining > 0:
        logger.info(f"Starting optimization: {remaining} trials remaining")

        try:
            study.optimize(
                objective,
                n_trials=remaining,
                n_jobs=n_workers,
                show_progress_bar=verbosity > 0 and n_workers == 1,
            )
        except KeyboardInterrupt:
            logger.warning("Optimization interrupted by user")
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise

        # Final summary
        done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        logger.info(f"Optimization completed: {len(study.trials)} total trials")
        if done > 0:
            logger.info(f"Best value: {study.best_value:.6f}")
            logger.info(f"Best parameters: {study.best_params}")
    else:
        logger.info(f"All {n_trials} trials already completed")

    return study
