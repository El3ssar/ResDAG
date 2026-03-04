"""Objective function builder for Optuna optimization.

This module provides the core machinery for creating Optuna objective functions
from user-defined callbacks.  It handles the full per-trial lifecycle: model
creation, training, forecasting, evaluation, and optional monitor-loss logging.

See Also
--------
resdag.hpo.run : High-level orchestrator that calls :func:`build_objective`.
resdag.hpo.losses : Loss functions available for evaluation.
"""

import gc
import logging
from typing import Any, Callable

import numpy as np
import optuna
import torch

from resdag.composition import ESNModel
from resdag.training import ESNTrainer

from .losses import LossProtocol

__all__ = ["build_objective"]

logger = logging.getLogger(__name__)


def _cleanup() -> None:
    """Reclaim memory between trials.

    Forces Python garbage collection and, when CUDA is available, empties
    the GPU memory cache to avoid out-of-memory errors across long studies.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_objective(
    model_creator: Callable[..., ESNModel],
    search_space: Callable[[optuna.Trial], dict[str, Any]],
    data_loader: Callable[[optuna.Trial], dict[str, torch.Tensor]],
    loss_fn: LossProtocol,
    targets_key: str = "output",
    drivers_keys: list[str] | None = None,
    horizon_key: str | None = None,
    catch_exceptions: bool = True,
    penalty_value: float = 1e10,
    monitor_losses: list[LossProtocol] | None = None,
    monitor_params: dict[str, dict[str, Any]] | None = None,
    device: torch.device | None = None,
    seed: int | None = None,
    clip_value: float | None = None,
    prune_on_clip: bool = False,
) -> Callable[[optuna.Trial], float]:
    """Build an Optuna objective function from user-defined callbacks.

    Creates a closure that wraps model creation, training, and evaluation
    into a single objective function that Optuna can optimize.

    Parameters
    ----------
    model_creator : Callable[..., ESNModel]
        Function that creates a fresh model given hyperparameters.
        Must accept all hyperparameters from ``search_space`` as keyword arguments.
    search_space : Callable[[Trial], dict[str, Any]]
        Function that defines the hyperparameter search space using Optuna's
        ``trial.suggest_*`` methods. Returns a dictionary of hyperparameters.
    data_loader : Callable[[Trial], dict[str, Tensor]]
        Function that loads and returns training/validation data. Must return
        a dictionary with keys: "warmup", "train", "target", "f_warmup", "val".
        Optionally include driver inputs with keys like "warmup_driver", "train_driver".
    loss_fn : LossProtocol
        Loss function to evaluate model performance.
    targets_key : str, default="output"
        Name of the readout layer target in the targets dict.
    drivers_keys : list[str], optional
        List of driver input keys in data dict (e.g., ["driver1", "driver2"]).
        If provided, these are passed as additional inputs during training/forecasting.
    horizon_key : str, optional
        Key in data dict specifying forecast horizon. If None, uses val.shape[1].
    catch_exceptions : bool, default=True
        If True, catch exceptions and return penalty_value instead of raising.
    penalty_value : float, default=1e10
        Value to return when a trial fails.
    monitor_losses : list[LossProtocol], optional
        Additional loss functions to compute and log (but not optimize on).
        These are logged as user attributes on the trial.
    monitor_params : dict[str, dict[str, Any]], optional
        Keyword arguments for each monitor loss. Keys are loss function names,
        values are dicts of kwargs. E.g., ``{"efh": {"threshold": 0.3}}``.
    device : torch.device, optional
        Device to place model and data on (e.g., ``torch.device("cuda")``).
        If None, uses default device from model/data.
    seed : int, optional
        Base seed for per-trial reproducibility. Each trial uses
        ``seed + trial.number`` to seed PyTorch and numpy.
    clip_value : float, optional
        Upper bound for the objective value.  When set and the raw loss
        exceeds this threshold, the value returned to Optuna is clamped to
        *clip_value* (or the trial is pruned, see *prune_on_clip*).  The
        unclipped value is always stored as the ``"raw_loss"`` user attribute.
        When ``None`` (default), no clipping is applied and
        ``raw_loss == loss``.
    prune_on_clip : bool, default=False
        If ``True`` **and** *clip_value* is set, trials whose raw loss exceeds
        *clip_value* are pruned (via ``optuna.TrialPruned``) instead of
        returning the clipped value.  Pruned trials do not count towards the
        study's completed trials.

    Returns
    -------
    Callable[[Trial], float]
        The objective function for Optuna to optimize.

    Example
    -------
    >>> objective = build_objective(
    ...     model_creator=my_model_creator,
    ...     search_space=my_search_space,
    ...     data_loader=my_data_loader,
    ...     loss_fn=get_loss("efh"),
    ...     monitor_losses=[get_loss("standard"), get_loss("lyap")],
    ...     monitor_params={"lyapunov_weighted_loss": {"lyapunov_time": 50}},
    ... )
    >>> study = optuna.create_study()
    >>> study.optimize(objective, n_trials=100)
    """

    def objective(trial: optuna.Trial) -> float:
        try:
            # 0. Per-trial reproducible seeding
            if seed is not None:
                trial_seed = seed + trial.number
                torch.manual_seed(trial_seed)
                np.random.seed(trial_seed % (2**32))

            # 1. Get hyperparameters from search space
            params = search_space(trial)

            # 2. Load data
            data = data_loader(trial)
            _validate_data_keys(data)

            # Move data to device
            if device is not None:
                data = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()
                }

            # 3. Create fresh model
            model = model_creator(**params)
            if device is not None:
                model = model.to(device)

            # 4. Prepare inputs
            warmup_inputs = (data["warmup"],)
            train_inputs = (data["train"],)

            if drivers_keys:
                for key in drivers_keys:
                    warmup_key = f"warmup_{key}"
                    train_key = f"train_{key}"
                    if warmup_key in data and train_key in data:
                        warmup_inputs = warmup_inputs + (data[warmup_key],)
                        train_inputs = train_inputs + (data[train_key],)

            # 5. Train with ESNTrainer
            trainer = ESNTrainer(model)
            trainer.fit(
                warmup_inputs=warmup_inputs,
                train_inputs=train_inputs,
                targets={targets_key: data["target"]},
            )

            # 6. Forecast
            horizon = (
                data[horizon_key] if horizon_key and horizon_key in data else data["val"].shape[1]
            )

            # Prepare forecast warmup inputs (feedback + drivers)
            f_warmup_inputs = (data["f_warmup"],)
            forecast_drivers_list = []

            if drivers_keys:
                for key in drivers_keys:
                    # Forecast warmup drivers (for warmup phase)
                    f_warmup_key = f"f_warmup_{key}"
                    if f_warmup_key in data:
                        f_warmup_inputs = f_warmup_inputs + (data[f_warmup_key],)

                    # Forecast drivers (for autoregressive phase)
                    forecast_key = f"forecast_{key}"
                    if forecast_key in data:
                        forecast_drivers_list.append(data[forecast_key])

            # Run forecast
            preds = model.forecast(
                *f_warmup_inputs,
                horizon=horizon,
                forecast_drivers=tuple(forecast_drivers_list),
            )

            # 7. Compute loss
            val = data["val"]
            timesteps = min(preds.shape[1], val.shape[1])

            # Convert to numpy for loss function
            preds_np = preds[:, :timesteps, :].detach().cpu().numpy()
            val_np = val[:, :timesteps, :].detach().cpu().numpy()

            raw_loss = float(loss_fn(val_np, preds_np))

            # Always store the unclipped value
            trial.set_user_attr("raw_loss", raw_loss)

            # Apply clipping
            if clip_value is not None and raw_loss > clip_value:
                if prune_on_clip:
                    raise optuna.TrialPruned(
                        f"raw_loss {raw_loss:.6f} exceeds clip_value {clip_value}"
                    )
                loss = clip_value
            else:
                loss = raw_loss

            trial.set_user_attr("loss", loss)

            # Compute and log monitor losses
            if monitor_losses:
                monitor_params_resolved = monitor_params or {}
                for monitor_fn in monitor_losses:
                    # Get loss name for logging and params lookup
                    loss_name = getattr(monitor_fn, "__name__", str(monitor_fn))
                    # Get kwargs for this monitor loss
                    kwargs = monitor_params_resolved.get(loss_name, {})
                    try:
                        monitor_value = float(monitor_fn(val_np, preds_np, **kwargs))
                        trial.set_user_attr(f"monitor_{loss_name}", monitor_value)
                    except Exception as e:
                        logger.warning(f"Monitor loss {loss_name} failed: {e}")
                        trial.set_user_attr(f"monitor_{loss_name}", None)

            return loss

        except Exception as e:
            if catch_exceptions:
                logger.warning(f"Trial {trial.number} failed: {e}")
                trial.set_user_attr("error", str(e))
                return penalty_value
            raise

        finally:
            _cleanup()

    return objective


def _validate_data_keys(data: dict[str, Any]) -> None:
    """Validate that the data dictionary contains all required keys.

    Parameters
    ----------
    data : dict[str, Any]
        Dictionary returned by the user-provided ``data_loader`` callback.

    Raises
    ------
    KeyError
        If one or more of the required keys (``"warmup"``, ``"train"``,
        ``"target"``, ``"f_warmup"``, ``"val"``) are missing.
    """
    required = ["warmup", "train", "target", "f_warmup", "val"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(
            f"Missing required keys in data dictionary: {missing}. Required keys: {required}"
        )
