"""Construct the numerical runner for raw Optuna model studies.

``build_objective`` returns a callable ``TrialRunner`` for direct Optuna use.
``run_hpo`` splits its preparation, numerical evaluation and finalization so
managed joblib workers never receive a Trial or storage object. Forecast chunks
reach growing checkpoints before reporting, allowing pruning to stop generation.
Raw multi-output graphs are scored on their first feedback output.
"""

from typing import Any, Callable, Literal

import optuna
import torch

from resdag.core import ESNModel

from .losses import LossProtocol
from .trial_runner import TrialCallback, TrialRunner, _cleanup, _validate_data_keys

__all__ = ["build_objective"]

# Re-exported for backward compatibility with callers/tests that imported these
# private helpers from ``resdag.hpo.objective`` before the TrialRunner refactor.
_cleanup = _cleanup
_validate_data_keys = _validate_data_keys


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
    n_checkpoints: int = 5,
    torch_scoring: bool = True,
    trial_callbacks: list[TrialCallback] | None = None,
    error_policy: Literal["fail", "penalty"] = "penalty",
) -> TrialRunner:
    """Build an Optuna objective from user-defined callbacks.

    Thin wrapper around :class:`resdag.hpo.trial_runner.TrialRunner`.  The returned
    object is callable: invoke it with an :class:`optuna.Trial` to return
    the (possibly clipped) loss. ``run_hpo`` uses its split lifecycle with
    managed joblib serialization and parent-owned checkpoint pruning.

    Parameters
    ----------
    model_creator : Callable[..., ESNModel]
        Function that creates a fresh model given hyperparameters.  Must accept
        all hyperparameters from ``search_space`` as keyword arguments.
    search_space : Callable[[Trial], dict[str, Any]]
        Function that defines the hyperparameter search space using Optuna's
        ``trial.suggest_*`` methods.  Returns a dictionary of hyperparameters.
    data_loader : Callable[[Trial], dict[str, Tensor]]
        Function that loads and returns training/validation data.  Must return a
        dictionary with keys: ``"warmup"``, ``"train"``, ``"target"``,
        ``"f_warmup"``, ``"val"``.  Optionally include driver inputs with keys
        like ``"warmup_driver"``, ``"train_driver"``.
    loss_fn : LossProtocol
        Loss function to evaluate model performance.
    targets_key : str, default="output"
        Name of the readout layer target in the targets dict.
    drivers_keys : list[str], optional
        List of driver input keys in the data dict (e.g. ``["driver1"]``).  If
        provided, these are passed as additional inputs during
        training/forecasting.
    horizon_key : str, optional
        Key in the data dict specifying the forecast horizon.  If ``None``, uses
        ``val.shape[1]``.
    catch_exceptions : bool, default=True
        If ``True``, catch exceptions and return ``penalty_value`` instead of
        raising under the legacy penalty policy. Pruning is always re-raised.
    error_policy : {"penalty", "fail"}, default="penalty"
        ``"fail"`` raises ``TrialEvaluationError`` for an invalid evaluation;
        pass that exception in Optuna's ``catch`` argument to continue. The
        high-level ``run_hpo`` configures this policy and its backend for you.
    penalty_value : float, default=1e10
        Value to return when a trial fails.
    monitor_losses : list[LossProtocol], optional
        Additional loss functions to compute and log (but not optimize on).
        These are logged as user attributes on the trial.
    monitor_params : dict[str, dict[str, Any]], optional
        Keyword arguments for each monitor loss.  Keys are loss function names,
        values are dicts of kwargs, e.g. ``{"efh": {"threshold": 0.3}}``.
    device : torch.device, optional
        Device to place model and data on (e.g. ``torch.device("cuda")``).  If
        ``None``, uses the default device from model/data.
    seed : int, optional
        Base seed for per-trial reproducibility.  Each trial uses
        ``seed + trial.number`` to seed PyTorch, NumPy, and Python's
        :mod:`random` module, and the same per-trial seed is threaded into
        ``model_creator`` (when it accepts a ``seed`` keyword) so the reservoir
        topology and input/feedback initializers also reproduce.
    clip_value : float, optional
        Upper bound for the objective value.  When set and the raw loss exceeds
        this threshold, the value returned to Optuna is clamped to *clip_value*
        (or the trial is pruned, see *prune_on_clip*).  The unclipped value is
        always stored as the ``"raw_loss"`` user attribute.  When ``None``
        (default), no clipping is applied and ``raw_loss == loss``.
    prune_on_clip : bool, default=False
        If ``True`` **and** *clip_value* is set, trials whose raw loss exceeds
        *clip_value* are pruned (via ``optuna.TrialPruned``) instead of
        returning the clipped value.  Pruned trials do not count towards the
        study's completed trials.
    n_checkpoints : int, default=5
        Number of growing horizon checkpoints at which to report intermediate
        losses and check for pruning.  The single forecast is sliced to each
        checkpoint (no repeated forecasts).  Values ``<= 1`` disable
        intermediate reporting (a single report at the full horizon, matching
        the legacy single-shot behaviour).
    torch_scoring : bool, default=True
        If ``True``, score on the forecast's device with ``torch.Tensor`` inputs
        until the final scalar, falling back to NumPy when the loss does not
        support tensors.  Set ``False`` to always score in NumPy.
    trial_callbacks : list[TrialCallback], optional
        Per-trial callbacks invoked after each successful evaluation.  Each
        receives ``(trial, context)`` and may call ``trial.report`` /
        ``trial.should_prune`` to act as a pruning signal.  Callback exceptions
        are logged and swallowed — a callback can never fail the trial.

    Returns
    -------
    TrialRunner
        A picklable callable objective for Optuna to optimize.

    Example
    -------
    >>> objective = build_objective(
    ...     model_creator=my_model_creator,
    ...     search_space=my_search_space,
    ...     data_loader=my_data_loader,
    ...     loss_fn=get_loss("efh"),
    ...     monitor_losses=[get_loss("standard"), get_loss("lyapunov")],
    ...     monitor_params={"lyapunov_weighted": {"lyapunov_t": 50}},
    ... )
    >>> study = optuna.create_study()
    >>> study.optimize(objective, n_trials=100)
    """
    return TrialRunner(
        model_creator=model_creator,
        search_space=search_space,
        data_loader=data_loader,
        loss_fn=loss_fn,
        targets_key=targets_key,
        drivers_keys=drivers_keys,
        horizon_key=horizon_key,
        catch_exceptions=catch_exceptions,
        penalty_value=penalty_value,
        monitor_losses=monitor_losses,
        monitor_params=monitor_params,
        device=device,
        seed=seed,
        clip_value=clip_value,
        prune_on_clip=prune_on_clip,
        n_checkpoints=n_checkpoints,
        torch_scoring=torch_scoring,
        trial_callbacks=trial_callbacks,
        error_policy=error_policy,
    )
