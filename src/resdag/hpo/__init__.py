"""
Hyperparameter Optimization
===========================

This module provides Optuna-based hyperparameter optimization for ESN models.
It supports multiple loss functions specialized for time series forecasting
and chaotic systems.

Installation
------------
This module requires the optional ``optuna`` dependency::

    pip install resdag[hpo]

or::

    pip install optuna

Functions
---------
run_hpo
    Run hyperparameter optimization study.
get_loss
    Get a loss function by name.
get_study_summary
    Generate summary of completed study.
resolve_pruner
    Resolve a pruner key/instance to an :class:`optuna.pruners.BasePruner`.

Pruners
-------
Pass ``pruner=`` to :func:`run_hpo` to enable Optuna early stopping, fed by the
per-trial intermediate forecast-horizon reports. Registry keys (``PRUNERS``):

- ``"asha"``: ASHA / SuccessiveHalvingPruner
- ``"hyperband"``: Hyperband pruner
- ``"median"``: Median pruner (recommended starting point)
- ``"threshold"``: Threshold pruner
- ``"none"``: NopPruner (no pruning, the default)

Loss Functions
--------------
The following loss functions are available:

- ``"efh"``: Expected Forecast Horizon (default, recommended for chaotic systems)
- ``"forecast_horizon"``: Forecast Horizon Loss (contiguous valid steps)
- ``"lyapunov"``: Lyapunov-weighted Loss (exponential decay)
- ``"standard"``: Standard Loss (mean geometric mean error)
- ``"soft_horizon"``: Soft valid horizon (cumulative survival probability)

Examples
--------
Basic HPO workflow:

>>> from resdag.hpo import run_hpo, get_study_summary
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
... )
>>> print(get_study_summary(study))

See Also
--------
resdag.training.ESNTrainer : Training interface.
resdag.utils.data : Data loading utilities.
"""

from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Any as _Any

# Loss functions don't require optuna - always available
from .losses import (
    LOSSES,
    LossProtocol,
    attractor_dimension_error,
    expected_forecast_horizon,
    forecast_horizon,
    get_loss,
    lyapunov_weighted,
    soft_valid_horizon,
    standard_loss,
)

# Check if optuna is available
try:
    import optuna as _optuna

    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False
    _optuna = None  # type: ignore


def _require_optuna() -> None:
    """Raise ImportError with helpful message if optuna is not installed."""
    if not _OPTUNA_AVAILABLE:
        raise ImportError(
            "The 'optuna' package is required for hyperparameter optimization. "
            "Install it with: pip install resdag[hpo] or pip install optuna"
        )


# Lazy imports for optuna-dependent functions
if _TYPE_CHECKING:
    from .pruners import PRUNERS as PRUNERS
    from .pruners import resolve_pruner as resolve_pruner
    from .resources import MemoryBudget as MemoryBudget
    from .resources import MemoryContext as MemoryContext
    from .resources import MemoryEstimate as MemoryEstimate
    from .resources import Screening as Screening
    from .run import run_hpo as run_hpo
    from .transfer import export_best_config as export_best_config
    from .transfer import transfer_trials as transfer_trials
    from .trial_runner import TrialEvaluationError as TrialEvaluationError
    from .tuning import PrefixScoreContext as PrefixScoreContext
    from .tuning import ScoreContext as ScoreContext
    from .tuning import TuningReport as TuningReport
    from .tuning import tune_model as tune_model
    from .utils import get_best_params as get_best_params
    from .utils import get_study_summary as get_study_summary
    from .utils import make_study_name as make_study_name


def __getattr__(name: str) -> _Any:
    """Lazy import for optuna-dependent functions."""
    if name in ("Screening", "MemoryBudget", "MemoryContext", "MemoryEstimate"):
        from . import resources

        return getattr(resources, name)

    if name in ("tune_model", "TuningReport", "ScoreContext", "PrefixScoreContext"):
        _require_optuna()
        from . import tuning

        return getattr(tuning, name)

    if name == "TrialEvaluationError":
        _require_optuna()
        from .trial_runner import TrialEvaluationError

        return TrialEvaluationError

    if name == "run_hpo":
        _require_optuna()
        from .run import run_hpo

        return run_hpo

    if name == "get_study_summary":
        _require_optuna()
        from .utils import get_study_summary

        return get_study_summary

    if name == "make_study_name":
        _require_optuna()
        from .utils import make_study_name

        return make_study_name

    if name == "get_best_params":
        _require_optuna()
        from .utils import get_best_params

        return get_best_params

    if name == "export_best_config":
        _require_optuna()
        from .transfer import export_best_config

        return export_best_config

    if name == "transfer_trials":
        _require_optuna()
        from .transfer import transfer_trials

        return transfer_trials

    if name == "resolve_pruner":
        _require_optuna()
        from .pruners import resolve_pruner

        return resolve_pruner

    if name == "PRUNERS":
        _require_optuna()
        from .pruners import PRUNERS

        return PRUNERS

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Main function
    "run_hpo",
    "tune_model",
    "TuningReport",
    "ScoreContext",
    "PrefixScoreContext",
    "Screening",
    "MemoryBudget",
    "MemoryEstimate",
    "MemoryContext",
    "TrialEvaluationError",
    # Loss functions (always available)
    "LOSSES",
    "LossProtocol",
    "get_loss",
    "expected_forecast_horizon",
    "forecast_horizon",
    "lyapunov_weighted",
    "soft_valid_horizon",
    "standard_loss",
    "attractor_dimension_error",
    # Utilities (require optuna)
    "get_best_params",
    "get_study_summary",
    "make_study_name",
    # Warm-start / transfer (require optuna)
    "export_best_config",
    "transfer_trials",
    # Pruners (require optuna)
    "PRUNERS",
    "resolve_pruner",
]


def __dir__() -> list[str]:
    """Restrict ``dir()`` / tab-completion to the public API (:pep:`562`)."""
    return list(__all__)
