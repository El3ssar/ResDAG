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

Loss Functions
--------------
The following loss functions are available:

- ``"efh"``: Expected Forecast Horizon (default, recommended for chaotic systems)
- ``"horizon"``: Forecast Horizon Loss (contiguous valid steps)
- ``"lyap"``: Lyapunov-weighted Loss (exponential decay)
- ``"standard"``: Standard Loss (mean geometric mean error)
- ``"discounted"``: Discounted RMSE (half-life weighted)

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

from typing import TYPE_CHECKING, Any

# Loss functions don't require optuna - always available
from .losses import (
    LOSSES,
    LossProtocol,
    discounted_rmse,
    expected_forecast_horizon,
    forecast_horizon,
    get_loss,
    lyapunov_weighted,
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
if TYPE_CHECKING:
    from .run import run_hpo as run_hpo
    from .utils import get_study_summary as get_study_summary
    from .utils import make_study_name as make_study_name


def __getattr__(name: str) -> Any:
    """Lazy import for optuna-dependent functions."""
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

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Main function
    "run_hpo",
    # Loss functions (always available)
    "LOSSES",
    "LossProtocol",
    "get_loss",
    "expected_forecast_horizon",
    "forecast_horizon",
    "lyapunov_weighted",
    "standard_loss",
    "discounted_rmse",
    # Utilities (require optuna)
    "get_study_summary",
    "make_study_name",
]
