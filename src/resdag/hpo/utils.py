"""Utility functions for hyperparameter optimization.

Provides helpers for study management, naming, and result summarization.
"""

import inspect
from pathlib import Path
from typing import Callable

import optuna

__all__ = ["make_study_name", "get_study_summary"]


def make_study_name(model_creator: Callable) -> str:
    """Generate a study name from the model creator function.

    Creates a unique study name based on the source file and function name.
    Useful for identifying studies in storage backends.

    Parameters
    ----------
    model_creator : Callable
        The model creator function to generate a name from.

    Returns
    -------
    str
        Study name in format "filename:function_name".

    Example
    -------
    >>> def my_model_creator(units):
    ...     return model
    >>> make_study_name(my_model_creator)
    'script:my_model_creator'
    """
    src = inspect.getsourcefile(model_creator) or "<interactive>"
    func = getattr(model_creator, "__name__", "model_creator")
    return f"{Path(src).stem}:{func}"


def get_study_summary(study: optuna.Study, top_n: int = 5) -> str:
    """Generate a human-readable summary of an Optuna study.

    Creates a formatted text summary including study statistics, best trial
    information, and top N trials.

    Parameters
    ----------
    study : optuna.Study
        The Optuna study to summarize.
    top_n : int, default=5
        Number of top-performing trials to include.

    Returns
    -------
    str
        Formatted multi-line summary string.

    Example
    -------
    >>> study = optuna.create_study()
    >>> # ... run optimization ...
    >>> print(get_study_summary(study))
    """
    lines = []
    lines.append("=" * 60)
    lines.append("Study Summary")
    lines.append("=" * 60)
    lines.append(f"Study Name: {study.study_name}")

    total_trials = len(study.trials)
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])

    lines.append(f"Total Trials: {total_trials}")
    lines.append(f"  Completed: {completed}")
    lines.append(f"  Pruned: {pruned}")
    lines.append(f"  Failed: {failed}")
    lines.append("")

    if completed > 0:
        lines.append("-" * 60)
        lines.append("Best Trial")
        lines.append("-" * 60)
        lines.append(f"Trial Number: {study.best_trial.number}")
        lines.append(f"Value: {study.best_value:.6f}")
        lines.append("")
        lines.append("Parameters:")
        for k, v in study.best_params.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.6f}")
            else:
                lines.append(f"  {k}: {v}")
        lines.append("")

        # Top N trials
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        sorted_trials = sorted(completed_trials, key=lambda t: t.value)

        lines.append("-" * 60)
        lines.append(f"Top {min(top_n, len(sorted_trials))} Trials")
        lines.append("-" * 60)
        for i, trial in enumerate(sorted_trials[:top_n], 1):
            lines.append(f"{i}. Trial {trial.number}: {trial.value:.6f}")

    lines.append("=" * 60)
    return "\n".join(lines)


def get_best_params(study: optuna.Study) -> dict:
    """Get the best parameters from a study.

    Convenience function that returns the best trial's parameters.

    Parameters
    ----------
    study : optuna.Study
        The Optuna study.

    Returns
    -------
    dict
        Dictionary of best hyperparameters.

    Raises
    ------
    ValueError
        If no completed trials exist.
    """
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise ValueError("No completed trials in study.")
    return study.best_params
