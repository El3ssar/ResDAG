"""Utility functions for hyperparameter optimization.

Provides helpers for study management, naming, and result summarization.

See Also
--------
resdag.hpo.run : High-level HPO orchestrator that uses these utilities.
"""

import functools
import hashlib
import inspect
from pathlib import Path
from typing import Callable

import optuna

__all__ = ["make_study_name", "get_study_summary"]


def _unwrap_callable(fn: Callable) -> Callable:
    """Recover the underlying callable from a wrapper such as ``functools.partial``.

    ``functools.partial`` objects (and other wrappers) hide the actual factory
    behind a ``func`` attribute and carry no usable ``__name__``. Following the
    ``func`` chain restores the original callable so its name and source file can
    be recovered. ``functools.wraps``-style wrappers already expose ``__name__``,
    so they are left untouched.

    Parameters
    ----------
    fn : Callable
        The callable to unwrap.

    Returns
    -------
    Callable
        The innermost callable reachable via ``func`` attributes, or ``fn`` if it
        is not a recognised wrapper.
    """
    seen: set[int] = set()
    while isinstance(fn, functools.partial) and id(fn) not in seen:
        seen.add(id(fn))
        fn = fn.func
    return fn


def make_study_name(model_creator: Callable) -> str:
    """Generate a stable study name from the model creator callable.

    Creates a study name based on the callable's source file and name. It is
    robust to callables that lack the usual introspection metadata:

    - ``functools.partial`` objects are unwrapped to their underlying function so
      the wrapped factory's name and source file are used (a bare ``partial``
      raises ``TypeError`` in :func:`inspect.getsourcefile`).
    - Callables without a usable ``__name__`` (e.g. lambdas or callable class
      instances) fall back to their type name plus a short stable hash of their
      ``repr``, so two logically-distinct creators do not collide onto the same
      study name and silently share persisted storage.

    Parameters
    ----------
    model_creator : Callable
        The model creator callable to generate a name from. May be a plain
        function, a :class:`functools.partial`, a lambda, or any callable object.

    Returns
    -------
    str
        Study name in the format ``"filename:identifier"``. ``filename`` is
        ``"<unknown>"`` when the source file cannot be determined.

    Example
    -------
    >>> def my_model_creator(units):
    ...     return model
    >>> make_study_name(my_model_creator)
    'script:my_model_creator'
    """
    fn = _unwrap_callable(model_creator)

    try:
        src = inspect.getsourcefile(fn) or "<interactive>"
    except (TypeError, OSError):
        src = "<unknown>"

    name = getattr(fn, "__name__", None)
    if not name or name == "<lambda>":
        digest = hashlib.sha1(repr(model_creator).encode("utf-8")).hexdigest()[:8]
        name = f"{type(fn).__name__}-{digest}"

    return f"{Path(src).stem}:{name}"


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
