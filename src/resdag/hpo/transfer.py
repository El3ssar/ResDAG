"""Warm-starting and cross-study transfer helpers for HPO.

This module provides the building blocks that let a study start *warm* instead
of cold:

- :func:`apply_warm_start` enqueues known-good configurations so they are
  evaluated **before** any sampler-proposed trial.
- :func:`transfer_trials` ingests already-evaluated ``COMPLETE`` trials from a
  prior study, keeping only the parameters that overlap with the new search
  space and logging the mismatched ones at ``INFO`` level.
- :func:`export_best_config` round-trips a study's best parameters into a plain
  ``dict`` suitable as a :func:`apply_warm_start` entry, so the winner of one
  run can seed the next.

All three operate on the parent-process :class:`optuna.Study` **before** any
workers are forked, so they compose cleanly with multi-process optimization
(``n_workers > 1``).

See Also
--------
resdag.hpo.run : High-level orchestrator that calls these helpers pre-fork.
"""

import logging
from typing import Any

import optuna
from optuna.trial import TrialState

from .storage import resolve_storage

__all__ = ["apply_warm_start", "transfer_trials", "export_best_config"]

logger = logging.getLogger(__name__)


def apply_warm_start(
    study: optuna.Study,
    warm_start: list[dict[str, Any]],
) -> int:
    """Enqueue known-good configurations ahead of sampler-proposed trials.

    Each mapping in *warm_start* is enqueued via
    :meth:`optuna.study.Study.enqueue_trial`, which guarantees the configuration
    is the next one handed to the objective — i.e. it runs **before** any trial
    the sampler would otherwise propose.  Enqueuing mutates the shared storage,
    so calling this in the parent process makes the warm-start trials visible to
    every forked worker.

    Parameters
    ----------
    study : optuna.Study
        The study to enqueue trials into.
    warm_start : list of dict
        Each dict maps parameter name to a fixed value.  Empty or ``None``
        entries are ignored.

    Returns
    -------
    int
        The number of configurations enqueued.

    Examples
    --------
    >>> apply_warm_start(study, [{"reservoir_size": 300, "spectral_radius": 0.9}])
    1
    """
    if not warm_start:
        return 0

    count = 0
    for params in warm_start:
        if not params:
            continue
        study.enqueue_trial(dict(params), skip_if_exists=False)
        count += 1

    if count:
        logger.info("Warm-started study with %d enqueued trial(s)", count)
    return count


def _resolve_source_study(
    transfer_from: "optuna.Study | str",
) -> optuna.Study:
    """Resolve *transfer_from* to a loaded :class:`optuna.Study`.

    Accepts an already-loaded study (returned as-is) or a storage path / URL
    (``"study.log"``, ``"study.db"``, ``"sqlite:///study.db"``) from which a
    single study is loaded.  Loading by path only succeeds when the storage
    holds exactly one study; otherwise the ambiguity is surfaced as a
    :class:`ValueError` so the caller passes an explicit study instead.

    Parameters
    ----------
    transfer_from : optuna.Study or str
        Source study, or a storage path/URL holding exactly one study.

    Returns
    -------
    optuna.Study
        The resolved source study.

    Raises
    ------
    ValueError
        If a storage path holds zero or more than one study, in which case the
        intended source is ambiguous.
    """
    if isinstance(transfer_from, optuna.Study):
        return transfer_from

    storage = resolve_storage(transfer_from, n_workers=1)
    assert storage is not None  # resolve_storage only returns None for in-memory
    summaries = optuna.study.get_all_study_summaries(storage=storage)
    if len(summaries) != 1:
        names = [s.study_name for s in summaries]
        raise ValueError(
            f"transfer_from={transfer_from!r} holds {len(summaries)} studies "
            f"({names}); pass an explicit optuna.Study to disambiguate."
        )
    return optuna.load_study(study_name=summaries[0].study_name, storage=storage)


def transfer_trials(
    study: optuna.Study,
    transfer_from: "optuna.Study | str",
    param_names: "set[str] | None" = None,
) -> int:
    """Copy ``COMPLETE`` trials from a prior study into *study*.

    Each finished trial from *transfer_from* is re-created with
    :func:`optuna.trial.create_trial` and added via
    :meth:`optuna.study.Study.add_trial`, so its evaluated value seeds the
    sampler without re-running the objective.  Only parameters that **overlap**
    with the new search space are kept:

    - When *param_names* is given, it is the authoritative set of search-space
      parameters; any source parameter outside it is dropped.
    - When *param_names* is ``None``, every source parameter is kept (the
      caller has asserted the search spaces match).

    Dropped (mismatched) parameters are reported once per source trial at
    ``INFO`` level.  Like :func:`apply_warm_start`, this mutates the shared
    storage in the parent process, so transferred trials are visible to forked
    workers.

    Parameters
    ----------
    study : optuna.Study
        Destination study to add trials to.
    transfer_from : optuna.Study or str
        Source study, or a storage path/URL holding exactly one study.
    param_names : set of str, optional
        Names of the new search space's parameters.  Source parameters outside
        this set are filtered out.  When ``None``, no filtering is applied.

    Returns
    -------
    int
        The number of trials successfully transferred.

    Notes
    -----
    A source trial that becomes empty after filtering (no overlapping
    parameters) is skipped, since a parameterless trial carries no transferable
    information to the sampler.

    Examples
    --------
    >>> transfer_trials(study, "prior_study.db", param_names={"spectral_radius"})
    7
    """
    source = _resolve_source_study(transfer_from)

    completed = [t for t in source.trials if t.state == TrialState.COMPLETE]
    transferred = 0
    for src in completed:
        if src.value is None:
            continue

        if param_names is None:
            kept_params = dict(src.params)
            kept_dists = dict(src.distributions)
        else:
            kept_params = {k: v for k, v in src.params.items() if k in param_names}
            kept_dists = {k: v for k, v in src.distributions.items() if k in param_names}
            dropped = sorted(set(src.params) - param_names)
            if dropped:
                logger.info(
                    "Transfer: dropping %d mismatched param(s) %s from source trial %d",
                    len(dropped),
                    dropped,
                    src.number,
                )

        if not kept_params:
            logger.info(
                "Transfer: skipping source trial %d (no overlapping params)",
                src.number,
            )
            continue

        new_trial = optuna.trial.create_trial(
            state=TrialState.COMPLETE,
            value=src.value,
            params=kept_params,
            distributions=kept_dists,
            user_attrs=dict(src.user_attrs),
        )
        study.add_trial(new_trial)
        transferred += 1

    logger.info(
        "Transferred %d of %d completed trial(s) from prior study %r",
        transferred,
        len(completed),
        source.study_name,
    )
    return transferred


def export_best_config(study: optuna.Study) -> dict[str, Any]:
    """Export a study's best parameters as a reusable ``warm_start`` entry.

    Returns the best completed trial's parameters as a plain ``dict`` that can
    be dropped straight into the ``warm_start`` list of a later
    :func:`resdag.hpo.run_hpo` call, letting the winner of one run seed the
    next.

    Parameters
    ----------
    study : optuna.Study
        A study with at least one ``COMPLETE`` trial.

    Returns
    -------
    dict
        A shallow copy of ``study.best_params``.

    Raises
    ------
    ValueError
        If the study has no completed trials.

    Examples
    --------
    >>> best = export_best_config(study)
    >>> run_hpo(..., warm_start=[best])  # doctest: +SKIP
    """
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not completed:
        raise ValueError("Cannot export best config: study has no completed trials.")
    return dict(study.best_params)
