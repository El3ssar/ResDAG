"""Reuse complete settings by re-evaluating them on a destination task.

Copying completed scores is an explicit operation restricted to matching bound
task protocols and objective directions. Transfer preserves full settings,
suggestions, and source-study provenance; it never filters a scored model.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from math import isfinite
from typing import Any, Literal

import optuna
from optuna.trial import TrialState

from ._provenance import (
    PROTOCOL_KEY,
    RESOLVED_SETTINGS_KEY,
    REUSED_SETTINGS_KEY,
    resolved_settings,
    trial_settings,
)
from .storage import _close_storage, resolve_storage

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
    the sampler would otherwise propose.  The parent enqueues settings before preparing finite numerical job windows.
    Queued fixed values remain available even when the current search-space
    function does not suggest them.

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
        settings = resolved_settings(params)
        study.enqueue_trial(
            settings, user_attrs={REUSED_SETTINGS_KEY: repr(settings)}, skip_if_exists=False
        )
        count += 1

    if count:
        logger.info("Warm-started study with %d enqueued trial(s)", count)
    return count


@contextmanager
def _source_study(
    transfer_from: "optuna.Study | str",
) -> Iterator[optuna.Study]:
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

    Yields
    ------
    optuna.Study
        The source study; only resources opened from a path are released.

    Raises
    ------
    ValueError
        If a storage path holds zero or more than one study, in which case the
        intended source is ambiguous.
    """
    if isinstance(transfer_from, optuna.Study):
        yield transfer_from
        return

    storage = resolve_storage(transfer_from, n_workers=1)
    assert storage is not None  # resolve_storage only returns None for in-memory
    primary_error: BaseException | None = None
    try:
        summaries = optuna.study.get_all_study_summaries(storage=storage)
        if len(summaries) != 1:
            names = [s.study_name for s in summaries]
            raise ValueError(
                f"transfer_from={transfer_from!r} holds {len(summaries)} studies "
                f"({names}); pass an explicit optuna.Study to disambiguate."
            )
        yield optuna.load_study(study_name=summaries[0].study_name, storage=storage)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        _close_storage(storage, primary_error=primary_error)


def transfer_trials(
    study: optuna.Study,
    transfer_from: "optuna.Study | str",
    *,
    mode: Literal["reevaluate", "scores"] = "reevaluate",
    protocol_hash: str | None = None,
) -> int:
    """Reuse complete configurations, re-evaluating them by default.

    ``mode="reevaluate"`` enqueues full recorded resolved settings, including
    fixed values that were not Optuna suggestions. Old scores are not copied.
    Source study/trial/protocol provenance is retained on every queued trial.
    A changed objective direction is allowed because the new task evaluates
    the settings again. No parameter is silently removed to fit a new space.

    ``mode="scores"`` imports finite completed values only when both studies
    are bound to exactly ``protocol_hash`` and their directions match. The
    caller's protocol must identify actual data, validation, objective and
    construction settings. An unbound or changed task must re-evaluate instead.
    All source parameters/distributions and complete resolved settings remain
    intact; incompatible model settings should be adapted explicitly before
    enqueueing, never combined with an old score.

    Returns the number of queued or copied configurations. Invalid/nonfinite
    source evaluations are skipped. Caller-owned study connections are not
    closed; a source opened from a path is released on success and failure.
    """
    if mode not in ("reevaluate", "scores"):
        raise ValueError("mode must be 'reevaluate' or 'scores'")
    with _source_study(transfer_from) as source:
        source_protocol = source.user_attrs.get(PROTOCOL_KEY)
        if mode == "scores":
            if (
                not protocol_hash
                or source_protocol != protocol_hash
                or study.user_attrs.get(PROTOCOL_KEY) != protocol_hash
            ):
                raise ValueError(
                    "Copying evaluated scores requires an identical bound task protocol; "
                    "use mode='reevaluate' to reuse settings for a new task"
                )
            if study.directions != source.directions:
                raise ValueError("Identical-score transfer requires matching objective directions")
        transferred = 0
        source_trials = source.get_trials(states=(TrialState.COMPLETE,))
        if mode == "reevaluate" and len(source.directions) == 1:
            maximize = source.direction == optuna.study.StudyDirection.MAXIMIZE
            source_trials.sort(
                key=lambda trial: (
                    (
                        (-(trial.value or 0.0) if maximize else trial.value or 0.0)
                        if trial.value is not None and isfinite(trial.value)
                        else float("inf")
                    ),
                    trial.number,
                )
            )
        entries = []
        for src in source_trials:
            values = src.values
            if (
                not values
                or not all(isfinite(value) for value in values)
                or "error" in src.user_attrs
                or src.user_attrs.get("nonfinite_loss", False)
            ):
                continue
            settings = trial_settings(src)
            if not settings:
                continue
            entries.append((src, settings))
        # Validate recorded configurations before changing the destination.
        # Storage write failures can still interrupt a standard Optuna import.
        for src, settings in entries:
            values = src.values
            provenance = {
                "mode": mode,
                "study_name": source.study_name,
                "trial_number": src.number,
                "protocol_hash": source_protocol,
                "resolved_settings_literal": repr(settings),
            }
            if mode == "reevaluate":
                study.enqueue_trial(
                    {**src.params, **settings},
                    user_attrs={"resdag_transfer": provenance, REUSED_SETTINGS_KEY: repr(settings)},
                    skip_if_exists=False,
                )
            else:
                attrs = dict(src.user_attrs)
                attrs["resdag_transfer"] = provenance
                attrs[RESOLVED_SETTINGS_KEY] = repr(settings)
                study.add_trial(
                    optuna.trial.create_trial(
                        state=TrialState.COMPLETE,
                        values=values,
                        params=dict(src.params),
                        distributions=dict(src.distributions),
                        user_attrs=attrs,
                    )
                )
            transferred += 1
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
        Complete recorded resolved settings, including fixed factory values.
        External studies without that record fall back to their suggestions.

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
    best = study.best_trial
    if (
        "error" in best.user_attrs
        or best.user_attrs.get("nonfinite_loss", False)
        or best.value is None
        or not isfinite(best.value)
    ):
        raise ValueError("Cannot export an invalid or non-finite winning configuration")
    return {**best.params, **trial_settings(best)}
