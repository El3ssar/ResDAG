"""Optuna pruner registry and resolution for HPO studies.

This module turns the ``pruner`` argument of :func:`resdag.hpo.run_hpo` into a
concrete :class:`optuna.pruners.BasePruner`.  A small registry maps short keys
(``"asha"``, ``"hyperband"``, ``"median"``, ``"threshold"``, ``"none"``) onto
factory callables, so a study can be early-stopped without the caller importing
Optuna's pruner classes by hand.

Pruning is driven by the intermediate ``trial.report`` calls that
:class:`resdag.hpo.runner.TrialRunner` emits at growing forecast-horizon
checkpoints: a configuration whose forecast has already diverged at an early
checkpoint reports a large loss, and the pruner can terminate it before paying
for the full horizon.  Resolution is identical in single- and multi-process
mode — the resolved pruner is passed to :func:`optuna.create_study` in the
parent **and** re-resolved (from the same key) inside each worker's
:func:`optuna.load_study`, so workers prune with the same policy.

See Also
--------
resdag.hpo.run : High-level orchestrator that resolves and wires the pruner.
resdag.hpo.runner : Emits the intermediate horizon reports that feed pruning.
resdag.hpo.runners : Single- and multi-process backends that honor the pruner.
"""

import functools
import logging
from typing import Callable

import optuna

__all__ = ["PRUNERS", "resolve_pruner"]

logger = logging.getLogger(__name__)

# Default upper-loss bound for the registry's ``"threshold"`` entry.
# :class:`optuna.pruners.ThresholdPruner` *requires* ``lower`` or ``upper`` at
# construction (a bare ``ThresholdPruner()`` raises), so the zero-arg factory has
# to supply one.  ``1.0`` prunes trials whose intermediate forecast loss exceeds
# 1.0 — divergent reservoir configurations blow well past this within the first
# few horizon checkpoints.  Callers needing a different bound (or a ``lower``
# bound) should pass a fully-configured ``ThresholdPruner`` instance instead of
# the bare key.
_THRESHOLD_DEFAULT_UPPER = 1.0

# Registry of pruner factories keyed by short, user-facing names.  Each factory
# takes no arguments and returns a fresh :class:`optuna.pruners.BasePruner`.  The
# ``"none"`` entry maps to :class:`optuna.pruners.NopPruner` — the default — so a
# pruner object always exists even when early stopping is disabled.
PRUNERS: dict[str, Callable[[], optuna.pruners.BasePruner]] = {
    "asha": optuna.pruners.SuccessiveHalvingPruner,
    "hyperband": optuna.pruners.HyperbandPruner,
    "median": optuna.pruners.MedianPruner,
    "threshold": functools.partial(optuna.pruners.ThresholdPruner, upper=_THRESHOLD_DEFAULT_UPPER),
    "none": optuna.pruners.NopPruner,
}


def resolve_pruner(
    pruner: "str | optuna.pruners.BasePruner | None",
) -> optuna.pruners.BasePruner:
    """Resolve a pruner specification to a concrete Optuna pruner.

    Accepts a registry key, an already-constructed pruner instance, or ``None``
    (and the string ``"none"``), always returning a usable
    :class:`optuna.pruners.BasePruner`.  ``None``/``"none"`` resolve to
    :class:`optuna.pruners.NopPruner` so the study never reaches Optuna without a
    pruner object.

    Parameters
    ----------
    pruner : str or optuna.pruners.BasePruner or None
        The pruner to resolve:

        - ``None`` or ``"none"`` → :class:`optuna.pruners.NopPruner` (no pruning).
        - A registry key — one of ``"asha"`` (ASHA /
          :class:`~optuna.pruners.SuccessiveHalvingPruner`), ``"hyperband"``,
          ``"median"`` (recommended starting point), or ``"threshold"`` (a
          :class:`~optuna.pruners.ThresholdPruner` with a default ``upper=1.0``
          loss bound — pass a configured instance for a different bound).
        - A :class:`optuna.pruners.BasePruner` instance, returned unchanged so
          callers can fully configure a pruner before passing it in.

    Returns
    -------
    optuna.pruners.BasePruner
        The resolved pruner, ready to hand to :func:`optuna.create_study` or
        :func:`optuna.load_study`.

    Raises
    ------
    ValueError
        If *pruner* is a string that is not a registered key.
    TypeError
        If *pruner* is neither ``None``, a string, nor a ``BasePruner``.

    Examples
    --------
    >>> from resdag.hpo.pruners import resolve_pruner
    >>> resolve_pruner("median")  # doctest: +ELLIPSIS
    <optuna.pruners._median.MedianPruner object at ...>
    >>> resolve_pruner(None)  # doctest: +ELLIPSIS
    <optuna.pruners._nop.NopPruner object at ...>
    """
    if pruner is None:
        return optuna.pruners.NopPruner()

    if isinstance(pruner, optuna.pruners.BasePruner):
        return pruner

    if isinstance(pruner, str):
        key = pruner.lower()
        if key not in PRUNERS:
            available = ", ".join(sorted(PRUNERS))
            raise ValueError(
                f"Unknown pruner '{pruner}'. Available registry keys: {available}. "
                "Alternatively pass an optuna.pruners.BasePruner instance."
            )
        return PRUNERS[key]()

    raise TypeError(
        "pruner must be None, a registry key string "
        f"({', '.join(sorted(PRUNERS))}), or an optuna.pruners.BasePruner; "
        f"got {type(pruner).__name__}."
    )
