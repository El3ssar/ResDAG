"""Optuna pruner registry and resolution for HPO studies.

This module turns the ``pruner`` argument of :func:`resdag.hpo.run_hpo` into a
concrete :class:`optuna.pruners.BasePruner`.  A small registry maps short keys
(``"asha"``, ``"hyperband"``, ``"median"``, ``"threshold"``, ``"none"``) onto
factory callables, so a study can be early-stopped without the caller importing
Optuna's pruner classes by hand.

Pruning is driven by intermediate ``trial.report`` calls at growing
forecast prefixes. Forecast chunks stop being generated as soon as the pruner
rejects a prefix. Both single- and multi-worker studies report/check the real
Optuna Trial in the parent; numerical workers receive only a progress callback
and never open storage or own a separate study/pruner.

See Also
--------
resdag.hpo.run : High-level orchestrator that resolves and wires the pruner.
resdag.hpo.trial_runner : Emits the intermediate horizon reports that feed pruning.
resdag.hpo.dispatch : Single- and multi-process backends that honor the pruner.
"""

import logging
from typing import Callable

import optuna

__all__ = ["PRUNERS", "resolve_pruner"]

logger = logging.getLogger(__name__)

# Default bound + warmup for the registry's ``"threshold"`` entry.
#
# :class:`optuna.pruners.ThresholdPruner` *requires* ``lower`` or ``upper`` at
# construction (a bare ``ThresholdPruner()`` raises), so the zero-arg factory has
# to supply one.  The default is tuned for the module's **default and recommended
# loss family** — the *negated forecast-horizon* losses (``"efh"``,
# ``"forecast_horizon"``, ``"soft_horizon"``).  Those losses return
# ``-horizon``: a good configuration scores a large *negative* value, while a
# **divergent** configuration whose forecast collapses immediately scores ``≈ 0``
# (a zero-length horizon).  The old default ``upper=1.0`` could therefore **never
# fire** for them — every report is ``≤ 0 < 1.0``.  We instead prune when an
# intermediate report *rises above* a small negative bound (``upper < 0``),
# i.e. the forecast horizon has effectively collapsed to zero.
#
# ``n_warmup_steps`` is expressed in **checkpoint indices** (the runner reports by
# 1-based checkpoint index, not horizon length — see
# :meth:`resdag.hpo.trial_runner.TrialRunner._evaluate_with_checkpoints`), so skipping
# the first two checkpoints is horizon-independent.  The skip matters because at
# the earliest checkpoint even a *good* configuration has not yet demonstrated a
# long horizon and reads ``≈ 0``; by the third checkpoint good and divergent
# configurations have separated cleanly.
#
# This default does **not** suit the raw-error losses (``"standard"``,
# ``"lyapunov"``), whose reports are strictly *positive* and whose good/bad scale
# is data-dependent — pass a fully-configured ``ThresholdPruner`` (e.g.
# ``ThresholdPruner(upper=<your bound>)``) for those, or use ``"median"`` for a
# scale-free relative policy.
_THRESHOLD_DEFAULT_UPPER = -0.1
_THRESHOLD_DEFAULT_WARMUP = 2


def _default_threshold_pruner() -> optuna.pruners.ThresholdPruner:
    """Build the registry's default divergence-catching ``ThresholdPruner``.

    Prunes negated-horizon trials whose intermediate loss rises above
    :data:`_THRESHOLD_DEFAULT_UPPER` (a collapsed, ``≈ 0`` forecast horizon)
    once past :data:`_THRESHOLD_DEFAULT_WARMUP` checkpoints.  See the module-level
    comment for why a fixed positive bound could never fire for the default loss.
    """
    return optuna.pruners.ThresholdPruner(
        upper=_THRESHOLD_DEFAULT_UPPER,
        n_warmup_steps=_THRESHOLD_DEFAULT_WARMUP,
    )


# Registry of pruner factories keyed by short, user-facing names.  Each factory
# takes no arguments and returns a fresh :class:`optuna.pruners.BasePruner`.  The
# ``"none"`` entry maps to :class:`optuna.pruners.NopPruner` — the default — so a
# pruner object always exists even when early stopping is disabled.
PRUNERS: dict[str, Callable[[], optuna.pruners.BasePruner]] = {
    "asha": optuna.pruners.SuccessiveHalvingPruner,
    "hyperband": optuna.pruners.HyperbandPruner,
    "median": optuna.pruners.MedianPruner,
    "threshold": _default_threshold_pruner,
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
          :class:`~optuna.pruners.ThresholdPruner` tuned for the default
          *negated-horizon* losses: it prunes a trial once its intermediate loss
          rises above ``upper=-0.1`` after two warmup checkpoints — i.e. its
          forecast horizon has collapsed to ``≈ 0``.  Pass a configured instance
          for a different bound, e.g. a positive ``upper`` for the raw-error
          ``"standard"`` / ``"lyapunov"`` losses).
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
