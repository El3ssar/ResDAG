"""
Canonical Dataset Generators
============================

Discoverable, library-style entry point for resdag's benchmark time-series
generators. This package re-exports the generators that live in
:mod:`resdag.utils.data` so that the two most standard import forms both work::

    import resdag.datasets
    from resdag.datasets import lorenz

The re-exported callables are the *same* function objects as
``resdag.utils.data.lorenz`` (etc.), so identity checks hold::

    resdag.datasets.lorenz is resdag.utils.data.lorenz  # True

Generators
----------
lorenz, rossler
    Chaotic continuous-time attractors (RK4 integration).
henon
    Chaotic discrete-time map.
mackey_glass
    Chaotic delay-differential-equation series.
narma
    Nonlinear system-identification benchmark (input/output series).
sine
    Simple periodic signal for sanity checks.

Each generator returns a ``(1, T, D)`` tensor, is seedable, supports a
transient/discard period, and supports optional normalization.

Examples
--------
>>> from resdag.datasets import lorenz
>>> series = lorenz(2000)  # (1, 2000, 3) — chaotic Lorenz-63 trajectory

See Also
--------
resdag.utils.data : Full data-loading / preparation module (the canonical home
    of these generators).
"""

from ..utils.data.datasets import henon, lorenz, mackey_glass, narma, rossler, sine

__all__ = [
    "lorenz",
    "rossler",
    "henon",
    "mackey_glass",
    "narma",
    "sine",
]


def __dir__() -> list[str]:
    """Restrict ``dir()`` / tab-completion to the public API (:pep:`562`)."""
    return list(__all__)
