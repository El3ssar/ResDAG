"""
Canonical Dataset Generators
============================

Canonical public home for resdag's benchmark time-series generators. The two
most standard import forms both work:

```python
import resdag.datasets
from resdag.datasets import lorenz
```

For back-compat the same callables are also reachable through
`resdag.utils.data`, and they are the *same* function objects, so identity
checks hold:

```text
resdag.datasets.lorenz is resdag.utils.data.lorenz  # True
```

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
kuramoto_sivashinsky
    1-D spatiotemporally chaotic PDE (ETDRK4) — the canonical parallel-RC
    benchmark; `D = n_grid` feature channels.

Each generator returns a `(1, T, D)` tensor, is seedable, supports a
transient/discard period, and supports optional normalization.

Examples
--------
>>> from resdag.datasets import lorenz
>>> series = lorenz(2000)  # (1, 2000, 3) — chaotic Lorenz-63 trajectory

See Also
--------
resdag.data : Canonical home for data preparation, file I/O, and streaming windows.
"""

from ..utils.data.datasets import (
    henon,
    kuramoto_sivashinsky,
    lorenz,
    mackey_glass,
    narma,
    rossler,
    sine,
)

__all__ = [
    "lorenz",
    "rossler",
    "henon",
    "mackey_glass",
    "narma",
    "sine",
    "kuramoto_sivashinsky",
]


def __dir__() -> list[str]:
    """Restrict `dir()` / tab-completion to the public API ([PEP 562](https://peps.python.org/pep-0562/))."""
    return list(__all__)
