"""Import-hygiene guarantees for the top-level :mod:`resdag` package.

A plain ``import resdag`` must be side-effect-free: it must not emit a
``DeprecationWarning`` and must not eagerly pull in the optional, heavy
``scipy``/``optuna`` stack (those belong to :mod:`resdag.hpo`, which is resolved
lazily via the package ``__getattr__``).  The deprecated ``resdag.composition``
and ``resdag.layers.custom`` shims must still be importable on demand and must
warn only when actually accessed — never on ``import resdag``.

The "clean import" checks run in a fresh subprocess so that modules already
imported by the surrounding test session cannot mask the behaviour under test.
"""

import importlib
import subprocess
import sys

import pytest

import resdag


def _run(code: str) -> subprocess.CompletedProcess[str]:
    """Execute *code* in a fresh interpreter and capture its output."""
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )


def test_import_resdag_is_warning_free() -> None:
    """``python -W error::DeprecationWarning -c 'import resdag'`` exits 0."""
    result = subprocess.run(
        [sys.executable, "-W", "error::DeprecationWarning", "-c", "import resdag"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_import_resdag_does_not_pull_scipy_or_optuna() -> None:
    """After ``import resdag`` neither scipy nor optuna is loaded."""
    code = (
        "import sys, resdag\n"
        "assert 'optuna' not in sys.modules, 'optuna eagerly imported'\n"
        "assert 'scipy' not in sys.modules, 'scipy eagerly imported'\n"
        "assert 'resdag.hpo' not in sys.modules, 'resdag.hpo eagerly imported'\n"
        "assert 'resdag.composition' not in sys.modules, 'resdag.composition eagerly imported'\n"
    )
    result = _run(code)
    assert result.returncode == 0, result.stderr


def test_lazy_names_resolve_on_demand() -> None:
    """``hpo``/``composition`` and the HPO helpers resolve through ``__getattr__``."""
    code = (
        "import resdag\n"
        "assert resdag.hpo.__name__ == 'resdag.hpo'\n"
        "assert resdag.composition.__name__ == 'resdag.composition'\n"
        "assert callable(resdag.run_hpo)\n"
        "assert isinstance(resdag.LOSSES, dict)\n"
        "assert callable(resdag.get_study_summary)\n"
    )
    result = _run(code)
    assert result.returncode == 0, result.stderr


def test_unknown_top_level_attribute_raises_attribute_error() -> None:
    """A bogus top-level attribute still raises ``AttributeError``."""
    with pytest.raises(AttributeError):
        resdag.does_not_exist  # type: ignore[attr-defined]


@pytest.mark.parametrize("module_name", ["resdag.composition", "resdag.layers.custom"])
def test_deprecated_shims_warn_on_import(module_name: str) -> None:
    """Importing a deprecated shim emits a ``DeprecationWarning`` on access."""
    # Drop any cached copy so the module body (which warns) re-executes.
    sys.modules.pop(module_name, None)
    with pytest.warns(DeprecationWarning):
        importlib.import_module(module_name)
