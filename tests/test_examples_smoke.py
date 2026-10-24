"""Smoke test for the shipped ``examples/`` scripts.

The numbered examples (``examples/00_*.py`` … ``examples/14_*.py``) are the
first thing a new user runs, yet nothing in CI touches them — a rename, a moved
symbol, or a dropped public export can break an example and ship silently. This
test closes that gap the cheap way: it discovers every ``examples/*.py`` and
*imports* it, which parses the file and resolves its top-level imports.

Import — not execution — is deliberate. Every example wraps its real work in a
``main()`` behind ``if __name__ == "__main__":``, so importing runs only
module-level code (imports and constants) and never the expensive training /
forecasting / HPO / GPU-benchmark bodies. That is enough to catch the class of
breakage this test exists for: ``ImportError``/``AttributeError`` from a moved
or renamed public symbol (e.g. after an API audit). It stays fast enough to run
on every CI job.

To avoid ``sys.modules`` collisions and import caching between examples, each
file is loaded fresh into a throwaway module object under a unique name.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

# Examples whose *top-level* imports pull in an optional dependency, mapped to
# the module that must be importable for the example to load. (Examples that
# import their optional dep lazily inside ``main()`` — e.g. 10_hpo.py, which
# imports optuna inside a try/except in main() — import fine without it and are
# intentionally absent here.)
_OPTIONAL_DEP_IMPORTS: dict[str, str] = {}


def _discover_examples() -> list[Path]:
    """All runnable example scripts, sorted by their numeric prefix."""
    if not _EXAMPLES_DIR.is_dir():  # pragma: no cover - defensive
        return []
    return sorted(p for p in _EXAMPLES_DIR.glob("*.py") if not p.name.startswith("_"))


_EXAMPLE_PATHS = _discover_examples()


def test_examples_directory_is_discovered() -> None:
    """Guard against the glob silently matching nothing (moved/renamed dir)."""
    assert _EXAMPLE_PATHS, f"no example scripts found under {_EXAMPLES_DIR}"


@pytest.mark.parametrize(
    "example_path",
    _EXAMPLE_PATHS,
    ids=[p.stem for p in _EXAMPLE_PATHS],
)
def test_example_imports(example_path: Path, isolated_component_registries) -> None:
    """Each example parses and its top-level imports resolve.

    The example's ``main()`` is *not* executed — importing only runs
    module-level code (imports + constants), which is what verifies that every
    public symbol the example references still exists after an API change.
    """
    required = _OPTIONAL_DEP_IMPORTS.get(example_path.name)
    if required is not None:
        pytest.importorskip(required, reason=f"{example_path.name} needs {required}")

    module_name = f"_resdag_example_{example_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, example_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    # Register before exec so any relative/self references resolve, then always
    # unregister to keep the throwaway module out of the shared import cache.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
