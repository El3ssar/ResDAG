"""Packaging and citation metadata guarantees.

These checks lock in adoption-critical, easy-to-drift facts:

* an MIT ``LICENSE`` file exists at the repository root (the README badge and
  ``license = "MIT"`` in ``pyproject.toml`` both promise one);
* a machine-readable ``CITATION.cff`` exists, parses, and declares a software
  ``version`` that stays in lock-step with :data:`resdag.__version__` (both are
  bumped together by python-semantic-release);
* every documentation URL uses the one canonical GitHub Pages casing,
  ``el3ssar.github.io/ResDAG/`` — the lowercase ``/resdag/`` spelling returns
  HTTP 404 and must never reappear;
* the package ships a PEP 561 ``py.typed`` marker so downstream type checkers
  honour ``resdag``'s inline annotations instead of treating it as untyped;
* mypy checks all first-party source; import-skipping rules can target only
  explicit third-party namespaces, including optional newer-Python packages.

The repository-layout checks are skipped gracefully when the tests run outside
a source checkout (e.g. against an installed wheel), where the root metadata
files are not shipped.
"""

import re
import subprocess
import tomllib
from importlib.resources import files
from pathlib import Path

import pytest

import resdag

ROOT = Path(__file__).resolve().parents[1]

# Built from fragments so this file does not itself trip the drift scan below.
_BAD_PAGES_URL = "el3ssar.github.io/" + "resdag/"
_GOOD_PAGES_URL = "el3ssar.github.io/ResDAG/"

# Text files worth scanning for stale doc-URL casing.
_TEXT_SUFFIXES = {".md", ".toml", ".yml", ".yaml", ".cff", ".py"}


def _require_root_file(name: str) -> Path:
    """Return *name* under the repo root, skipping if it is not present."""
    path = ROOT / name
    if not path.is_file():
        pytest.skip(f"{name} not present (running outside a source checkout)")
    return path


def test_license_file_present_and_mit() -> None:
    """An MIT ``LICENSE`` exists with the expected copyright holder."""
    text = _require_root_file("LICENSE").read_text(encoding="utf-8")
    assert "MIT License" in text
    assert "Daniel Estevez-Moya" in text
    assert "WITHOUT WARRANTY OF ANY KIND" in text


def test_citation_cff_present_and_valid() -> None:
    """``CITATION.cff`` parses and carries the fields GitHub's widget needs."""
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load(_require_root_file("CITATION.cff").read_text(encoding="utf-8"))

    assert data["cff-version"] == "1.2.0"
    assert data["type"] == "software"
    assert data["license"] == "MIT"
    assert data["message"]
    assert data["title"]
    assert data["authors"], "CITATION.cff must list at least one author"
    assert data["authors"][0]["family-names"] == "Estevez-Moya"


def test_citation_version_matches_package() -> None:
    """The cited ``version`` stays in lock-step with ``resdag.__version__``."""
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load(_require_root_file("CITATION.cff").read_text(encoding="utf-8"))
    assert str(data["version"]) == resdag.__version__


def test_citation_and_pyproject_use_canonical_pages_casing() -> None:
    """The Documentation URL uses the casing that returns HTTP 200."""
    pyproject = _require_root_file("pyproject.toml").read_text(encoding="utf-8")
    assert _GOOD_PAGES_URL in pyproject
    assert _BAD_PAGES_URL not in pyproject

    citation = _require_root_file("CITATION.cff").read_text(encoding="utf-8")
    assert _GOOD_PAGES_URL in citation
    assert _BAD_PAGES_URL not in citation


def test_py_typed_marker_ships_with_package() -> None:
    """``resdag`` ships a PEP 561 ``py.typed`` marker.

    Without it, downstream type checkers treat an installed ``resdag`` as
    untyped and ignore its inline annotations — defeating the strict-typing
    policy in ``pyproject.toml``. ``importlib.resources`` resolves the marker
    inside the *installed* package, so this also holds for a built wheel.
    """
    marker = files("resdag") / "py.typed"
    assert marker.is_file(), "resdag must ship a py.typed marker (PEP 561)"


def test_py_typed_marker_is_empty() -> None:
    """The marker is the empty-file form.

    A non-empty ``py.typed`` (e.g. ``partial\\n``) signals a *partial* stub
    package; ``resdag`` is fully typed inline, so the marker must be empty.
    """
    marker = files("resdag") / "py.typed"
    assert marker.read_text(encoding="utf-8") == ""


def test_mypy_overrides_cannot_skip_or_suppress_first_party_checking() -> None:
    """Only explicitly named third-party import trees can be skipped.

    Optional dependencies may require a newer Python parser than ResDAG's
    minimum version. Skipping those sources must never exempt ResDAG itself
    or introduce error suppressions into its whole-package type check.
    """
    pyproject = _require_root_file("pyproject.toml")
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    overrides = data.get("tool", {}).get("mypy", {}).get("overrides", [])
    if not overrides:
        pytest.skip("no mypy overrides defined")

    for entry in overrides:
        assert entry.get("follow_imports") == "skip"
        assert not entry.get("ignore_errors")
        assert "disable_error_code" not in entry
        modules = entry["module"]
        if isinstance(modules, str):
            modules = [modules]
        assert modules
        for module in modules:
            namespace = module.partition(".")[0]
            assert (
                namespace.isidentifier() and namespace != "resdag"
            ), f"Import policy must name an external namespace, not {module!r}"


def test_no_lowercase_pages_url_in_tracked_files() -> None:
    """No tracked text file resurrects the 404-ing lowercase Pages URL."""
    try:
        listing = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git not available; cannot enumerate tracked files")

    tracked = [line for line in listing.stdout.splitlines() if line]
    if not tracked:
        pytest.skip("no tracked files reported by git")

    self_rel = Path(__file__).resolve().relative_to(ROOT).as_posix()
    offenders = []
    for rel in tracked:
        if rel == self_rel or Path(rel).suffix not in _TEXT_SUFFIXES:
            continue
        path = ROOT / rel
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if re.search(re.escape(_BAD_PAGES_URL), text):
            offenders.append(rel)

    assert not offenders, f"lowercase Pages URL found in: {offenders}"
