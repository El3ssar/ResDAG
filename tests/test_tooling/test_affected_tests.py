"""Tests for ``tools/affected_tests.py`` — the selective-CI test selector.

These exercise the selector against the *real* repository layout (the same
import graph CI sees), asserting structural invariants rather than exact file
lists so they stay robust as the suite grows. The two properties that matter
most are:

* **Safety** — broad-impact changes and the selector's own failure modes fan
  out to the whole suite, and tests are never *under*-selected for a change
  they actually depend on (including dependencies reached only via a shared
  ``conftest`` fixture).
* **Selectivity** — an independent subsystem's change does not drag in
  unrelated packages' tests.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_SELECTOR = REPO_ROOT / "tools" / "affected_tests.py"


def _load_selector():
    spec = importlib.util.spec_from_file_location("affected_tests", _SELECTOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses can resolve the module by name
    # (CPython 3.14 looks it up in sys.modules during class processing).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AT = _load_selector()


# --------------------------------------------------------------------------- #
# Broad-impact / fail-safe behaviour
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "path",
    [
        "pyproject.toml",
        "uv.lock",
        "tests/conftest.py",
        "src/resdag/__init__.py",
        "tools/affected_tests.py",
        ".github/workflows/ci.yml",
    ],
)
def test_broad_impact_changes_run_full_suite(path: str) -> None:
    sel = AT.select_tests([path])
    assert sel.mode == "full"
    assert sel.tests == [AT.FULL_SUITE_TARGET]


def test_docs_only_change_runs_nothing() -> None:
    sel = AT.select_tests(["README.md", "docs/guide/index.md"])
    assert sel.mode == "none"
    assert sel.tests == []


def test_changed_test_file_is_always_selected() -> None:
    sel = AT.select_tests(["tests/test_hpo/test_losses.py"])
    assert "tests/test_hpo/test_losses.py" in sel.tests


# --------------------------------------------------------------------------- #
# Re-export resolution
# --------------------------------------------------------------------------- #


def test_facade_name_resolves_to_defining_module() -> None:
    index = AT.build_module_index()
    resolver = AT.ReExportResolver(index)
    got = resolver.resolve_ref(AT.ImportRef("resdag.layers", ["CGReadoutLayer"]))
    assert got == {"resdag.layers.readouts.cg_readout"}


def test_submodule_reexport_resolves_to_submodule() -> None:
    index = AT.build_module_index()
    resolver = AT.ReExportResolver(index)
    # `from resdag.utils import prepare_esn_data` re-exports through
    # ``resdag.utils.data`` to its canonical home ``resdag.data.prepare``,
    # not "all of utils".
    got = resolver.resolve_symbol("resdag.utils", "prepare_esn_data")
    assert got == {"resdag.data.prepare"}


# --------------------------------------------------------------------------- #
# Selectivity: independent subsystems stay isolated
# --------------------------------------------------------------------------- #


def test_hpo_change_does_not_pull_layers_or_init() -> None:
    sel = AT.select_tests(["src/resdag/hpo/losses.py"])
    assert "tests/test_hpo/test_losses.py" in sel.tests
    assert not any(t.startswith("tests/test_init/") for t in sel.tests)
    assert not any(t.startswith("tests/test_layers/") for t in sel.tests)


def test_io_change_stays_within_utils() -> None:
    sel = AT.select_tests(["src/resdag/utils/data/io.py"])
    assert "tests/test_utils/test_io.py" in sel.tests
    assert not any(t.startswith("tests/test_hpo/") for t in sel.tests)
    assert not any(t.startswith("tests/test_layers/") for t in sel.tests)


def test_layers_change_tracks_attractor_loss_dependency() -> None:
    sel = AT.select_tests(["src/resdag/layers/cells/esn_cell.py"])
    assert any(t.startswith("tests/test_layers/") for t in sel.tests)
    # The attractor-dimension loss imports observables, which also implement
    # reservoir diagnostics. At module granularity this is a real dependency.
    assert "tests/test_hpo/test_losses.py" in sel.tests
    assert "tests/test_hpo/test_storage.py" not in sel.tests


# --------------------------------------------------------------------------- #
# Safety: no under-selection, including via shared fixtures
# --------------------------------------------------------------------------- #


def test_source_change_selects_its_mirror_tests() -> None:
    sel = AT.select_tests(["src/resdag/init/graphs/erdos_renyi.py"])
    assert "tests/test_init/test_topologies.py" in sel.tests


def test_fixture_only_test_is_tracked_through_conftest() -> None:
    # tests/test_core/test_model.py builds its model purely through the
    # make_tiny_model fixture and imports only resdag.core — yet a change to the
    # readout it constructs must still select it.
    sel = AT.select_tests(["src/resdag/layers/readouts/cg_readout.py"])
    assert "tests/test_core/test_model.py" in sel.tests


def test_fixture_closure_captures_constructed_modules() -> None:
    resolver = AT.ReExportResolver(AT.build_module_index())
    closures = AT.build_fixture_closures(resolver)
    tiny = closures["make_tiny_model"]
    assert "resdag.layers.readouts.cg_readout" in tiny
    assert "resdag.layers.reservoirs.esn" in tiny
    # And the precise closure does not balloon to an entire subtree.
    assert "resdag.layers.transforms.power" not in tiny


def test_reservoir_change_selects_model_and_training_tests() -> None:
    sel = AT.select_tests(["src/resdag/layers/reservoirs/esn.py"])
    assert "tests/test_layers/test_esn_layer.py" in sel.tests
    assert "tests/test_models/test_premade_models.py" in sel.tests
    assert "tests/test_training/test_trainer.py" in sel.tests


def test_package_init_change_expands_to_subtree() -> None:
    # Editing a sub-package __init__ marks the whole sub-package changed, so its
    # consumers' tests are selected.
    sel = AT.select_tests(["src/resdag/layers/__init__.py"])
    assert any(t.startswith("tests/test_layers/") for t in sel.tests)


# --------------------------------------------------------------------------- #
# Regression guards: files the import graph can't see must still select the
# tests whose whole purpose is guarding them (previously all selected nothing).
# --------------------------------------------------------------------------- #


def test_golden_spec_helper_selects_model_tests() -> None:
    # tests/test_models/golden_forecast.py is a *helper* module (not a test_*.py),
    # invisible to the source import graph. Editing it (e.g. loosening
    # PREFIX_ATOL) must still run the golden regression test beside it.
    sel = AT.select_tests(["tests/test_models/golden_forecast.py"])
    assert sel.mode != "none"
    assert "tests/test_models/test_golden_forecast.py" in sel.tests


def test_golden_fixtures_select_golden_test() -> None:
    # Regenerated golden fixtures are data files under tests/ with no import
    # edge; the documented "regen + new .npz, no src change" workflow must
    # re-run the golden regression test.
    for fixture in (
        "tests/fixtures/golden_forecast/classic_esn.npz",
        "tests/fixtures/golden_forecast/ott_esn.npz",
    ):
        sel = AT.select_tests([fixture])
        assert sel.mode != "none", fixture
        assert "tests/test_models/test_golden_forecast.py" in sel.tests, fixture


def test_perf_targets_json_selects_benchmark_tests() -> None:
    # benchmarks/targets.json holds the perf-gate floors read at runtime by the
    # perf-regression test (incl. its well-formed check). A malformed/edited
    # targets.json must run those tests on the PR, not only on main/nightly.
    sel = AT.select_tests(["benchmarks/targets.json"])
    assert sel.mode != "none"
    assert "tests/test_benchmarks/test_perf_regression.py" in sel.tests


def test_reusable_test_workflow_is_broad_impact() -> None:
    # .github/workflows/_test.yml is the reusable workflow that actually invokes
    # pytest; a change to it can affect how *any* test runs.
    sel = AT.select_tests([".github/workflows/_test.yml"])
    assert sel.mode == "full"
    assert sel.tests == [AT.FULL_SUITE_TARGET]


def test_regen_golden_script_selects_golden_test() -> None:
    # tools/regen_golden_forecasts.py defines/writes the golden specs the
    # regression test asserts against; changing it must re-run that test.
    sel = AT.select_tests(["tools/regen_golden_forecasts.py"])
    assert sel.mode != "none"
    assert "tests/test_models/test_golden_forecast.py" in sel.tests


# --------------------------------------------------------------------------- #
# Invariant: the new data-dependency edges never *reduce* or over-fan-out the
# common path (ordinary src/test edits keep their prior, tight selection).
# --------------------------------------------------------------------------- #


def test_tests_helper_edit_does_not_run_full_suite() -> None:
    # A non-test helper under a test dir scopes to that dir's tests, never the
    # whole suite (only conftest/broad-impact do that).
    sel = AT.select_tests(["tests/test_models/golden_forecast.py"])
    assert sel.mode == "selected"
    assert all(t.startswith("tests/test_models/") for t in sel.tests)


def test_real_test_file_beside_helper_selects_only_itself() -> None:
    # A genuine test_*.py edit is still selected as just itself (the helper
    # sibling logic must skip real test files).
    sel = AT.select_tests(["tests/test_models/test_esn_builder.py"])
    assert sel.tests == ["tests/test_models/test_esn_builder.py"]


def test_source_edit_selects_golden_guard_through_helper_import() -> None:
    sel = AT.select_tests(["src/resdag/layers/reservoirs/esn.py"])
    assert "tests/test_layers/test_esn_layer.py" in sel.tests
    assert "tests/test_models/test_golden_forecast.py" in sel.tests


def test_deleted_source_module_runs_full_suite() -> None:
    sel = AT.select_tests(["src/resdag/removed_module.py"])
    assert sel.mode == "full"
    assert sel.tests == [AT.FULL_SUITE_TARGET]


def test_invalid_base_propagates_for_ci_fallback(monkeypatch) -> None:
    def broken_diff(*args):
        raise subprocess.CalledProcessError(128, ["git", *args])

    monkeypatch.setattr(AT, "_git", broken_diff)
    with pytest.raises(subprocess.CalledProcessError):
        AT.changed_files_from_git("missing-base", include_dirty=False)
