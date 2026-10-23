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
        "tools/ci_environment.py",
        ".github/actions/setup-environment/action.yml",
        ".github/workflows/ci.yml",
        ".github/workflows/compatibility.yml",
    ],
)
def test_broad_impact_changes_run_full_suite(path: str) -> None:
    sel = AT.select_tests([path])
    assert sel.mode == "full"
    assert sel.tests == [AT.FULL_SUITE_TARGET]


def test_docs_only_change_runs_executable_documentation_checks() -> None:
    sel = AT.select_tests(["README.md", "docs/guide/index.md"])
    assert sel.mode == "selected"
    assert "tests/test_init/test_correctness_contracts.py" in sel.tests
    assert "tests/test_examples_smoke.py" in sel.tests
    assert "tests/test_workflow_guide.py" in sel.tests
    assert "tests/test_documented_workflows.py" in sel.tests
    assert "tests/test_trajectory_guide.py" in sel.tests
    assert "tests/test_named_trajectory_guide.py" in sel.tests
    assert "tests/test_custom_components_guide.py" in sel.tests
    assert "tests/test_docs/test_forecast_experiment.py" in sel.tests
    assert "tests/test_docs/test_memory_workflows.py" in sel.tests
    assert "tests/test_hpo/test_tuning_guides.py" in sel.tests


def test_forecast_example_change_runs_its_body_not_only_its_import() -> None:
    selected = AT.select_tests(["examples/17_forecast_experiment.py"])
    assert "tests/test_examples_smoke.py" in selected.tests
    assert "tests/test_docs/test_forecast_experiment.py" in selected.tests


def test_cpu_workflow_harness_change_runs_its_protocol_and_real_cli_checks() -> None:
    selected = AT.select_tests(["benchmarks/cpu_workflows.py"])
    assert "tests/test_cpu_workflows.py" in selected.tests
    assert "tests/test_regression/test_cpu_workflow_evidence.py" in selected.tests


@pytest.mark.parametrize(
    "record", ["accuracy", "auto", "compile", "interleaved", "storage", "workflows"]
)
def test_cpu_measurement_data_changes_run_scientific_evidence_checks(record: str) -> None:
    selected = AT.select_tests([f"docs/assets/data/phase-five-cpu-{record}-2026-10-23.json"])
    assert "tests/test_regression/test_cpu_workflow_evidence.py" in selected.tests
    assert "tests/test_workflow_guide.py" in selected.tests
    assert "tests/test_docs/test_memory_workflows.py" in selected.tests


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


def test_hpo_change_reaches_common_workflow_but_not_initializers() -> None:
    sel = AT.select_tests(["src/resdag/hpo/losses.py"])
    assert "tests/test_hpo/test_losses.py" in sel.tests
    # Model/ESN now expose tune(), whose optional HPO package imports losses.
    # These workflows are no longer independent of HPO. Initializers still are.
    assert "tests/test_hpo/test_tuning_workflow.py" in sel.tests
    assert "tests/test_facade.py" in sel.tests
    assert not any(t.startswith("tests/test_init/") for t in sel.tests)


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


def test_shared_quality_reference_selects_every_consumer() -> None:
    selection = AT.select_tests(["tests/_reference_initialization.py"])
    assert selection.mode != "none"
    assert {
        "tests/test_reference_initialization.py",
        "tests/test_models/test_parallel_esn.py",
        "tests/test_training/test_force.py",
    } <= set(selection.tests)


def test_perf_targets_json_selects_benchmark_tests() -> None:
    # benchmarks/targets.json holds the perf-gate floors read at runtime by the
    # perf-regression test (incl. its well-formed check). A malformed/edited
    # targets.json must run those tests on the PR, not only on main/nightly.
    sel = AT.select_tests(["benchmarks/targets.json"])
    assert sel.mode != "none"
    assert "tests/test_benchmarks/test_perf_regression.py" in sel.tests


@pytest.mark.parametrize(
    "path",
    [
        "benchmarks/rc_bench/data.py",
        "benchmarks/rc_bench/adapters/reservoirpy_adapter.py",
        "benchmarks/rc_bench/budget.py",
        "benchmarks/rc_bench/budget_adapters.py",
        "benchmarks/rc_bench/_threads.py",
        "benchmarks/julia/rc_bench.jl",
        "benchmarks/julia/rc_budget.jl",
        "benchmarks/julia/Project.toml",
        "benchmarks/julia/Manifest.toml",
    ],
)
def test_comparison_changes_select_physical_window_oracles(path: str) -> None:
    selected = AT.select_tests([path])
    assert selected.mode == "selected"
    assert "tests/test_benchmark_protocol.py" in selected.tests
    assert "tests/test_benchmark_budget.py" in selected.tests
    assert "tests/test_benchmark_julia_protocol.py" in selected.tests
    if path.startswith("benchmarks/rc_bench/"):
        assert "tests/test_benchmark_threads.py" in selected.tests


def test_budget_protocol_documentation_selects_budget_oracles() -> None:
    selected = AT.select_tests(["benchmarks/BUDGET_PROTOCOL.md"])
    assert "tests/test_benchmark_budget.py" in selected.tests
    assert "tests/test_benchmark_threads.py" in selected.tests


@pytest.mark.parametrize(
    "path", ["docs/workflows/sparse-reservoirs.md", "docs/reference/lsmr-readout.md"]
)
def test_memory_workflow_documentation_selects_its_executable_recipes(path: str) -> None:
    selected = AT.select_tests([path])
    assert "tests/test_docs/test_memory_workflows.py" in selected.tests


@pytest.mark.parametrize(
    "path",
    [
        "benchmarks/accuracy_regression.py",
        "benchmarks/tuned_accuracy.py",
        "benchmarks/results/phase-five-accuracy-2026-10-23.json",
        "benchmarks/results/phase-five-tuned-accuracy-2026-10-23.json",
        "docs/assets/data/phase-five-capability-2026-10-23.json",
    ],
)
def test_accuracy_experiment_selects_its_regression_tests(path: str) -> None:
    selected = AT.select_tests([path])
    assert "tests/test_regression/test_phase5_accuracy.py" in selected.tests


@pytest.mark.parametrize(
    "path",
    [
        "benchmarks/accuracy_regression.py",
        "benchmarks/tuned_accuracy.py",
        "benchmarks/rc_bench/_threads.py",
        "benchmarks/requirements.txt",
    ],
)
def test_accuracy_workers_select_capability_cli_guards(path: str) -> None:
    selected = AT.select_tests([path])
    assert "tests/test_regression/test_tuned_capability_cli.py" in selected.tests


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


@pytest.mark.parametrize("changed", ["docs/start/end-to-end.md", "hooks/docs_autogen.py"])
def test_documentation_changes_select_executable_catalog(changed):
    selection = AT.select_tests([changed])
    assert "tests/test_init/test_correctness_contracts.py" in selection.tests


def test_example_body_changes_select_example_checks():
    selection = AT.select_tests(["examples/10_hpo.py"])
    assert "tests/test_examples_smoke.py" in selection.tests


@pytest.mark.parametrize(
    "changed",
    [
        "src/resdag/artifacts/__init__.py",
        "src/resdag/artifacts/_api.py",
        "src/resdag/artifacts/_models.py",
        "src/resdag/artifacts/_codecs.py",
        "src/resdag/artifacts/_registry.py",
        "docs/workflows/save-load.md",
    ],
)
def test_artifact_sources_and_guide_select_roundtrips_and_executable_recipe(changed):
    selection = AT.select_tests([changed])
    artifact_tests = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "tests/test_artifacts").glob("test_*.py")
    }
    assert artifact_tests
    assert artifact_tests <= set(selection.tests)
    assert "tests/test_artifact_guides.py" in selection.tests
    if changed.startswith("docs/"):
        assert "tests/test_core/test_diagrams.py" in selection.tests
        assert "tests/test_workflow_guide.py" in selection.tests


def test_diagram_source_also_selects_its_rendered_guide():
    selection = AT.select_tests(["src/resdag/core/_diagram.py"])
    assert "tests/test_artifact_guides.py" in selection.tests
    assert "tests/test_core/test_diagrams.py" in selection.tests
