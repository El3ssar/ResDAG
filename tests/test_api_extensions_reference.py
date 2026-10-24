"""Public extension points and their explicit reference inventories stay aligned."""

import importlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs" / "reference"

PUBLIC_PACKAGES = (
    "resdag.init",
    "resdag.init.graphs",
    "resdag.init.input_feedback",
    "resdag.init.matrices",
    "resdag.init.topology",
    "resdag.init.utils",
    "resdag.data",
    "resdag.datasets",
    "resdag.utils",
    "resdag.utils.data",
    "resdag.artifacts",
)

PUBLIC_LEAVES = (
    "resdag.init.recurrent",
    "resdag.init.topology.registry",
    "resdag.init.input_feedback.registry",
    "resdag.init.utils.resolve",
    "resdag.data.io",
    "resdag.data.prepare",
    "resdag.data.dataset",
    "resdag.utils.general",
    "resdag.utils.seeds",
)


def _reference_members(page):
    """Read explicit mkdocstrings member inventories without executing examples."""
    module = None
    for line in (REFERENCE / page).read_text().splitlines():
        if line.startswith("::: "):
            module = line.removeprefix("::: ").strip()
        elif line and not line[0].isspace():
            module = None
        elif module and (match := re.fullmatch(r"\s+- (\w+)\s*", line)):
            yield module, match.group(1)


@pytest.mark.parametrize("module_name", PUBLIC_PACKAGES)
def test_public_extension_package_exports_resolve_and_complete(module_name):
    module = importlib.import_module(module_name)
    names = module.__all__
    assert len(names) == len(set(names))
    assert all(not name.startswith("_") and hasattr(module, name) for name in names)
    assert set(dir(module)) == set(names)


@pytest.mark.parametrize("module_name", PUBLIC_LEAVES)
def test_public_leaf_star_imports_exclude_implementation_dependencies(module_name):
    module = importlib.import_module(module_name)
    namespace = {}
    exec(f"from {module_name} import *", namespace)
    assert set(namespace) - {"__builtins__"} == set(module.__all__)
    assert set(module.__all__).isdisjoint(
        {"Any", "Callable", "Mapping", "Literal", "Path", "np", "torch", "inspect", "random"}
    )


def test_hpo_retains_extension_contracts_and_hides_preparation_builders():
    hpo = importlib.import_module("resdag.hpo")
    resources = importlib.import_module("resdag.hpo.resources")
    assert {
        "MemoryBudget",
        "MemoryContext",
        "MemoryEstimate",
        "Screening",
        "ScoreContext",
        "PrefixScoreContext",
        "TuningReport",
        "tune_model",
        "run_hpo",
        "get_best_params",
        "transfer_trials",
        "export_best_config",
        "TrialEvaluationError",
    }.issubset(hpo.__all__)
    assert {"MemoryContext", "estimate_memory"}.issubset(resources.__all__)
    assert set(dir(hpo)) == set(hpo.__all__)
    for name in ("memory_context", "raw_memory_context"):
        assert not hasattr(resources, name)
        assert name not in resources.__all__
    for name in ("Any", "TYPE_CHECKING", "TuningData"):
        assert not hasattr(hpo, name)


def test_hpo_study_utility_exports_include_best_params():
    pytest.importorskip("optuna")
    hpo = importlib.import_module("resdag.hpo")
    utilities = importlib.import_module("resdag.hpo.utils")
    assert "get_best_params" in utilities.__all__
    assert hpo.get_best_params is utilities.get_best_params
    namespace = {}
    exec("from resdag.hpo.utils import *", namespace)
    assert set(namespace) - {"__builtins__"} == set(utilities.__all__)


def test_custom_initializer_contracts_remain_public():
    required = {
        "resdag.init.recurrent": {
            "get_recurrent_initializer",
            "register_recurrent_initializer",
            "show_recurrent_initializers",
            "RecurrentInitializerSpec",
        },
        "resdag.init.topology": {
            "TopologyInitializer",
            "GraphTopology",
            "MatrixTopology",
            "register_graph_topology",
            "register_matrix_topology",
        },
        "resdag.init.input_feedback": {
            "InputFeedbackInitializer",
            "FunctionInitializer",
            "register_input_feedback",
        },
        "resdag.init.utils": {
            "TopologySpec",
            "InitializerSpec",
            "RecurrentInitializerSpec",
            "resolve_topology",
            "resolve_initializer",
            "resolve_recurrent_initializer",
        },
    }
    for module_name, names in required.items():
        module = importlib.import_module(module_name)
        assert names.issubset(module.__all__)
        assert all(hasattr(module, name) for name in names)


def test_data_namespaces_preserve_existing_canonical_function_identities():
    data = importlib.import_module("resdag.data")
    datasets = importlib.import_module("resdag.datasets")
    utilities = importlib.import_module("resdag.utils.data")
    for name in data.__all__:
        if hasattr(utilities, name):
            assert getattr(data, name) is getattr(utilities, name)
    for name in datasets.__all__:
        assert getattr(datasets, name) is getattr(utilities, name)


@pytest.mark.parametrize("extension", ("npy", "npz"))
def test_numpy_file_loaders_reject_pickled_object_arrays(tmp_path, extension):
    numpy = pytest.importorskip("numpy")
    data = importlib.import_module("resdag.data")
    path = tmp_path / f"objects.{extension}"
    values = numpy.array([["object payload"]], dtype=object)
    if extension == "npy":
        numpy.save(path, values)
        loader = data.load_npy
    else:
        numpy.savez(path, data=values)
        loader = data.load_npz
    with pytest.raises(ValueError, match="Object arrays.*allow_pickle=False"):
        loader(path)


@pytest.mark.parametrize("page", ("init.md", "data.md", "utils.md", "metrics.md", "artifacts.md"))
def test_reference_members_are_supported_public_exports(page):
    members = list(_reference_members(page))
    assert members
    for module_name, name in members:
        module = importlib.import_module(module_name)
        assert name in module.__all__, (page, module_name, name)
        assert hasattr(module, name), (page, module_name, name)


def test_hpo_reference_preserves_advanced_extensions_without_private_builders():
    pytest.importorskip("optuna")
    members = set(_reference_members("hpo.md"))
    assert ("resdag.hpo.resources", "MemoryContext") in members
    assert ("resdag.hpo.resources", "estimate_memory") in members
    assert ("resdag.hpo.transfer", "apply_warm_start") in members
    for module_name, name in members:
        module = importlib.import_module(module_name)
        assert name in module.__all__, (module_name, name)
        assert hasattr(module, name), (module_name, name)
