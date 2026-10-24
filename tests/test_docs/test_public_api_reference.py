"""Source-only guards for reference coverage and supported extension contracts."""

import ast
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _tree(relative):
    return ast.parse((ROOT / relative).read_text())


def _exports(relative):
    return next(
        ast.literal_eval(node.value)
        for node in _tree(relative).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
    )


def _class(relative, name):
    return next(
        node
        for node in _tree(relative).body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def _methods(node):
    return {
        child.name: child
        for child in node.body
        if isinstance(child, ast.FunctionDef)
        and not any(
            isinstance(decorator, ast.Attribute) and decorator.attr == "setter"
            for decorator in child.decorator_list
        )
    }


def test_every_top_level_export_has_one_reference_table_entry():
    source = (ROOT / "docs/reference/top-level.md").read_text()
    rows = re.findall(r"(?m)^\| (?:\[`(\w+)`\]\[[^\]]+\]|`(\w+)`) \|", source)
    documented = [linked or plain for linked, plain in rows]
    exported = _exports("src/resdag/__init__.py")
    assert len(exported) == len(set(exported))
    assert set(documented) == set(exported)
    assert all(count == 1 for count in Counter(documented).values())


def test_core_and_layer_namespaces_have_explicit_reference_coverage():
    core = (ROOT / "docs/reference/core.md").read_text()
    layer = (ROOT / "docs/reference/layers.md").read_text()
    lsmr = (ROOT / "docs/reference/lsmr-readout.md").read_text()
    core_members = set(re.findall(r"(?m)^\s+- (\w+)\s*$", core))
    assert core_members | {"ReservoirStates"} == set(_exports("src/resdag/core/__init__.py"))
    assert "::: resdag.core.ReservoirStates" in core
    layer_members = set(re.findall(r"(?m)^\s+- (\w+)\s*$", layer))
    for namespace in ("cells", "reservoirs", "readouts", "transforms"):
        for name in _exports(f"src/resdag/layers/{namespace}/__init__.py"):
            assert name in layer_members or f"::: resdag.layers.{namespace}.{name}" in lsmr


def test_facade_public_operations_have_referenceable_docstrings():
    methods = _methods(_class("src/resdag/facade.py", "ESN"))
    protocol = {"extra_repr", "get_extra_state", "set_extra_state"}
    public = {name: method for name, method in methods.items() if not name.startswith("_")}
    assert {"fit", "forecast", "forward_stateless", "reset_state", "get_state", "set_state"} <= (
        public.keys()
    )
    assert all(ast.get_docstring(method) for name, method in public.items() if name not in protocol)


def test_cell_and_solver_extension_hooks_remain_public_and_documented():
    contracts = [
        (
            "src/resdag/layers/cells/base_cell.py",
            "ReservoirCell",
            {
                "state_size",
                "output_size",
                "init_state",
                "forward",
                "validate_state",
                "project_inputs",
                "step",
            },
        ),
        (
            "src/resdag/layers/cells/continuous_cell.py",
            "ContinuousReservoirCell",
            {"vector_field", "output_from_state"},
        ),
        (
            "src/resdag/layers/cells/spiking_cell.py",
            "SpikingReservoirCell",
            {"neuron_update", "project_inputs"},
        ),
        (
            "src/resdag/layers/readouts/base.py",
            "ReadoutLayer",
            {"fit", "solve", "forward"},
        ),
        (
            "src/resdag/layers/reservoirs/base_reservoir.py",
            "BaseReservoirLayer",
            {"forward_stateless", "step_stateless", "reset_state", "get_state", "set_state"},
        ),
    ]
    for relative, name, required in contracts:
        methods = _methods(_class(relative, name))
        assert required <= methods.keys()
        assert all(ast.get_docstring(methods[method]) for method in required)


def test_documentation_hides_torch_plumbing_without_renaming_protocol_hooks():
    configuration = (ROOT / "mkdocs.yml").read_text()
    assert "inherited_members: false" in configuration
    filters = configuration.split("            filters:\n", 1)[1].split("\n\n", 1)[0]
    patterns = re.findall(r'^\s*- "(.+)"\s*$', filters, re.M)

    def visible(name):
        shown = not name.startswith("_")
        for pattern in patterns:
            exclude = pattern.startswith("!")
            if re.search(pattern[1:] if exclude else pattern, name):
                shown = not exclude
        return shown

    for name in ("extra_repr", "get_extra_state", "set_extra_state", "_get_execution_plan"):
        assert not visible(name)
    for name in ("__init__", "forward", "fit", "forecast", "solve", "project_inputs", "step"):
        assert visible(name)
    for relative, name in (
        ("src/resdag/facade.py", "ESN"),
        ("src/resdag/core/workflow.py", "Model"),
    ):
        assert {"get_extra_state", "set_extra_state"} <= _methods(_class(relative, name)).keys()


def test_private_projection_validation_is_not_an_extra_cell_extension_hook():
    methods = _methods(_class("src/resdag/layers/cells/_projected.py", "ProjectedInputCell"))
    assert "check_step" not in methods and "_check_step" in methods
    for relative in (
        "src/resdag/layers/cells/cornn_cell.py",
        "src/resdag/layers/cells/legendre_delay_cell.py",
    ):
        attributes = {
            node.attr for node in ast.walk(_tree(relative)) if isinstance(node, ast.Attribute)
        }
        assert "check_step" not in attributes and "_check_step" in attributes
