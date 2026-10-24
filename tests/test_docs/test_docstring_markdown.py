"""Runtime API documentation uses the Markdown consumed by mkdocstrings."""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "src/resdag"
REST = re.compile(r":[\w:]+:`|(?m:^\s*\.\.\s+(?:[a-z][\w-]*::|\[\d+\]))")


def test_runtime_docstrings_and_attribute_documentation_use_markdown():
    problems = []
    for path in sorted(RUNTIME.rglob("*.py")):
        source = path.read_text()
        for node in ast.walk(ast.parse(source)):
            if not isinstance(
                node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                continue
            docstring = ast.get_docstring(node, clean=False)
            if docstring and REST.search(docstring):
                problems.append(f"{path.relative_to(ROOT)}:{node.body[0].lineno}")
        comments = "\n".join(re.findall(r"(?m)^\s*#: ?(.*)$", source))
        if REST.search(comments):
            problems.append(f"{path.relative_to(ROOT)}:attribute documentation")
    assert not problems, "Unrendered reStructuredText: " + ", ".join(problems)


def test_numpy_parser_uses_real_generator_defaults_in_parameter_tables():
    griffe = pytest.importorskip("griffe")
    module = griffe.load(
        "resdag.utils.data.datasets",
        search_paths=[str(ROOT / "src")],
        docstring_parser="numpy",
    )
    for name in (
        "lorenz",
        "rossler",
        "henon",
        "mackey_glass",
        "narma",
        "sine",
        "kuramoto_sivashinsky",
    ):
        function = module[name]
        signature_default = function.parameters["normalize"].default
        parameters = next(
            section.value
            for section in function.docstring.parsed
            if section.kind.value == "parameters"
        )
        normalize = next(parameter for parameter in parameters if parameter.name == "normalize")
        assert normalize.value == signature_default == "None", name
        assert all(choice in normalize.description for choice in ('"standard"', '"minmax"', "None"))


def test_data_reference_renders_each_public_member_once_without_repeated_module_bodies():
    source = (ROOT / "docs/reference/data.md").read_text()
    directives = re.findall(r"(?m)^::: (\S+)$", source)
    assert len(directives) == len(set(directives))
    assert not {"resdag.data", "resdag.datasets", "resdag.utils.data"} & set(directives)
    assert "resdag.data.prepare_esn_data" in directives
    assert "resdag.datasets.narma" in directives
