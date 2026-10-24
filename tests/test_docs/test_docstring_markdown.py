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


def _default_key(value):
    if value is None:
        return ("required",)
    source = str(value).strip().strip("`")
    try:
        literal = ast.literal_eval(source)
        if isinstance(literal, (bool, int, float, str, type(None))):
            return ("literal", type(literal).__name__, literal)
    except (ValueError, SyntaxError):
        pass
    try:
        return ("expression", ast.dump(ast.parse(source, mode="eval"), include_attributes=False))
    except SyntaxError:
        return ("text", source)


def _parameter_defaults(module, griffe):
    visited, paths, checked, problems = set(), set(), 0, []

    def visit(obj):
        nonlocal checked
        if obj.is_alias or obj.path in visited:
            return
        visited.add(obj.path)
        if obj.kind.value == "module":
            paths.add(obj.filepath.resolve())
        try:
            signature = obj.parameters
        except AttributeError:
            signature = None
        if obj.docstring and signature is not None:
            for section in obj.docstring.parsed:
                if section.kind.value not in {"parameters", "other parameters"}:
                    continue
                for parameter in section.value:
                    try:
                        actual = signature[parameter.name]
                    except KeyError:
                        continue
                    if actual.kind in (
                        griffe.ParameterKind.var_positional,
                        griffe.ParameterKind.var_keyword,
                    ):
                        continue
                    checked += 1
                    if _default_key(parameter.value) != _default_key(actual.default):
                        problems.append(
                            f"{obj.path}.{parameter.name}: signature {actual.default!r}, "
                            f"rendered {parameter.value!r}"
                        )
        if hasattr(obj, "members"):
            for member in obj.members.values():
                visit(member)

    visit(module)
    return paths, checked, problems


def test_every_runtime_numpy_parameter_default_matches_its_actual_signature():
    griffe = pytest.importorskip("griffe")
    module = griffe.load(
        "resdag",
        search_paths=[str(ROOT / "src")],
        allow_inspection=False,
        docstring_parser="numpy",
        docstring_options={"warnings": False},
    )
    paths, checked, problems = _parameter_defaults(module, griffe)
    assert paths == {path.resolve() for path in RUNTIME.rglob("*.py")}
    assert checked > 1000, "The audit must include the complete runtime reference."
    assert not problems, "\n".join(problems)


def test_separate_parameter_headers_and_prose_choices_preserve_distinct_defaults(tmp_path):
    griffe = pytest.importorskip("griffe")
    source = tmp_path / "numpy_parameter_fixture.py"
    source.write_text('''def example(alpha=0.3, beta=0.05, method=None, alignment="next"):
    """Correct NumPy metadata for different defaults and optional choices.

    Parameters
    ----------
    alpha : float, optional
        Previous-output coefficient.
    beta : float, optional
        Memory-product coefficient.
    method : str or None, optional
        Choose "minmax", "standard", or None.
    alignment : str, optional
        Choose "next" or "legacy".
    """
''')
    module = griffe.load(
        source.stem,
        search_paths=[str(tmp_path)],
        allow_inspection=False,
        docstring_parser="numpy",
    )
    _, checked, problems = _parameter_defaults(module, griffe)
    assert checked == 4
    assert not problems, "\n".join(problems)
    parameters = next(
        section.value
        for section in module["example"].docstring.parsed
        if section.kind.value == "parameters"
    )
    assert (
        str(next(parameter for parameter in parameters if parameter.name == "alignment").annotation)
        == "str"
    )


def test_data_reference_renders_each_public_member_once_without_repeated_module_bodies():
    source = (ROOT / "docs/reference/data.md").read_text()
    directives = re.findall(r"(?m)^::: (\S+)$", source)
    assert len(directives) == len(set(directives))
    assert not {"resdag.data", "resdag.datasets", "resdag.utils.data"} & set(directives)
    assert "resdag.data.prepare_esn_data" in directives
    assert "resdag.datasets.narma" in directives
