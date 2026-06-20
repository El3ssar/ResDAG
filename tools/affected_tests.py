#!/usr/bin/env python3
"""Static, dependency-aware test selector for ResDAG.

Given the set of files changed by a branch (relative to ``main``), this tool
prints the subset of the pytest suite that *could* be affected by the change —
so CI (and agents verifying their own work) can run a few seconds of tests
instead of the full ~4-minute suite.

How it decides what to run
--------------------------
The naive approach — "run the tests that import the changed module" — does not
work here, because ``resdag/__init__.py`` (and the sub-package ``__init__``
files) eagerly re-export the whole API, so almost every test transitively
imports almost everything. Instead this tool reasons at the level of the real
*first-party import graph*:

1. **Resolve re-exports.** Each ``__init__.py`` is parsed so that
   ``from resdag.layers import CGReadoutLayer`` resolves to the *defining*
   module (``resdag.layers.readouts.cg_readout``), not "all of ``layers``".
2. **Source import graph.** Every module under ``src/resdag`` is parsed for its
   first-party imports, giving a module-to-module dependency graph. A change to
   module *M* impacts *M* plus everything that (transitively) imports it
   (reverse reachability).
3. **Per-test association.** Each test file is associated with the source
   modules it imports (re-exports resolved) *plus* the modules constructed by
   any shared ``conftest`` fixtures it uses (so a test that builds a reservoir
   only through ``make_tiny_model`` is still tied to the reservoir code).
4. **Selection.** A test runs if it was itself changed, or if its associated
   modules intersect the impacted set.

Fail-safe behaviour
-------------------
The tool errs toward running *more* tests, never fewer:

* A change to a "broad-impact" file (``pyproject.toml``, the root ``conftest``,
  ``src/resdag/__init__.py``, a CI workflow, this selector itself, ...) selects
  the **whole** suite.
* Any import it cannot statically resolve falls back to the whole containing
  package subtree.
* Any unexpected error makes ``main()`` exit non-zero, and the CI workflow
  treats that as "run everything".

The full suite still runs on every push to ``main`` and on the nightly
schedule, so selection only ever trades a little extra latency-safety on PRs
for a lot of speed — it is never the last line of defence before a release.

Usage
-----
::

    # What would run for the current branch vs. origin/main?
    python tools/affected_tests.py --explain

    # Feed the selection straight to pytest (the one-liner agents should use):
    pytest --no-cov -q $(python tools/affected_tests.py --format args)

    # Machine-readable, for CI:
    python tools/affected_tests.py --base "$BASE_SHA" --format json
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
PACKAGE_DIR = SRC_ROOT / "resdag"
TESTS_DIR = REPO_ROOT / "tests"
TOP_PACKAGE = "resdag"

# Sentinel test target meaning "run the entire suite".
FULL_SUITE_TARGET = "tests"

# Files whose change can affect arbitrary tests in ways the import graph cannot
# localise — a change to any of these selects the whole suite. Paths are
# repo-relative and matched exactly; the callables match by predicate.
_BROAD_IMPACT_EXACT = frozenset(
    {
        "pyproject.toml",
        "uv.lock",
        "setup.py",
        "setup.cfg",
        "MANIFEST.in",
        "tox.ini",
        # Shared test infrastructure and the public-API / version surface.
        "tests/__init__.py",
        "src/resdag/__init__.py",
        # This selector and its own tests gate everything when they change.
        "tools/affected_tests.py",
        # Workflows that run the test suite.
        ".github/workflows/ci.yml",
        ".github/workflows/release.yml",
    }
)


def _is_broad_impact(rel_path: str) -> bool:
    """Return True if changing ``rel_path`` should trigger the full suite."""
    if rel_path in _BROAD_IMPACT_EXACT:
        return True
    # Any conftest.py (root or per-directory) reshapes fixtures for its subtree.
    if rel_path == "conftest.py" or rel_path.endswith("/conftest.py"):
        return True
    return False


# --------------------------------------------------------------------------- #
# Module bookkeeping
# --------------------------------------------------------------------------- #


def _module_name(path: Path) -> str | None:
    """Dotted ``resdag.*`` module name for a source file, or None if outside."""
    try:
        rel = path.resolve().relative_to(PACKAGE_DIR)
    except ValueError:
        return None
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    elif parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    else:
        return None
    return ".".join([TOP_PACKAGE, *parts]) if parts else TOP_PACKAGE


@dataclass(frozen=True)
class ModuleIndex:
    """Maps dotted module names to files and records which are packages."""

    files: dict[str, Path]
    packages: frozenset[str]

    def __contains__(self, module: str) -> bool:
        return module in self.files

    def is_package(self, module: str) -> bool:
        return module in self.packages

    def subtree(self, module: str) -> set[str]:
        """``module`` plus every module nested beneath it."""
        prefix = module + "."
        return {m for m in self.files if m == module or m.startswith(prefix)}


def build_module_index() -> ModuleIndex:
    """Index every ``resdag`` module under ``src/``."""
    files: dict[str, Path] = {}
    packages: set[str] = set()
    for path in PACKAGE_DIR.rglob("*.py"):
        name = _module_name(path)
        if name is None:
            continue
        files[name] = path
        if path.name == "__init__.py":
            packages.add(name)
    return ModuleIndex(files=files, packages=frozenset(packages))


# --------------------------------------------------------------------------- #
# Import parsing & re-export resolution
# --------------------------------------------------------------------------- #


@dataclass
class ImportRef:
    """A single ``import`` / ``from ... import ...`` resolved to an absolute
    ``resdag`` target module plus the names it pulls (``None`` for a whole
    ``import module`` and ``["*"]`` for a star import)."""

    module: str
    names: list[str] | None


def _resolve_relative(module: str | None, level: int, pkg_parts: list[str]) -> str | None:
    """Resolve a relative import to an absolute dotted name.

    ``pkg_parts`` is the package of the *importing* module (the module's parent
    package parts, e.g. ``["resdag", "layers", "reservoirs"]`` for
    ``resdag.layers.reservoirs.esn``).
    """
    if level <= 0:
        return module
    base = pkg_parts[: len(pkg_parts) - (level - 1)] if level - 1 <= len(pkg_parts) else []
    if not base:
        return module
    if module:
        return ".".join([*base, *module.split(".")])
    return ".".join(base)


def _parse_import_refs(path: Path, module_name: str | None) -> list[ImportRef]:
    """Extract first-party (``resdag.*``) import references from a file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, OSError):
        return []
    # The importing module's package parts, for relative-import resolution.
    if module_name is None:
        pkg_parts: list[str] = []
    else:
        parts = module_name.split(".")
        # A package's own files resolve relative imports against the package
        # itself; a regular module resolves against its parent package.
        pkg_parts = parts if path.name == "__init__.py" else parts[:-1]

    refs: list[ImportRef] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == TOP_PACKAGE or alias.name.startswith(TOP_PACKAGE + "."):
                    refs.append(ImportRef(module=alias.name, names=None))
        elif isinstance(node, ast.ImportFrom):
            target = _resolve_relative(node.module, node.level, pkg_parts)
            if not target:
                continue
            if target == TOP_PACKAGE or target.startswith(TOP_PACKAGE + "."):
                names = (
                    ["*"]
                    if any(a.name == "*" for a in node.names)
                    else [a.name for a in node.names]
                )
                refs.append(ImportRef(module=target, names=names))
    return refs


class ReExportResolver:
    """Resolves ``from <pkg> import <name>`` to the defining leaf module."""

    def __init__(self, index: ModuleIndex) -> None:
        self.index = index
        # origin[pkg][name] = module the name is imported *from* in pkg/__init__
        self.origin: dict[str, dict[str, str]] = {}
        self.star: dict[str, set[str]] = {}
        for pkg in index.packages:
            self._index_package(pkg)

    def _index_package(self, pkg: str) -> None:
        origin: dict[str, str] = {}
        star: set[str] = set()
        for ref in _parse_import_refs(self.index.files[pkg], pkg):
            if ref.names is None:
                # `import resdag.x.y` inside an __init__ — bind the leaf name.
                leaf = ref.module.rsplit(".", 1)[-1]
                origin[leaf] = ref.module
            elif ref.names == ["*"]:
                star.add(ref.module)
            else:
                for name in ref.names:
                    # `from .data import x` -> x lives in resdag...data; but
                    # `from . import data` re-exports the *submodule* data.
                    submod = f"{ref.module}.{name}"
                    origin[name] = submod if submod in self.index else ref.module
        self.origin[pkg] = origin
        if star:
            self.star[pkg] = star

    def resolve_symbol(self, pkg: str, name: str) -> set[str]:
        """Resolve ``name`` exported by ``pkg`` to defining module(s).

        Follows chained re-exports through nested ``__init__`` files. Returns
        the containing package subtree when the name cannot be resolved (so the
        caller conservatively depends on everything under it).
        """
        seen: set[tuple[str, str]] = set()
        cur = pkg
        while True:
            if (cur, name) in seen:
                break
            seen.add((cur, name))
            origin = self.origin.get(cur, {})
            if name in origin:
                target = origin[name]
                if self.index.is_package(target) and name in self.origin.get(target, {}):
                    cur = target
                    continue
                if self.index.is_package(target):
                    # Re-exported as a sub-thing of a package we can't pinpoint.
                    sub = f"{target}.{name}"
                    return {sub} if sub in self.index else self.index.subtree(target)
                return {target}
            break
        # Unresolved. A submodule named after `name`?
        sub = f"{pkg}.{name}"
        if sub in self.index:
            return {sub}
        # Star re-export or genuinely unknown — depend on the whole package.
        if pkg in self.index:
            return self.index.subtree(pkg)
        return set()

    def resolve_ref(self, ref: ImportRef) -> set[str]:
        """Resolve an :class:`ImportRef` to a set of concrete source modules."""
        mod = ref.module
        # A direct module/file import: depend on it (names are attributes of it).
        if mod in self.index and not self.index.is_package(mod):
            return {mod}
        if self.index.is_package(mod):
            if ref.names is None or ref.names == ["*"]:
                return self.index.subtree(mod)
            out: set[str] = set()
            for name in ref.names:
                sub = f"{mod}.{name}"
                if sub in self.index:
                    out.add(sub)  # submodule import, e.g. `from resdag.utils import data`
                else:
                    out |= self.resolve_symbol(mod, name)
            return out
        # `mod` is not itself indexed (e.g. resdag.x.y.attr): walk up to the
        # longest existing module/package prefix.
        parts = mod.split(".")
        for i in range(len(parts) - 1, 0, -1):
            prefix = ".".join(parts[:i])
            if prefix in self.index:
                if self.index.is_package(prefix):
                    return self.index.subtree(prefix)
                return {prefix}
        return set()

    def resolve_alias_usage(self, tree: ast.AST, alias: str) -> set[str]:
        """Resolve attribute chains rooted at a bare ``import resdag as alias``.

        ``alias.lorenz`` -> the datasets module; ``alias.utils.prepare_esn_data``
        -> the prepare module. If the alias is ever used as a plain value
        (passed, reassigned, ``getattr``-ed) we cannot reason about it, so we
        conservatively depend on the entire package.
        """
        out: set[str] = set()
        # An attribute chain like ``rd.utils.prepare_esn_data`` contains nested
        # Attribute nodes; only the outermost should be resolved (the inner
        # ``rd.utils`` is a partial prefix, not a real dependency).
        inner_links = {
            id(n.value)
            for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Attribute)
        }
        chain_name_nodes = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or id(node) in inner_links:
                continue
            attrs: list[str] = []
            cur: ast.expr = node
            while isinstance(cur, ast.Attribute):
                attrs.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name) and cur.id == alias:
                chain_name_nodes += 1
                attrs.reverse()  # outermost-first: e.g. ["utils", "prepare_esn_data"]
                out |= self._resolve_attr_chain(attrs)
        total_name_nodes = sum(
            1 for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == alias
        )
        if total_name_nodes > chain_name_nodes:
            # Bare/dynamic use of the alias — be safe.
            return self.index.subtree(TOP_PACKAGE)
        return out

    def _resolve_attr_chain(self, attrs: list[str]) -> set[str]:
        if not attrs:
            return self.index.subtree(TOP_PACKAGE)
        first, rest = attrs[0], attrs[1:]
        top_sub = f"{TOP_PACKAGE}.{first}"
        if top_sub in self.index:
            if self.index.is_package(top_sub) and rest:
                return self.resolve_ref(ImportRef(module=top_sub, names=[rest[0]]))
            if self.index.is_package(top_sub):
                return self.index.subtree(top_sub)
            return {top_sub}
        return self.resolve_symbol(TOP_PACKAGE, first)


# --------------------------------------------------------------------------- #
# Dependency graph & impact
# --------------------------------------------------------------------------- #


@dataclass
class DependencyGraph:
    index: ModuleIndex
    resolver: ReExportResolver
    deps: dict[str, set[str]] = field(default_factory=dict)
    reverse: dict[str, set[str]] = field(default_factory=dict)

    def impacted(self, changed_modules: set[str]) -> set[str]:
        """All modules reachable from ``changed_modules`` via reverse edges."""
        out: set[str] = set()
        stack = list(changed_modules)
        while stack:
            m = stack.pop()
            if m in out:
                continue
            out.add(m)
            stack.extend(self.reverse.get(m, ()))
        return out


# Registry-backed providers are dispatched by *string* name at runtime
# (``topology="erdos_renyi"``, ``feedback_initializer="chebyshev"``, ...), so a
# static import graph cannot see which consumer uses which provider. We model
# the worst case by making the resolver depend on EVERY provider: a change to
# any provider then impacts everything that resolves a topology / initializer /
# matrix (i.e. all reservoir construction), never under-selecting. These files
# change rarely, so the extra breadth costs little.
_REGISTRY_PROVIDER_PREFIXES = (
    "resdag.init.graphs",
    "resdag.init.input_feedback",
    "resdag.init.matrices",
)
_REGISTRY_ANCHOR = "resdag.init.utils.resolve"


def _is_registry_provider(module: str) -> bool:
    return any(module == p or module.startswith(p + ".") for p in _REGISTRY_PROVIDER_PREFIXES)


def build_graph(index: ModuleIndex, resolver: ReExportResolver) -> DependencyGraph:
    deps: dict[str, set[str]] = {}
    reverse: dict[str, set[str]] = {m: set() for m in index.files}
    for module, path in index.files.items():
        targets: set[str] = set()
        for ref in _parse_import_refs(path, module):
            targets |= resolver.resolve_ref(ref)
        targets.discard(module)
        deps[module] = targets
        for t in targets:
            reverse.setdefault(t, set()).add(module)

    # Synthetic edges for string-dispatched registries (see note above).
    if _REGISTRY_ANCHOR in index.files:
        for module in index.files:
            if _is_registry_provider(module):
                deps[_REGISTRY_ANCHOR].add(module)
                reverse.setdefault(module, set()).add(_REGISTRY_ANCHOR)

    return DependencyGraph(index=index, resolver=resolver, deps=deps, reverse=reverse)


# --------------------------------------------------------------------------- #
# conftest fixtures -> source modules
# --------------------------------------------------------------------------- #


def _conftest_files() -> list[Path]:
    return sorted(TESTS_DIR.rglob("conftest.py"))


def build_fixture_closures(resolver: ReExportResolver) -> dict[str, set[str]]:
    """Map each shared fixture name to the source modules it constructs.

    A fixture that builds an ``ESNLayer`` ties every test that *uses* it to the
    reservoir source, even when the test never imports the reservoir directly.
    """
    closures: dict[str, set[str]] = {}
    for conftest in _conftest_files():
        try:
            tree = ast.parse(conftest.read_text(encoding="utf-8"), filename=str(conftest))
        except (SyntaxError, OSError):
            continue
        name_to_modules = _imported_name_map(conftest, resolver)
        aliases = _facade_aliases(tree)

        # First pass: direct module references in each fixture body.
        direct: dict[str, set[str]] = {}
        fixture_params: dict[str, set[str]] = {}
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not _is_fixture(node):
                continue
            mods: set[str] = set()
            for name in _names_used(node):
                mods |= name_to_modules.get(name, set())
            for alias in aliases:
                mods |= resolver.resolve_alias_usage(node, alias)
            direct[node.name] = mods
            fixture_params[node.name] = {a.arg for a in node.args.args}

        # Second pass: fold in fixtures referenced as parameters (fixtures that
        # build on other fixtures inherit their closures).
        for fname in direct:
            closure = set(direct[fname])
            stack = list(fixture_params.get(fname, ()))
            seen: set[str] = set()
            while stack:
                dep = stack.pop()
                if dep in seen or dep not in direct:
                    continue
                seen.add(dep)
                closure |= direct[dep]
                stack.extend(fixture_params.get(dep, ()))
            closures[fname] = closure
    return closures


def _is_fixture(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        # Matches @fixture, @pytest.fixture, @pytest.fixture(...)
        if isinstance(target, ast.Attribute) and target.attr == "fixture":
            return True
        if isinstance(target, ast.Name) and target.id == "fixture":
            return True
    return False


def _names_used(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _facade_aliases(tree: ast.AST) -> set[str]:
    """Names bound to a bare ``import resdag`` (e.g. ``rd``)."""
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == TOP_PACKAGE:
                    aliases.add(a.asname or a.name)
    return aliases


def _imported_name_map(path: Path, resolver: ReExportResolver) -> dict[str, set[str]]:
    """Map each name a file imports from resdag to the module(s) it resolves to."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, OSError):
        return {}
    out: dict[str, set[str]] = {}
    pkg_parts: list[str] = []  # tests / conftest are not resdag packages
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            target = _resolve_relative(node.module, node.level, pkg_parts)
            if not target or not (target == TOP_PACKAGE or target.startswith(TOP_PACKAGE + ".")):
                continue
            for a in node.names:
                if a.name == "*":
                    continue
                bound = a.asname or a.name
                out.setdefault(bound, set())
                out[bound] |= resolver.resolve_ref(ImportRef(module=target, names=[a.name]))
    return out


# --------------------------------------------------------------------------- #
# Test association & selection
# --------------------------------------------------------------------------- #


def _iter_test_files() -> list[Path]:
    return sorted(p for p in TESTS_DIR.rglob("test_*.py"))


def associate_test(
    path: Path,
    resolver: ReExportResolver,
    fixture_closures: dict[str, set[str]],
) -> set[str]:
    """Source modules a single test file depends on."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, OSError):
        # Unparseable test: be safe and tie it to everything.
        return resolver.index.subtree(TOP_PACKAGE)

    modules: set[str] = set()
    for ref in _parse_import_refs(path, None):
        if ref.module == TOP_PACKAGE and ref.names is None:
            continue  # bare `import resdag`; handled via alias analysis below
        modules |= resolver.resolve_ref(ref)
    for alias in _facade_aliases(tree):
        modules |= resolver.resolve_alias_usage(tree, alias)

    # Fixtures used as test-function parameters pull in what they construct.
    used_fixtures = _fixtures_used(tree, set(fixture_closures))
    for fx in used_fixtures:
        modules |= fixture_closures.get(fx, set())
    return modules


def _fixtures_used(tree: ast.AST, fixture_names: set[str]) -> set[str]:
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in node.args.args:
                if arg.arg in fixture_names:
                    used.add(arg.arg)
    return used


@dataclass
class Selection:
    mode: str  # "full" | "selected" | "none"
    tests: list[str]  # repo-relative test paths (or ["tests"] for full)
    reason: str
    changed_files: list[str]


def select_tests(changed_files: list[str]) -> Selection:
    """Core entry point: changed files -> tests to run."""
    changed = sorted(set(changed_files))

    broad = [f for f in changed if _is_broad_impact(f)]
    if broad:
        return Selection(
            mode="full",
            tests=[FULL_SUITE_TARGET],
            reason=f"broad-impact change: {', '.join(broad)}",
            changed_files=changed,
        )

    index = build_module_index()
    resolver = ReExportResolver(index)
    graph = build_graph(index, resolver)
    fixture_closures = build_fixture_closures(resolver)

    # Changed source modules (expand changed package __init__ to its subtree).
    changed_modules: set[str] = set()
    for f in changed:
        p = REPO_ROOT / f
        mod = _module_name(p)
        if mod is None:
            continue
        if index.is_package(mod):
            changed_modules |= index.subtree(mod)
        elif mod in index:
            changed_modules.add(mod)
    impacted = graph.impacted(changed_modules)

    changed_tests = {
        f for f in changed if f.startswith("tests/") and Path(f).name.startswith("test_")
    }

    selected: set[str] = set(changed_tests)
    for test_path in _iter_test_files():
        rel = test_path.resolve().relative_to(REPO_ROOT).as_posix()
        if rel in selected:
            continue
        assoc = associate_test(test_path, resolver, fixture_closures)
        if assoc & impacted:
            selected.add(rel)

    if not selected:
        reason = (
            "no source or test files changed" if not impacted else "no tests depend on the change"
        )
        return Selection(mode="none", tests=[], reason=reason, changed_files=changed)

    n_src = len(changed_modules)
    return Selection(
        mode="selected",
        tests=sorted(selected),
        reason=f"{len(selected)} test file(s) affected by {n_src} changed module(s)",
        changed_files=changed,
    )


# --------------------------------------------------------------------------- #
# Git plumbing & CLI
# --------------------------------------------------------------------------- #


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


@lru_cache(maxsize=1)
def _default_base() -> str:
    for ref in ("origin/main", "main"):
        try:
            _git("rev-parse", "--verify", "--quiet", ref)
            return ref
        except subprocess.CalledProcessError:
            continue
    return "HEAD"


def changed_files_from_git(base: str | None, include_dirty: bool) -> list[str]:
    """Repo-relative paths changed on this branch relative to ``base``."""
    base = base or _default_base()
    files: set[str] = set()
    # Committed changes since the merge-base (three-dot).
    diff_spec = f"{base}...HEAD" if base != "HEAD" else "HEAD"
    try:
        out = _git("diff", "--name-only", "--diff-filter=d", diff_spec)
        files.update(line.strip() for line in out.splitlines() if line.strip())
    except subprocess.CalledProcessError:
        pass
    if include_dirty:
        # Unstaged + staged + untracked working-tree changes (local use).
        for extra in (
            _git("diff", "--name-only", "--diff-filter=d"),
            _git("diff", "--name-only", "--diff-filter=d", "--cached"),
            _git("ls-files", "--others", "--exclude-standard"),
        ):
            files.update(line.strip() for line in extra.splitlines() if line.strip())
    return sorted(files)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--base",
        default=None,
        help="Base ref/SHA to diff against (default: origin/main, then main).",
    )
    parser.add_argument(
        "--changed",
        nargs="*",
        default=None,
        help="Explicit list of changed files (skips git; mainly for testing/CI).",
    )
    parser.add_argument(
        "--no-dirty",
        action="store_true",
        help="Ignore uncommitted working-tree changes (CI uses this).",
    )
    parser.add_argument(
        "--format",
        choices=("args", "lines", "json"),
        default="lines",
        help="Output format. 'args': one space-separated line for pytest. "
        "'lines': one path per line. 'json': structured result.",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Print a human-readable summary to stderr.",
    )
    args = parser.parse_args(argv)

    if args.changed is not None:
        changed = sorted(set(args.changed))
    else:
        changed = changed_files_from_git(args.base, include_dirty=not args.no_dirty)

    selection = select_tests(changed)

    if args.explain:
        print(
            f"[affected-tests] mode={selection.mode} :: {selection.reason}",
            file=sys.stderr,
        )
        if selection.mode == "selected":
            for t in selection.tests:
                print(f"  + {t}", file=sys.stderr)

    if args.format == "json":
        print(
            json.dumps(
                {
                    "mode": selection.mode,
                    "tests": selection.tests,
                    "reason": selection.reason,
                    "changed_files": selection.changed_files,
                },
                indent=2,
            )
        )
    elif args.format == "args":
        print(" ".join(selection.tests))
    else:  # lines
        for t in selection.tests:
            print(t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
