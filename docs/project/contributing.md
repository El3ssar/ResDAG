---
description: Dev environment setup, the local quality gate, test-suite layout, and where new components and their documentation plug into ResDAG.
---

<span class="nb-kicker">Project</span>

# Contributing

Issues and pull requests are welcome at
[github.com/El3ssar/resdag](https://github.com/El3ssar/resdag). The workflow
below covers everything a change needs before it is merged.

## Setup

```bash
git clone https://github.com/El3ssar/resdag && cd resdag
uv sync --extra dev          # or: pip install -e ".[dev]"
```

The dev extra brings in pytest, ruff, black, mypy, and optuna (so the HPO
tests run).

## Quality gate

```bash
uv run pytest                        # full suite with coverage
uv run pytest -m benchmark --no-cov  # performance assertions, deselected by default
uv run ruff check src/ tests/
uv run black --check src/ tests/
uv run mypy src/
```

Run all of it locally before opening a pull request — CI currently builds
only the docs, so tests and lint are not enforced remotely.

The suite in `tests/` mirrors `src/resdag/` (`test_layers/`, `test_models/`,
…), shares fixtures through `tests/conftest.py`, and uses three markers:
`gpu` (auto-skipped without CUDA), `slow` (always runs; deselect with
`-m "not slow"`), and `benchmark` (deselected by default).

---

## Adding components

The library is built to be extended without touching its internals:

- **Topologies** — register a graph generator with
  `@register_graph_topology("name", **defaults)` or a matrix builder with
  `@register_matrix_topology("name", **defaults)`, and it becomes available
  as `ESNLayer(topology="name")`. See [Initialization](../build/initialization/index.md).
- **Input/feedback initializers** — register a function or
  `InputFeedbackInitializer` subclass with
  `@register_input_feedback("name")`.
- **Reservoir cells and layers** — subclass `ReservoirCell` for the
  single-step update, wrap it in a `BaseReservoirLayer` subclass for the
  sequence loop. See [Layers](../build/layers/index.md).
- **Readouts** — subclass `ReadoutLayer` and implement its fitting
  interface; `ESNTrainer` picks it up automatically. See
  [Readouts](../build/readouts/index.md).

New public symbols go in both the subpackage `__init__.py` and — if they
belong in the top-level namespace — `src/resdag/__init__.py` plus its
`__all__`. Docstrings are NumPy-style; the [Reference](../reference/index.md)
is generated from them.

---

## Documentation

The site is built with MkDocs (`mkdocs serve` for a live preview,
`mkdocs build --strict` for the production build). It is modular, so most
component additions need no nav or index edits:

- **Topology and initializer pages** are generated at build time from the
  live registries by `hooks/docs_autogen.py` — docstring summary, parameter
  table, and gallery figure. Registering a new component is all it takes
  for its page to appear under Topologies or Initializers.
- **Layer, readout, and architecture pages** are handwritten, one Markdown
  file per component. Drop the file into the matching folder under
  `docs/build/` and it is picked up by the nav and by the section index's
  card grid; the card text comes from the page's `description` frontmatter.
- **Figures** under `docs/assets/figures/` are regenerated with
  `uv run python scripts/generate_docs_figures.py` (use `--only` to rebuild
  a subset). Do not edit the images by hand.

## Releases

Releases are fully automated — no manual tags, no manual version edits.
Every push to `main` runs the test gate, then
[python-semantic-release](https://python-semantic-release.readthedocs.io/)
reads the commit messages since the last release and acts accordingly:

| commit message starts with | effect |
| --- | --- |
| `feat: ...` | minor bump (0.5.0 → 0.6.0), release |
| `fix: ...` or `perf: ...` | patch bump (0.5.0 → 0.5.1), release |
| `feat!: ...` or a `BREAKING CHANGE:` footer | minor bump while 0.x, major from 1.0 |
| `docs:`, `chore:`, `ci:`, `refactor:`, `test:`, `style:`, `build:` | no release |

A release rewrites `__version__` in `src/resdag/__init__.py` (the only
place the version lives), commits, tags `vX.Y.Z`, publishes a GitHub
release with generated notes, and uploads the build to PyPI via trusted
publishing. The documentation deploys on the same push through the docs
workflow, and version strings on these pages substitute automatically.

Practical rule: merge pull requests with **squash merge** and give the
squash commit a conventional title — that title alone decides the bump.
