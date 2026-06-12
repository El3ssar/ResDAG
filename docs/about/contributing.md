---
description: Development setup, the quality gate, and what a good ResDAG pull request looks like.
---

<span class="rd-eyebrow">About</span>

# Contributing

Two commands to a working dev environment, four to a green quality gate.
Contributions of all sizes are welcome — topologies and initializers
especially, since that's where reservoir research happens.

## Development setup

```bash
git clone https://github.com/El3ssar/resdag.git
cd resdag
uv sync --extra dev
# or: pip install -e ".[dev]"
```

## Quality checks

Run all four before opening a PR (CI currently gates only the docs build):

```bash
pytest
ruff check src/ tests/
black src/ tests/
mypy src/
```

## Documentation

```bash
pip install -e ".[docs]"
mkdocs serve
mkdocs build --strict
```

## Adding components

New topologies, initializers, cells, and readouts each slot into one
directory — and, for topologies and initializers, one registry — the
[custom components](../cookbook/custom-components.md) recipe walks through
all four. One rule to remember: register new graph/initializer modules in
the corresponding `__init__.py` so registration runs at package load time.

## Pull requests

1. Branch from `main`.
2. Keep changes focused; match NumPy-style docstrings on public API.
3. Add tests under `tests/` mirroring `src/resdag/`.
4. Ensure `pytest` and `mkdocs build --strict` pass.
5. Open a PR with a short summary and test plan.

Bugs go to [GitHub Issues](https://github.com/El3ssar/resdag/issues).
