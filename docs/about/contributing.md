# Contributing

## Development setup

```bash
git clone https://github.com/El3ssar/resdag.git
cd resdag
uv sync --dev
# or: pip install -e ".[dev]"
```

## Quality checks

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

## Adding extensions

| Contribution | Guide |
|--------------|-------|
| Graph topology | [Custom topology](../extending/custom-topology.md) |
| Input initializer | [Custom initializer](../extending/custom-initializer.md) |
| Reservoir cell | [Custom cell](../extending/custom-cell.md) |
| Readout | [Custom readout](../extending/custom-readout.md) |

Register new graph/initializer modules in the corresponding `__init__.py` so
imports run at package load time.

## Pull requests

1. Branch from `main`.
2. Keep changes focused; match NumPy-style docstrings on public API.
3. Add tests under `tests/` mirroring `src/resdag/`.
4. Ensure `pytest` and `mkdocs build --strict` pass.
5. Open a PR with a short summary and test plan.

Report bugs via [GitHub Issues](https://github.com/El3ssar/resdag/issues).
