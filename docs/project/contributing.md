---
description: Dev environment setup, the local quality gate, test-suite layout, and where new components plug into ResDAG.
---

<span class="nb-kicker">Project</span>

# Contributing

Issues and pull requests are welcome at
[github.com/El3ssar/resdag](https://github.com/El3ssar/resdag). The loop
below is everything a change needs before it ships.

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

Run all of it locally — CI currently gates only the docs build, so the test
and lint gate is on you, not the robot.

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
  as `ESNLayer(topology="name")`. See [Initialization](../build/initialization.md).
- **Input/feedback initializers** — register a function or
  `InputFeedbackInitializer` subclass with
  `@register_input_feedback("name")`.
- **Reservoir cells and layers** — subclass `ReservoirCell` for the
  single-step update, wrap it in a `BaseReservoirLayer` subclass for the
  sequence loop. See [Layers](../build/layers.md).
- **Readouts** — subclass `ReadoutLayer` and implement its fitting
  interface; `ESNTrainer` picks it up automatically.

New public symbols go in both the subpackage `__init__.py` and — if they
belong in the top-level namespace — `src/resdag/__init__.py` plus its
`__all__`. Docstrings are NumPy-style; the [Reference](../reference/index.md)
is generated from them.
