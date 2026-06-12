---
description: The complete public API of ResDAG 0.5.0, generated from source — a quick map of where every symbol lives.
---

<span class="nb-kicker">Reference</span>

# Reference

The complete public API, one page per area, generated from the source at
v0.5.0 — if a signature here disagrees with your installed version, trust
your installed version.

## Where do I find X

| You are looking for | Page |
| ------------------- | ---- |
| Everything importable straight from `resdag` | [Top level](top-level.md) |
| `ESNModel`, `Input`, `reservoir_input` | [Core](core.md) |
| Reservoir layers, cells, readouts, transforms | [Layers](layers.md) |
| Topologies, input/feedback initializers, spec resolvers | [Initialization](init.md) |
| `ESNTrainer` | [Training](training.md) |
| Premade architectures, coupled ensembles, aggregators | [Models & ensembles](models.md) |
| `run_hpo`, loss functions, study utilities | [HPO](hpo.md) |
| Data loading, splitting, RNG, ESP index | [Utilities](utils.md) |

---

## How this reference is built

Every entry is rendered by [mkdocstrings](https://mkdocstrings.github.io/)
from the NumPy-style docstrings in `src/resdag` at build time — signatures,
defaults, and type annotations come straight from the code, not from a
hand-maintained copy. Names with a leading underscore are private and
excluded. For prose-first treatments of the same machinery, start from
[Build](../build/index.md) and [Work](../workflows/index.md).
