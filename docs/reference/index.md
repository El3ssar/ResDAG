---
description: The complete public API of ResDAG, with a map of where each symbol lives.
---

<span class="nb-kicker">Reference</span>

# Reference

The complete public API, one page per area. This reference documents
ResDAG {{ resdag_version }}; if a signature here disagrees with your installed version,
trust your installed version.

## Where do I find X

| You are looking for | Page |
| ------------------- | ---- |
| Everything importable straight from `resdag` | [Top level](top-level.md) |
| The `ESN` one-object estimator (`fit().forecast()`) | [Facade](facade.md) |
| `ESNModel`, `ReservoirFeatureExtractor`, `Input`, `reservoir_input` | [Core](core.md) |
| Reservoir layers, cells, readout solvers, transforms | [Layers](layers.md) |
| Topologies, input/feedback initializers, spec resolvers | [Initialization](init.md) |
| `ESNTrainer` | [Training](training.md) |
| Valid prediction time, forecast quality in Lyapunov times | [Metrics](metrics.md) |
| Premade architectures, coupled ensembles, aggregators | [Models & ensembles](models.md) |
| `run_hpo`, loss functions, study utilities | [HPO](hpo.md) |
| `TimeSeriesWindowDataset`, `make_dataloader`, streaming training | [Work · Streaming](../workflows/streaming.md) |
| Data loading, splitting, RNG, ESP index | [Utilities](utils.md) |

---

## Scope

This reference covers the public API only; names with a leading underscore
are private and not documented. For prose-first treatments of the same
components, start from [Build](../build/index.md) and
[Work](../workflows/index.md).
