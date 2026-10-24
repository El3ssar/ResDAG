---
title: "Reference"
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
| `Model`, `Feedback`, state and pipeline adapters, symbolic input helpers | [Core](core.md) |
| Complete model artifacts, codecs and custom component registration | [Model artifacts](artifacts.md) |
| Reservoir layers, cells, readout solvers, transforms | [Layers](layers.md) |
| Topologies, input/feedback initializers, spec resolvers | [Initialization](init.md) |
| `ESNTrainer` | [Training](training.md) |
| Valid prediction time, forecast quality in Lyapunov times | [Metrics](metrics.md) |
| Premade architectures, coupled ensembles, aggregators | [Models & ensembles](models.md) |
| `run_hpo`, loss functions, study utilities | [HPO](hpo.md) |
| `TimeSeriesWindowDataset`, `make_dataloader`, streaming training | [Work · Streaming](../workflows/streaming.md) |
| Data preparation, trajectory alignment, file loading and generators | [Data and datasets](data.md) |
| RNG streams, device selection and ESP index | [Utilities](utils.md) |

---

## Scope

Public package exports are declared in `__all__` and shown by `dir()`. The
reference covers normal model use and supported extension contracts, including
custom cells, initializers, readouts and artifact codecs. Execution planners and
fit-preparation helpers with a leading underscore are private. Torch's required
checkpoint/representation hooks remain operational but are omitted from ordinary
method listings. For prose-first treatments of the same
components, start from [Build](../build/index.md) and
[Work](../workflows/index.md).
