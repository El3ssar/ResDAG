# API reference

Auto-generated from NumPy-style docstrings in `src/resdag/`. Use the sections below
or the search bar to jump to a symbol.

## Models & training

| Topic | Page |
|-------|------|
| Package exports | [Top-level API](top-level.md) |
| `ESNModel`, forecasting, persistence | [Core](core.md) |
| Premade architectures | [Premade models](models.md) |
| `ESNTrainer` | [Training](training.md) |
| Coupled ensembles | [Ensemble](ensemble.md) |

## Layers

| Topic | Page |
|-------|------|
| `ESNLayer`, `NGReservoir` | [Reservoirs](layers/reservoirs.md) |
| `ESNCell`, `NGCell` | [Cells](layers/cells.md) |
| `CGReadoutLayer` | [Readouts](layers/readouts.md) |
| Augmentation & concat | [Transforms](layers/transforms.md) |

## Initialization

| Topic | Page |
|-------|------|
| Topology registry | [Topology](init/topology.md) |
| Input/feedback weights | [Input & feedback](init/input-feedback.md) |
| Graph generators (17) | [Graphs](init/graphs.md) |
| `resolve_topology`, `resolve_initializer` | [Resolvers](init/resolvers.md) |

## Tooling

| Topic | Page |
|-------|------|
| `run_hpo`, objectives | [HPO — run](hpo/run.md) |
| Loss functions | [HPO — losses](hpo/losses.md) |
| Storage & runners | [HPO — internals](hpo/internals.md) |
| Study utilities | [HPO — utils](hpo/utils.md) |
| Data I/O & splits | [Utils — data](utils/data.md) |
| `esp_index` | [Utils — states](utils/states.md) |
