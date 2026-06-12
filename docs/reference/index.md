---
description: Auto-generated API reference for every public class and function in resdag, mapped by task.
---

<span class="rd-eyebrow">Reference</span>

# API reference

Every page in this section is generated straight from the NumPy-style
docstrings in `src/resdag/` — signatures, parameters, shapes, and examples
come from the source, so they can't drift from the code. Use the map below
to land on the right page first try.

## Where do I find X

| I want… | Go to |
|---------|-------|
| The layer classes (`ESNLayer`, `NGReservoir`, `CGReadoutLayer`, transforms) | [Reservoirs](layers/reservoirs.md) · [Cells](layers/cells.md) · [Readouts](layers/readouts.md) · [Transforms](layers/transforms.md) |
| Model building, `warmup()`, `forecast()`, save/load | [Core — `ESNModel`](core.md) |
| Topology and initializer registries | [Topology](init/topology.md) · [Input/feedback](init/input-feedback.md) · [Graphs](init/graphs.md) · [Resolvers](init/resolvers.md) |
| The trainer (`ESNTrainer.fit`) | [Training](training.md) |
| Premade factories (`ott_esn`, `classic_esn`, …) | [Premade models](models.md) |
| Hyperparameter optimization (`run_hpo`, losses) | [HPO](hpo/run.md) · [Losses](hpo/losses.md) · [Internals](hpo/internals.md) · [Utils](hpo/utils.md) |
| Coupled ensemble forecasting | [Ensemble](ensemble.md) |
| Data loading and splits, ESP diagnostics | [Data](utils/data.md) · [States](utils/states.md) |
| What `import resdag` actually exports | [Top-level exports](top-level.md) |

Looking for *how* rather than *what*? The [course](../learn/index.md) teaches
the workflow and the [cookbook](../cookbook/index.md) solves one problem per
page — the reference is for when you know the name and need the signature.
