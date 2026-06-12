---
description: Twelve self-contained ResDAG recipes — one problem per page, copy-paste ready.
---

<span class="rd-eyebrow">Cookbook</span>

# Cookbook

One problem per page, copy-paste ready. Every recipe is a self-contained
solution built on the real API — skim the code, take what you need, follow
the Related links when a recipe touches deeper machinery. If you haven't
trained a first model yet, start with the
[quickstart](../learn/quickstart.md).

<div class="grid cards" markdown>

- **[Topologies](topologies.md)**

    ---

    Wire the recurrent matrix: 17 graph topologies, the built-in
    `"orthogonal"` builder, or any function you can write.

- **[Input & feedback initializers](initializers.md)**

    ---

    Shape how signals enter the reservoir — 11 built-ins, plain
    functions, or `torch.nn.init` directly.

- **[Data preparation](data.md)**

    ---

    Load a file, split it into the five tensors every ESN workflow
    needs, and never misalign a forecast again.

- **[Multi-readout models](multi-readout.md)**

    ---

    Several heads on one reservoir, each fitted against its own
    targets in a single pass.

- **[Coupled ensembles](ensembles.md)**

    ---

    N independently-trained ESNs sharing one aggregated feedback —
    variance reduction for chaotic forecasting.

- **[Next-Generation RC](ngrc.md)**

    ---

    Reservoir computing without recurrent weights: delay embeddings
    and polynomial features via `NGReservoir`.

- **[Save, load, checkpoint](save-load.md)**

    ---

    Persist trained models, reservoir states, and metadata — and
    restore them without surprises.

- **[PyTorch pipelines & SGD](pipelines.md)**

    ---

    ESN layers are `nn.Module`s: mix them with gradient-trained
    components in ordinary PyTorch pipelines.

- **[GPU & performance](gpu.md)**

    ---

    Move models and data to CUDA, and know which knobs actually buy
    you speed.

- **[Hyperparameter optimization](hpo.md)**

    ---

    Optuna-backed `run_hpo` with forecast-horizon losses and
    multi-worker studies.

- **[Visualizing architectures](visualization.md)**

    ---

    Render any model's DAG with `plot_model()` to see what you
    actually built.

- **[Custom components](custom-components.md)**

    ---

    Write and register your own topologies, initializers, cells, and
    transform layers.

</div>

## Related

- [Learn track](../learn/index.md) — the guided path from zero to tuned forecasts.
- [Under the hood](../under-the-hood/index.md) — the math and mechanics behind these recipes.
- [API reference](../reference/index.md) — every signature, verbatim.
