# User guide

Each page is a self-contained walkthrough — code + figures + commentary
— for a specific workflow. Read the first four if you're just learning
ResDAG; the rest are reference for specific tasks.

### Core concepts

- **[Built-in models](builtin-models.md)** — tour of the six premade
  factories (`classic_esn`, `ott_esn`, `power_augmented`, `linear_esn`,
  `headless_esn`, `coupled_ensemble_esn`) with architecture diagrams and
  recommendations.
- **[Functional API](functional-api.md)** — build your own architecture
  with `pytorch_symbolic`. Walks simple → complex through five
  canonical patterns (minimal, input-driven, multi-readout, parallel
  reservoirs, augmented).
- **[Graph topologies](topologies.md)** — the 17 registered reservoir
  topologies, what each looks like in matrix form, when to use which,
  and how to register your own.
- **[Input & feedback initializers](initializers.md)** — the 11
  registered input/feedback initializers, weight-matrix heatmaps, and
  how to plug in custom ones.

### Workflows

- **[Data preparation](data-preparation.md)** — `load_file`,
  `prepare_esn_data`, normalization, the definition of `f_warmup`.
- **[Forecasting chaotic systems](chaotic-systems.md)** — Lorenz-style
  workflow with `ott_esn` and validation on `val`.
- **[Input-driven systems](input-driven-systems.md)** — feedback plus
  exogenous drivers in the symbolic graph and in `forecast`.
- **[Multi-readout models](multi-readout.md)** — several readout heads
  in one `ESNTrainer.fit`.
- **[Coupled ensembles](coupled-ensembles.md)** — `coupled_ensemble_esn`
  with shared feedback during autoregression.
- **[Hyperparameter optimization](hyperparameter-optimization.md)** —
  `run_hpo`, loss registry, parallel studies.

### Tooling

- **[Save and load](save-and-load.md)** — weights and optional
  reservoir states.
- **[Visualizing architectures](visualizing-architectures.md)** —
  `summary()` and `plot_model()`, simple → complex gallery.
- **[GPU and performance](gpu-and-performance.md)** — devices,
  batching, readout precision, HPO workers.

When you want a single runnable script with embedded figures, jump to
the [Examples](../examples/index.md) gallery.
