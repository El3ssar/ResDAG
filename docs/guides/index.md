# Guides

Task-oriented pages: data loading and splits, forecasting, ensembles,
hyperparameter search, persistence, and performance. Each guide assumes
`prepare_esn_data` unless noted otherwise.

**[Data preparation](data-preparation.md)** — `load_file`, `prepare_esn_data`,
normalization, and the definition of `f_warmup`.

**[Forecasting chaotic systems](chaotic-systems.md)** — Lorenz-style workflow with
`ott_esn` and validation on `val`.

**[Input-driven systems](input-driven-systems.md)** — feedback plus exogenous drivers
in the symbolic graph and in `forecast`.

**[Multi-readout models](multi-readout.md)** — several readout heads in one
`ESNTrainer.fit`.

**[Coupled ensemble forecasting](coupled-ensembles.md)** — `coupled_ensemble_esn`
with shared feedback during autoregression.

**[Hyperparameter optimization](hyperparameter-optimization.md)** — `run_hpo`,
loss registry, parallel studies.

**[Save and load](save-and-load.md)** — weights and optional reservoir states.

**[Visualizing architectures](visualizing-architectures.md)** — `summary()` and
`plot_model()`.

**[GPU and performance](gpu-and-performance.md)** — devices, batching, readout
precision, HPO workers.
