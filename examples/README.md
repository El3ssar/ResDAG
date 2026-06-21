# ResDAG Examples

Fifteen self-contained, runnable scripts covering the whole library. Every
script is deterministic (fixed seeds), prints what it is doing in numbered
sections, and finishes in well under a minute on CPU — except the GPU
benchmark, which takes a couple of minutes.

```bash
# Run any example
uv run python examples/01_quickstart.py

# Optional plots in 01, 06, and 13 (only if matplotlib is installed)
uv run python examples/01_quickstart.py --plot
```

## Map

| # | Script | What it teaches | Runtime (CPU) |
|---|--------|-----------------|---------------|
| 00 | `00_easiest_api.py` | The headline one-liner: `rd.ESN(...).fit(series).forecast(horizon=N)`; numpy round-trip; dropping down to `esn.model` | ~3 s |
| 01 | `01_quickstart.py` | Smallest end-to-end flow: data → `classic_esn` → `ESNTrainer` → `forecast` → MSE | ~5 s |
| 02 | `02_premade_models.py` | All six factories incl. `coupled_ensemble_esn`; one comparison table on a shared Lorenz task | ~5 s |
| 03 | `03_functional_api.py` | Building DAGs by hand: minimal, input-driven, ott-style augmentation, parallel reservoirs, multi-readout | ~3 s |
| 04 | `04_topologies_and_initializers.py` | Topology/initializer registries; bare-callable topologies, `(fn, params)`, `torch.nn.init.*`, `register_matrix_topology`, the `"orthogonal"` matrix topology | ~5 s |
| 05 | `05_training_paths.py` | Three training styles compared: algebraic ridge, frozen reservoir + SGD head, full BPTT | ~10 s |
| 06 | `06_forecasting.py` | Two-phase forecasting, driver time alignment (the #1 silent bug), `return_warmup`, multi-output | ~3 s |
| 07 | `07_ensembles.py` | Coupled ensembles: fit/forecast, single-vs-ensemble accuracy, aggregators, per-member spread | ~10 s |
| 08 | `08_save_load.py` | Persistence: weight roundtrips, checkpoints with states + metadata, forecast continuity, cross-device | ~5 s |
| 09 | `09_visualization.py` | `summary()` and graphviz `plot_model()` (shapes, trainable markers, layouts, file output) | ~3 s |
| 10 | `10_hpo.py` | Minimal Optuna study via `run_hpo` (needs `pip install resdag[hpo]`; skips cleanly otherwise) | ~10 s |
| 11 | `11_gpu_benchmark.py` | CPU vs GPU timing of forward/fit/forecast at three scales; skips cleanly without CUDA | ~30 s – 2 min |
| 12 | `12_feature_extractor.py` | `ReservoirFeatureExtractor` in `nn.Sequential`: frozen features → Adam regression head, a classification head, and `from_model` reuse | ~5 s |
| 13 | `13_windowed_forecast.py` | `windowed_forecast`: gap-filling reconstruction of a sparsely-observed Lorenz trajectory (alternate teacher-force re-sync + autonomous free-run); scoring the unseen gaps | ~10 s |
| 14 | `14_streaming_dataloader.py` | `TimeSeriesWindowDataset` + `make_dataloader`: windowed SGD (frozen head and full BPTT) and an algebraic `IncrementalRidgeReadout.partial_fit`/`finalize` fit, all over a standard PyTorch `DataLoader` | ~10 s |

## Suggested order

Start with **00** for the easiest possible API — the whole train-and-forecast
workflow as a single expression — then **01** to see the same flow spelled out
explicitly with the composable building blocks, and **02** to learn which
premade architecture fits your problem. When the factories
are not enough, **03** shows how to compose arbitrary DAGs and **04** how to
shape the weight matrices that go inside them. **05** and **06** are the
core skills — how models are trained and how forecasting actually works
(read 06 carefully if you use exogenous drivers), and **13** extends 06 to
gap-filling reconstruction of sparsely-observed signals. After that, the
rest are independent: **07** (ensembles) and **10** (hyperparameter search)
when you want better forecasts, **08** and **09** when you need persistence
and architecture inspection, **11** when deciding whether your workload
belongs on a GPU, and **14** when you want the plain-PyTorch `DataLoader` path
for minibatched SGD or streaming algebraic fits.

## More

- Full documentation: <https://el3ssar.github.io/ResDAG/>
- Developer reference: [`../CLAUDE.md`](../CLAUDE.md)
- Releases: <https://github.com/El3ssar/ResDAG/releases>
