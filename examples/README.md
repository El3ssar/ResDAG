# resdag Examples

This directory contains runnable scripts demonstrating every major feature of
the library. They are numbered roughly by progression: registry primitives →
single models → training → forecasting → HPO → ensembles → pipeline
integration.

All examples target the **pytorch_symbolic-based 0.4.0 API**. The legacy
`ModelBuilder` / `model_scope` APIs from pre-0.3 are gone.

## Quick reference

| #  | File                                    | What it teaches                                                                       |
| -- | --------------------------------------- | ------------------------------------------------------------------------------------- |
| 00 | `00_registry_system.py`                 | The topology + input/feedback registries; `show_topologies()`, `get_topology()`        |
| 01 | `01_reservoir_with_topology.py`         | Plugging graph topologies into a single `ESNLayer`                                     |
| 02 | `02_input_feedback_initializers.py`     | Input / feedback weight initialisers                                                  |
| 03 | `03_model_composition.py`               | Building `ESNModel`s by hand with `pytorch_symbolic.Input`                            |
| 04 | `04_model_visualization.py`             | `model.summary()` and graphviz-backed `model.plot_model()`                            |
| 05 | `05_functional_api.py`                  | Functional building idioms with `pytorch_symbolic`                                    |
| 06 | `06_premade_models.py`                  | The premade factories: `classic_esn`, `ott_esn`, `headless_esn`, `linear_esn`        |
| 07 | `07_save_load_models.py`                | `ESNModel.save / load`, checkpoints, cross-device loading                              |
| 08 | `08_forecasting.py`                     | Algebraic training + `forecast()` with the unified driver passing                      |
| 09 | `09_training.py`                        | `ESNTrainer` workflows: single / multi-readout, driving inputs                         |
| 10 | `10_hpo.py`                             | Optuna-based hyperparameter optimisation via `run_hpo`                                |
| 11 | `11_ensemble_forecasting.py`            | `CoupledEnsembleESNModel`: mean / median / outlier-filtered aggregation                |
| 12 | `12_pipeline_integration.py`            | ESN as a torch sub-module: frozen-feature + SGD head, fully trainable, `nn.Sequential` |
| 13 | `13_model_visualization.py`             | Alternative visualisation pipeline (was `09_model_visualization_new.py`)              |

## Running examples

```bash
# A single example
uv run --extra dev python examples/06_premade_models.py

# All of them in order
for f in examples/[0-9][0-9]_*.py; do
    echo "=== $f ==="
    uv run --extra dev python "$f"
done
```

## Companion docs

- `../docs/training-paths.md` — when to pick algebraic vs mixed vs full-BPTT.
- `../CHANGELOG.md` — recent API changes, especially the 0.4.0 entries.
- `../CLAUDE.md` — full developer reference.

## Need help?

- Top-level README: `../README.md`
- API reference (in source): `src/resdag/`
