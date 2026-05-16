# Changelog

All notable changes to `resdag` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] — Library-wide revision

This release reworks the public API around `pytorch_symbolic` composition,
trims long-standing inconsistencies, and adds first-class support for using
`ESNModel` inside any PyTorch pipeline (SGD or algebraic). **There are no
back-compat shims** — call sites that relied on the old shapes must be
updated. See `docs/training-paths.md` for an overview of the two training
paths now supported (algebraic readout vs SGD).

### Added

- **Coupled ensemble forecasting**
  - `resdag.ensemble.CoupledEnsembleESNModel` — N independently-trained ESN
    sub-models sharing an aggregated feedback signal during autoregressive
    forecasting.
  - `resdag.models.coupled_ensemble_esn(seed=...)` factory. Seeded
    construction is reproducible: two ensembles built with the same seed have
    bit-identical weights.
  - `CoupledEnsembleESNModel.fit(n_workers=N)` thread-parallel sub-model
    fitting (BLAS releases the GIL, so this is a real speed-up; no pickling).
  - `examples/11_ensemble_forecasting.py` end-to-end demo.
  - New `resdag.ensemble.aggregators` subpackage; `OutliersFilteredMean`
    lives there now (re-exported as `resdag.OutliersFilteredMean` and
    `resdag.ensemble.OutliersFilteredMean`).
- **Composition helpers**
  - `resdag.composition.reservoir_input(feature_size, dtype=None)` —
    convenience constructor for symbolic inputs; replaces the misleading
    hardcoded `ps.Input((100, F))` pattern.
- **Public API surface**
  - Top-level `resdag.__all__` now includes `BaseReservoirLayer`,
    `ReservoirCell`, `ESNCell`, `ReadoutLayer`, `Power`,
    `FeaturePartitioner`, `SelectiveDropout`, `power_augmented`,
    `reservoir_input`.
- **API ergonomics**
  - `ESNModel.warmup(reset=True)` and `ESNModel.forecast(reset=True)` reset
    reservoir states by default. Opt out with `reset=False` to continue from
    a previously saved state. Mirrored on `CoupledEnsembleESNModel`.
  - `BaseReservoirLayer.set_random_state(batch_size=..., device=..., dtype=...)`
    lazily initialises the state when called with `batch_size`. Forwarded
    through `ESNModel.set_random_reservoir_states`.
  - `ESNModel.set_reservoir_states(states, strict=True)` raises `KeyError`
    on missing or unknown reservoir keys.
- **Architectural**
  - `ReservoirCell.validate_state(state)` is the new authority on per-cell
    state-shape contracts. `BaseReservoirLayer.set_state` delegates to it.
    `NGCell` overrides for its 3-D delay-buffer layout.
  - `ReadoutLayer.fit` now owns shape normalisation, validation, and the
    parameter copy-back. Subclasses override only `_fit_impl(states, targets)
    -> (coefs, intercept|None)`. `CGReadoutLayer` migrated to the new hook.
  - `CGReadoutLayer(use_float64=True)` — default preserves the old numerical
    behaviour; set to `False` to stay in `float32` when memory is the
    bottleneck.
- **HPO**
  - `resdag.hpo.objective._validate_data_keys` now also reports unknown /
    typo-style keys and verifies each entry is a 3-D `torch.Tensor`.
    Driver keys (`warmup_<key>`, `train_<key>`, `f_warmup_<key>`,
    `forecast_<key>`) are validated when present.
- **Docs & examples**
  - `docs/training-paths.md` — when to pick algebraic vs frozen-SGD vs full
    BPTT, with code for each.
  - `examples/12_pipeline_integration.py` — `ESNModel` as `nn.Sequential`
    block, both frozen-reservoir + SGD head and fully trainable.
  - `examples/13_model_visualization.py` — renamed from
    `09_model_visualization_new.py` so every numbered example has a unique
    slot. Refreshed `examples/README.md`.
- **Tests**
  - `tests/test_ensemble/test_coupled.py` (15 tests): construction, seeded
    reproducibility, fit/forecast variants, parallel-fit determinism,
    aggregator paths (incl. the `OutliersFilteredMean` all-outliers
    regression for commit f0bd4a7), state save/load round-trips.
  - `tests/test_training/test_sgd_path.py` (5 tests, incl. GPU): asserts
    `trainable=True` models train cleanly with Adam and `trainable=False`
    models expose no autograd-tracked parameters.
  - `tests/test_input_feedback/test_registry.py`: parallels the topology
    registry tests.
  - `tests/test_training/test_trainer.py::TestESNTrainerInternals`:
    descriptor-sentinel regression test ensuring `ESNTrainer` never touches
    `_execution_order_*` / `_node_to_layer_name`; branching-DAG fit test.

### Changed (breaking)

- **`ESNModel.forecast` signature** — was
  `forecast(*warmup_inputs, horizon, forecast_drivers=None, ...)`; now
  `forecast(warmup_inputs, forecast_inputs=None, *, horizon, ...)`.
  `warmup_inputs` is a tuple/list of `(feedback, driver1, ...)` (a bare
  tensor is accepted for the common feedback-only case); `forecast_inputs`
  is the optional driver tuple (feedback is generated autoregressively).
  Same change on `CoupledEnsembleESNModel.forecast`. Call sites must be
  updated; there is no compatibility shim.
- **`ESNCell.input_size`** — `input_size=0` is now treated as
  `input_size is None`. The `(reservoir_size, 0)` weight tensor previously
  produced by every premade factory is no longer created.
- **`show_topologies()` / `show_input_initializers()`** — return the sorted
  list of registered names (in addition to printing them) when called
  without a name. They return `None` only when inspecting a specific entry.
  Restores the pre-0.3.1 contract.
- **`OutliersFilteredMean` location** — moved from `resdag.layers.custom`
  to `resdag.ensemble.aggregators`. The old import path `from
  resdag.layers import OutliersFilteredMean` no longer works; use
  `from resdag.ensemble.aggregators import OutliersFilteredMean` or the
  top-level `resdag.OutliersFilteredMean`.
- **All premade factories** (`classic_esn`, `ott_esn`, `linear_esn`,
  `headless_esn`, `power_augmented`) use `reservoir_input(feedback_size)`
  instead of the misleading hardcoded `ps.Input((100, F))`. `model.input_shape`
  is now `torch.Size([1, 1, F])`, clearly marking the time axis as a
  placeholder.

### Changed (non-breaking)

- `ESNTrainer` no longer reads `pytorch_symbolic` private attributes
  (`_execution_order_nodes`, `_execution_order_layers`,
  `_node_to_layer_name`). It walks `model.named_modules()` instead — graph
  order is recovered naturally from hook execution.
- Error messages on `BaseReservoirLayer.set_state`, `CGReadoutLayer.fit`,
  and `ESNModel.forecast` (feedback-dim mismatch) now include the
  layer/cell/readout name and a recovery hint.
- `__version__` bumped to `0.4.0`.

### Fixed

- `OutliersFilteredMean` returns the plain mean of all samples (rather than
  zero) at positions where every sample is classified as an outlier.

### Repo hygiene

- `.gitignore`: add `site/` (mkdocs build output).
- New `CHANGELOG.md`.
- New `docs/` directory (currently hosts `training-paths.md`).

## [0.3.1] — 2025

### Changed

- Improved format of `show_topologies` and `show_input_initializers` output.
- Topology and input/feedback registries refactored for clarity.

## [0.3.0]

### Added

- Next Generation Reservoir Computing (NG-RC) layer and cell (`NGCell`,
  `NGReservoir`).
- Python 3.14 compatibility.

[0.4.0]: https://github.com/El3ssar/ResDAG/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/El3ssar/ResDAG/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/El3ssar/ResDAG/releases/tag/v0.3.0
