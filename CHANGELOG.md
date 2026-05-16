# Changelog

All notable changes to `resdag` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Coupled ensemble forecasting:
  - `resdag.ensemble.CoupledEnsembleESNModel` — N independently-trained ESN
    sub-models sharing an aggregated feedback signal during autoregressive
    forecasting.
  - `resdag.models.coupled_ensemble_esn` factory.
  - `examples/11_ensemble_forecasting.py` end-to-end demo.

## [0.4.0] — Phase 1 (Bug fixes, exports, hygiene)

### Added

- `resdag.composition.reservoir_input(feature_size, dtype=None)` — convenience
  constructor for a symbolic input tensor whose time dimension is an explicit
  placeholder. Re-exported as `resdag.reservoir_input`.
- Public top-level exports now include `power_augmented`, `ReadoutLayer`,
  `BaseReservoirLayer`, `ReservoirCell`, `ESNCell`, `Power`,
  `FeaturePartitioner`, `SelectiveDropout`.
- `BaseReservoirLayer` re-exported from `resdag.layers`.
- New test module `tests/test_input_feedback/` for the input/feedback
  initializer registry.
- `CHANGELOG.md`.

### Changed

- `show_topologies()` and `show_input_initializers()` now return the sorted
  list of registered names (in addition to printing them) when called without
  a name. Returns `None` only when inspecting a specific entry. This restores
  the pre-0.3.1 contract that one test still pinned.
- `ESNCell` treats `input_size == 0` the same as `input_size is None`. The
  zero-column `(reservoir_size, 0)` weight tensor previously produced by the
  premade factories is no longer created.
- All premade factories (`classic_esn`, `ott_esn`, `linear_esn`,
  `headless_esn`, `power_augmented`) pass `input_size=None` instead of `0`
  and use `reservoir_input(feedback_size)` instead of the misleading
  hardcoded `ps.Input((100, feedback_size))`. The resulting `model.input_shape`
  is now `torch.Size([1, 1, F])`, clearly marking the time axis as a placeholder.

### Fixed

- `OutliersFilteredMean` returns the plain mean of all samples (rather than
  zero) at positions where every sample is classified as an outlier.

### Repo hygiene

- `.gitignore`: add `site/` (mkdocs build output).
- `__version__` bumped to `0.4.0`.

## [0.3.1] — 2025

### Changed

- Improved format of `show_topologies` and `show_input_initializers` output.
- Topology and input/feedback registries refactored for clarity.

## [0.3.0]

### Added

- Next Generation Reservoir Computing (NG-RC) layer and cell (`NGCell`,
  `NGReservoir`).
- Python 3.14 compatibility.

[Unreleased]: https://github.com/El3ssar/ResDAG/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/El3ssar/ResDAG/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/El3ssar/ResDAG/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/El3ssar/ResDAG/releases/tag/v0.3.0
