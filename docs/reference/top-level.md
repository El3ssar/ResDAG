# Top-level API

Symbols re-exported from `import resdag` (see `resdag.__all__`).

::: resdag
    options:
      members:
        - ESNModel
        - reservoir_input
        - BaseReservoirLayer
        - ESNCell
        - ESNLayer
        - NGCell
        - NGReservoir
        - ReservoirCell
        - CGReadoutLayer
        - ReadoutLayer
        - Concatenate
        - FeaturePartitioner
        - OutliersFilteredMean
        - Power
        - SelectiveDropout
        - SelectiveExponentiation
        - CoupledEnsembleESNModel
        - ESNTrainer
        - classic_esn
        - coupled_ensemble_esn
        - ott_esn
        - headless_esn
        - linear_esn
        - power_augmented
      show_submodules: false
      heading_level: 2

Optional HPO symbols (`run_hpo`, `LOSSES`, `get_study_summary`) are available
when Optuna is installed — see [HPO — run](hpo/run.md).
