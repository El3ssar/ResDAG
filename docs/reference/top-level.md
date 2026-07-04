---
description: Every symbol importable directly from the resdag namespace, mapped to its canonical reference page.
---

<span class="nb-kicker">Reference</span>

# Top level

Every symbol below is importable directly from the `resdag` namespace. Each
is a re-export; the canonical page listed alongside it documents it in full.

```python
import resdag as rd

model = rd.ott_esn(reservoir_size=500, feedback_size=3, output_size=3)
trainer = rd.ESNTrainer(model)
```

## Composition

| Symbol | Canonical home |
| ------ | -------------- |
| [`ESN`][resdag.facade.ESN] | `resdag.facade` |
| [`ESNModel`][resdag.core.ESNModel] | `resdag.core` |
| [`ReservoirFeatureExtractor`][resdag.core.ReservoirFeatureExtractor] | `resdag.core` |
| [`Input`][resdag.core.Input] | `resdag.core` (re-export of `pytorch_symbolic.Input`) |
| [`reservoir_input`][resdag.core.reservoir_input] | `resdag.core` |

## Reservoirs and cells

| Symbol | Canonical home |
| ------ | -------------- |
| [`ESNLayer`][resdag.layers.reservoirs.ESNLayer] | `resdag.layers.reservoirs` |
| [`NGReservoir`][resdag.layers.reservoirs.NGReservoir] | `resdag.layers.reservoirs` |
| [`BaseReservoirLayer`][resdag.layers.reservoirs.BaseReservoirLayer] | `resdag.layers.reservoirs` |
| [`ESNCell`][resdag.layers.cells.ESNCell] | `resdag.layers.cells` |
| [`NGCell`][resdag.layers.cells.NGCell] | `resdag.layers.cells` |
| [`ReservoirCell`][resdag.layers.cells.ReservoirCell] | `resdag.layers.cells` |

## Readouts and transforms

| Symbol | Canonical home |
| ------ | -------------- |
| [`ReadoutLayer`][resdag.layers.readouts.ReadoutLayer] | `resdag.layers.readouts` |
| [`RidgeReadoutLayer`][resdag.layers.readouts.RidgeReadoutLayer] | `resdag.layers.readouts` |
| [`CholeskyReadoutLayer`][resdag.layers.readouts.CholeskyReadoutLayer] | `resdag.layers.readouts` |
| [`CGReadoutLayer`][resdag.layers.readouts.CGReadoutLayer] | `resdag.layers.readouts` |
| [`SVDReadoutLayer`][resdag.layers.readouts.SVDReadoutLayer] | `resdag.layers.readouts` |
| [`PinvReadoutLayer`][resdag.layers.readouts.PinvReadoutLayer] | `resdag.layers.readouts` |
| [`IncrementalRidgeReadout`][resdag.layers.readouts.IncrementalRidgeReadout] | `resdag.layers.readouts` |
| [`Concatenate`][resdag.layers.transforms.Concatenate] | `resdag.layers.transforms` |
| [`FeaturePartitioner`][resdag.layers.transforms.FeaturePartitioner] | `resdag.layers.transforms` |
| [`Power`][resdag.layers.transforms.Power] | `resdag.layers.transforms` |
| [`SelectiveDropout`][resdag.layers.transforms.SelectiveDropout] | `resdag.layers.transforms` |
| [`SelectiveExponentiation`][resdag.layers.transforms.SelectiveExponentiation] | `resdag.layers.transforms` |
| [`Standardize`][resdag.layers.transforms.Standardize] | `resdag.layers.transforms` |

## Models and ensembles

| Symbol | Canonical home |
| ------ | -------------- |
| [`classic_esn`][resdag.models.classic_esn.classic_esn] | `resdag.models` |
| [`ott_esn`][resdag.models.ott_esn.ott_esn] | `resdag.models` |
| [`power_augmented`][resdag.models.power_augmented.power_augmented] | `resdag.models` |
| [`linear_esn`][resdag.models.linear_esn.linear_esn] | `resdag.models` |
| [`headless_esn`][resdag.models.headless_esn.headless_esn] | `resdag.models` |
| [`coupled_ensemble_esn`][resdag.models.coupled_ensemble_esn.coupled_ensemble_esn] | `resdag.models` |
| [`CoupledEnsembleESNModel`][resdag.ensemble.CoupledEnsembleESNModel] | `resdag.ensemble` |
| [`OutliersFilteredMean`][resdag.ensemble.aggregators.OutliersFilteredMean] | `resdag.ensemble.aggregators` |

## Training

| Symbol | Canonical home |
| ------ | -------------- |
| [`ESNTrainer`][resdag.training.ESNTrainer] | `resdag.training` |

## Data, datasets, and diagnostics

| Symbol | Canonical home |
| ------ | -------------- |
| [`TimeSeriesWindowDataset`][resdag.data.TimeSeriesWindowDataset] | `resdag.data` |
| [`make_dataloader`][resdag.data.make_dataloader] | `resdag.data` |
| `lorenz`, `rossler`, `henon`, `mackey_glass`, `narma`, `sine` | `resdag.datasets` |
| [`esp_index`][resdag.utils.states.esp_index.esp_index] | `resdag.utils.states` |

The generators in `resdag.datasets` each return a `(1, n_timesteps, features)`
tensor; their first positional argument is `n_timesteps`. `TimeSeriesWindowDataset`
and `make_dataloader` are the torch `DataLoader` path — see
[Work · Streaming & DataLoaders](../workflows/streaming.md).

---

## Lazy HPO attributes

[`run_hpo`][resdag.hpo.run_hpo], [`LOSSES`][resdag.hpo.losses.LOSSES], and
[`get_study_summary`][resdag.hpo.utils.get_study_summary] are also reachable
as `rd.run_hpo`, `rd.LOSSES`, and `rd.get_study_summary`. They are resolved
lazily via module `__getattr__`, so `optuna` remains an optional dependency
until one of these attributes is first accessed.

## Submodules

`resdag.core`, `resdag.layers`, `resdag.init`, `resdag.training`,
`resdag.models`, `resdag.ensemble`, `resdag.hpo`, and `resdag.utils` are all
importable as attributes, alongside the convenience aliases `resdag.graphs`,
`resdag.topology`, and `resdag.input_feedback` (re-exports of the matching
`resdag.init` subpackages).
