---
description: Every symbol importable directly from the resdag namespace, mapped to its canonical reference page.
---

<span class="nb-kicker">Reference</span>

# Top level

The tables below cover every name in `resdag.__all__`, including its modules,
types and release string. Most are re-exports from the canonical home shown
alongside them. Optional HPO convenience attributes are listed separately.

```python
import resdag as rd

model = rd.ott_esn(reservoir_size=500, feedback_size=3, output_size=3)
trainer = rd.ESNTrainer(model)
```

## Composition

| Symbol | Canonical home |
| ------ | -------------- |
| [`ESN`][resdag.facade.ESN] | `resdag.facade` |
| [`Model`][resdag.core.Model] | `resdag.core` |
| [`Feedback`][resdag.core.Feedback] | `resdag.core` |
| [`ESNModel`][resdag.core.ESNModel] | `resdag.core` |
| [`ReservoirFeatureExtractor`][resdag.core.ReservoirFeatureExtractor] | `resdag.core` |
| [`ReservoirModelModule`][resdag.core.ReservoirModelModule] | `resdag.core` |
| [`ReservoirStates`][resdag.core.ReservoirStates] | `resdag.core` |
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
| [`ES2NCell`][resdag.layers.cells.ES2NCell] | `resdag.layers.cells` |
| [`ResESNCell`][resdag.layers.cells.ResESNCell] | `resdag.layers.cells` |
| [`GatedReservoirCell`][resdag.layers.cells.GatedReservoirCell] | `resdag.layers.cells` |
| [`ContinuousReservoirCell`][resdag.layers.cells.ContinuousReservoirCell] | `resdag.layers.cells` |
| [`EuSNCell`][resdag.layers.cells.EuSNCell] | `resdag.layers.cells` |
| [`SpikingReservoirCell`][resdag.layers.cells.SpikingReservoirCell] | `resdag.layers.cells` |
| [`LIFCell`][resdag.layers.cells.LIFCell] | `resdag.layers.cells` |
| [`NGReservoirLayer`][resdag.layers.reservoirs.NGReservoirLayer] | `resdag.layers.reservoirs` |
| [`ES2NLayer`][resdag.layers.reservoirs.ES2NLayer] | `resdag.layers.reservoirs` |
| [`ResESNLayer`][resdag.layers.reservoirs.ResESNLayer] | `resdag.layers.reservoirs` |
| [`GatedReservoirLayer`][resdag.layers.reservoirs.GatedReservoirLayer] | `resdag.layers.reservoirs` |
| [`GatedReservoir`][resdag.layers.reservoirs.GatedReservoir] | `resdag.layers.reservoirs` |
| [`EuSNLayer`][resdag.layers.reservoirs.EuSNLayer] | `resdag.layers.reservoirs` |
| [`LIFReservoirLayer`][resdag.layers.reservoirs.LIFReservoirLayer] | `resdag.layers.reservoirs` |
| [`LIFReservoir`][resdag.layers.reservoirs.LIFReservoir] | `resdag.layers.reservoirs` |
| [`CoRNNCell`][resdag.layers.cells.CoRNNCell] | `resdag.layers.cells` |
| [`CoRNNLayer`][resdag.layers.reservoirs.CoRNNLayer] | `resdag.layers.reservoirs` |
| [`LegendreDelayCell`][resdag.layers.cells.LegendreDelayCell] | `resdag.layers.cells` |
| [`LegendreDelayLayer`][resdag.layers.reservoirs.LegendreDelayLayer] | `resdag.layers.reservoirs` |
| [`TorchRecurrentCell`][resdag.layers.cells.TorchRecurrentCell] | `resdag.layers.cells` |
| [`TorchRecurrentLayer`][resdag.layers.reservoirs.TorchRecurrentLayer] | `resdag.layers.reservoirs` |

## Readouts and transforms

| Symbol | Canonical home |
| ------ | -------------- |
| [`ReadoutLayer`][resdag.layers.readouts.ReadoutLayer] | `resdag.layers.readouts` |
| [`AutoReadoutLayer`][resdag.layers.readouts.AutoReadoutLayer] | `resdag.layers.readouts` |
| [`ReadoutSolveReport`][resdag.layers.readouts.ReadoutSolveReport] | `resdag.layers.readouts` |
| [`RidgeReadoutLayer`][resdag.layers.readouts.RidgeReadoutLayer] | `resdag.layers.readouts` |
| [`CholeskyReadoutLayer`][resdag.layers.readouts.CholeskyReadoutLayer] | `resdag.layers.readouts` |
| [`CGReadoutLayer`][resdag.layers.readouts.CGReadoutLayer] | `resdag.layers.readouts` |
| [`LSMRReadoutLayer`][resdag.layers.readouts.LSMRReadoutLayer] | `resdag.layers.readouts` ([CPU fitting contract](lsmr-readout.md)) |
| [`LSMRColumnReport`][resdag.layers.readouts.LSMRColumnReport] | `resdag.layers.readouts` |
| [`LSMRSolveReport`][resdag.layers.readouts.LSMRSolveReport] | `resdag.layers.readouts` |
| [`SVDReadoutLayer`][resdag.layers.readouts.SVDReadoutLayer] | `resdag.layers.readouts` |
| [`PinvReadoutLayer`][resdag.layers.readouts.PinvReadoutLayer] | `resdag.layers.readouts` |
| [`IncrementalRidgeReadout`][resdag.layers.readouts.IncrementalRidgeReadout] | `resdag.layers.readouts` |
| [`Concatenate`][resdag.layers.transforms.Concatenate] | `resdag.layers.transforms` |
| [`FeaturePartitioner`][resdag.layers.transforms.FeaturePartitioner] | `resdag.layers.transforms` |
| [`Power`][resdag.layers.transforms.Power] | `resdag.layers.transforms` |
| [`SelectiveDropout`][resdag.layers.transforms.SelectiveDropout] | `resdag.layers.transforms` |
| [`SelectiveExponentiation`][resdag.layers.transforms.SelectiveExponentiation] | `resdag.layers.transforms` |
| [`Standardize`][resdag.layers.transforms.Standardize] | `resdag.layers.transforms` |
| [`IncrementalRidgeReadoutLayer`][resdag.layers.readouts.IncrementalRidgeReadoutLayer] | `resdag.layers.readouts` |
| [`RLSReadout`][resdag.layers.readouts.RLSReadout] | `resdag.layers.readouts` |
| [`FISTAReadoutLayer`][resdag.layers.readouts.FISTAReadoutLayer] | `resdag.layers.readouts` |
| [`FISTAReadout`][resdag.layers.readouts.FISTAReadout] | `resdag.layers.readouts` |
| [`ExtendedSquare`][resdag.layers.transforms.ExtendedSquare] | `resdag.layers.transforms` |
| [`NLAT2`][resdag.layers.transforms.NLAT2] | `resdag.layers.transforms` |
| [`NLAT3`][resdag.layers.transforms.NLAT3] | `resdag.layers.transforms` |
| [`Pad`][resdag.layers.transforms.Pad] | `resdag.layers.transforms` |
| [`PartialSquare`][resdag.layers.transforms.PartialSquare] | `resdag.layers.transforms` |

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
| [`deep_esn`][resdag.models.deep_esn.deep_esn] | `resdag.models` |
| [`es2n`][resdag.models.es2n.es2n] | `resdag.models` |
| [`eusn`][resdag.models.eusn.eusn] | `resdag.models` |
| [`gated_esn`][resdag.models.gated_esn.gated_esn] | `resdag.models` |
| [`KnowledgeModel`][resdag.models.hybrid_esn.KnowledgeModel] | `resdag.models` |
| [`hybrid_esn`][resdag.models.hybrid_esn.hybrid_esn] | `resdag.models` |
| [`lsm`][resdag.models.lsm.lsm] | `resdag.models` |
| [`ParallelESNModel`][resdag.models.parallel_esn.ParallelESNModel] | `resdag.models` |
| [`parallel_esn`][resdag.models.parallel_esn.parallel_esn] | `resdag.models` |
| [`resesn`][resdag.models.resesn.resesn] | `resdag.models` |

## Training

| Symbol | Canonical home |
| ------ | -------------- |
| [`ESNTrainer`][resdag.training.ESNTrainer] | `resdag.training` |
| [`CalibrationReport`][resdag.training.CalibrationReport] | `resdag.training` |
| [`calibrate_flow_control`][resdag.training.calibrate_flow_control] | `resdag.training` |
| [`calibrate_intrinsic_plasticity`][resdag.training.calibrate_intrinsic_plasticity] | `resdag.training` |

## Model artifacts

| Symbol | Canonical home |
| ------ | -------------- |
| [`load`][resdag.artifacts.load] | `resdag.artifacts` |
| [`register_component`][resdag.artifacts.register_component] | `resdag.artifacts` |
| [`ComponentCodec`][resdag.artifacts.ComponentCodec] | `resdag.artifacts` |
| [`ArtifactError`][resdag.artifacts.ArtifactError] | `resdag.artifacts` |

## Data, datasets, and diagnostics

| Symbol | Canonical home |
| ------ | -------------- |
| [`TimeSeriesWindowDataset`][resdag.data.TimeSeriesWindowDataset] | `resdag.data` |
| [`make_dataloader`][resdag.data.make_dataloader] | `resdag.data` |
| [`lorenz`][resdag.datasets.lorenz] | `resdag.datasets` |
| [`rossler`][resdag.datasets.rossler] | `resdag.datasets` |
| [`henon`][resdag.datasets.henon] | `resdag.datasets` |
| [`mackey_glass`][resdag.datasets.mackey_glass] | `resdag.datasets` |
| [`narma`][resdag.datasets.narma] | `resdag.datasets` |
| [`sine`][resdag.datasets.sine] | `resdag.datasets` |
| [`kuramoto_sivashinsky`][resdag.datasets.kuramoto_sivashinsky] | `resdag.datasets` |
| [`SeedStreams`][resdag.utils.SeedStreams] | `resdag.utils` |
| [`esp_index`][resdag.utils.states.esp_index.esp_index] | `resdag.utils.states` |

The generators in `resdag.datasets` each return a `(1, n_timesteps, features)`
tensor; their first positional argument is `n_timesteps`. `TimeSeriesWindowDataset`
and `make_dataloader` are the torch `DataLoader` path — see
[Work · Streaming & DataLoaders](../workflows/streaming.md).

---

## Optional HPO namespace

[`run_hpo`][resdag.hpo.run_hpo], [`LOSSES`][resdag.hpo.losses.LOSSES], and
[`get_study_summary`][resdag.hpo.utils.get_study_summary] live under `rd.hpo`.
That namespace is imported lazily. Use `model.tune(...)` for the common workflow;
advanced runners and losses keep their own documented module. `optuna` is required for running studies and
inspecting study objects; the losses remain available without it.

## Submodules

| Symbol | Canonical home or reference |
| ------ | --------------------------- |
| `core` | [Core](core.md) |
| `data` | [Data](data.md) |
| `datasets` | [Dataset generators](data.md#dataset-generators) |
| `ensemble` | [Models and ensembles](models.md) |
| `hpo` | [Hyperparameter tuning](hpo.md); imported lazily |
| `init` | [Initialization](init.md) |
| `layers` | [Layers](layers.md) |
| `metrics` | [Metrics](metrics.md) |
| `models` | [Models and ensembles](models.md) |
| `training` | [Training](training.md) |
| `utils` | [Utilities](utils.md) |
| `graphs` | `resdag.init.graphs`; [initialization](init.md) |
| `topology` | `resdag.init.topology`; [initialization](init.md) |
| `input_feedback` | `resdag.init.input_feedback`; [initialization](init.md) |
| `__version__` | Installed ResDAG release string |
