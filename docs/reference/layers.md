---
description: API reference for resdag.layers — reservoir cells and layers, readouts, and transform layers.
---

<span class="nb-kicker">Reference</span>

# Layers

The `nn.Module` components used to build models: single-step reservoir
cells, the stateful sequence layers that wrap them, readout layers, and
transform layers. All are importable directly from `resdag.layers`.

For automatic fitting, solver overrides and diagnostics, start with the
[readout solver guide](../guides/readout-solvers.md). `rd.ESN` uses
`AutoReadoutLayer` by default; lower-level factories expose it as `readout="auto"`.

::: resdag.layers.cells
    options:
      members:
        - ReservoirCell
        - ESNCell
        - NGCell
        - ES2NCell
        - ResESNCell
        - GatedReservoirCell
        - ContinuousReservoirCell
        - EuSNCell
        - SpikingReservoirCell
        - LIFCell
        - CoRNNCell
        - LegendreDelayCell
        - TorchRecurrentCell

::: resdag.layers.reservoirs
    options:
      members:
        - ESNLayer
        - BaseReservoirLayer
        - NGReservoirLayer
        - NGReservoir
        - ES2NLayer
        - ResESNLayer
        - GatedReservoirLayer
        - GatedReservoir
        - EuSNLayer
        - LIFReservoirLayer
        - LIFReservoir
        - CoRNNLayer
        - LegendreDelayLayer
        - TorchRecurrentLayer

The opt-in [LSMR readout and its solve reports](lsmr-readout.md#api) have a
dedicated API reference and memory/device guide.

::: resdag.layers.readouts
    options:
      members:
        - ReadoutLayer
        - AutoReadoutLayer
        - ReadoutSolveReport
        - CGReadoutLayer
        - RidgeReadoutLayer
        - CholeskyReadoutLayer
        - SVDReadoutLayer
        - PinvReadoutLayer
        - IncrementalRidgeReadoutLayer
        - IncrementalRidgeReadout
        - RLSReadout
        - FISTAReadoutLayer
        - FISTAReadout

::: resdag.layers.transforms
    options:
      members:
        - Concatenate
        - ExtendedSquare
        - FeaturePartitioner
        - NLAT2
        - NLAT3
        - Pad
        - PartialSquare
        - Power
        - SelectiveDropout
        - SelectiveExponentiation
        - Standardize
