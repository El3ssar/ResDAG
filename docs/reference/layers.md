---
description: API reference for resdag.layers — reservoir cells and layers, readouts, and transform layers.
---

<span class="nb-kicker">Reference</span>

# Layers

The `nn.Module` components used to build models: single-step reservoir
cells, the stateful sequence layers that wrap them, readout layers, and
transform layers. All are importable directly from `resdag.layers`.

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

::: resdag.layers.readouts
    options:
      members:
        - ReadoutLayer
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
