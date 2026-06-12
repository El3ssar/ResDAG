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

::: resdag.layers.reservoirs
    options:
      members:
        - BaseReservoirLayer
        - ESNLayer
        - NGReservoir

::: resdag.layers.readouts
    options:
      members:
        - ReadoutLayer
        - CGReadoutLayer

::: resdag.layers.transforms
    options:
      members:
        - Concatenate
        - FeaturePartitioner
        - Power
        - SelectiveDropout
        - SelectiveExponentiation
