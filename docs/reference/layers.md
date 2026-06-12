---
description: API reference for resdag.layers — reservoir cells and layers, readouts, and transform layers.
---

<span class="nb-kicker">Reference</span>

# Layers

Every `nn.Module` you compose models from: single-step cells, the stateful
sequence layers that wrap them, algebraically-fitted readouts, and the
transform layers in between. All are importable from `resdag.layers`
directly.

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
