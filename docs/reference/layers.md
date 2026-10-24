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

## Single-step cells

The public extension contract is `ReservoirCell`: implement `state_size`,
`output_size`, `init_state` and `forward(inputs, state)`. Override
`validate_state` for a different packed-state layout. Optional `project_inputs`
and `step` split a sequence projection from the recurrent update; override
both together. These are supported subclass hooks, not implementation helpers.

Continuous cells add `vector_field` and optional `output_from_state`; spiking
cells add `neuron_update`. Their documented state packing is part of each
contract. See [custom components](../advanced/custom-components.md) for a
complete cell, transform and readout example.

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

## Sequence layers

`BaseReservoirLayer` owns sequence iteration, reset and carried-state behavior.
Its `forward_stateless` and `step_stateless` return explicit recurrent state
without changing stored state. Concrete layers inherit that contract; common
inherited Torch methods are not repeated below. Use ordinary `to`, `train`,
`eval`, hooks and state dictionaries as for any `nn.Module`.

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

## Readouts

Use `fit` for algebraic training and ordinary module calls for predictions.
Custom solvers implement `ReadoutLayer.solve(states, targets)`, returning
coefficients in `(input_features, output_features)` order and an optional
intercept. The base class handles shapes, validation and parameter copy-back.
`solve` can update diagnostics or online statistics; it is a fitting hook.

Incremental ridge uses `partial_fit` then `finalize`; RLS supports `partial_fit`
and resetting `fit_online`. Both expose `n_seen` as a progress diagnostic.
Their persistent statistics keep stable checkpoint keys. See
[custom solvers](../build/readouts/cg-readout.md#custom-solvers).

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

## Transforms

Transforms are ordinary Torch modules. A custom pointwise transform can declare
`supports_framewise=True` for fast autonomous execution; temporal transforms
use the [forecast capability contract](../advanced/forecast-capabilities.md).
`Standardize.fit` and `inverse` implement a reversible fitted transform.

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
