---
title: "Layers"
description: Stateful reservoirs split into a cell and a layer, readouts trained algebraically or by gradient, and parameterless transforms that connect them into a DAG.
---

<span class="nb-kicker">Build</span>

# Layers

Every architecture in the library is built from three kinds of component,
all of which operate on `(batch, time, features)` tensors. **Reservoirs**
hold the stateful dynamics; every reservoir family splits the work the
same way: a cell that owns the single-step update and its parameters, and
a layer that owns the sequence loop and the full state-management API.
**[Readouts](../readouts/index.md)** are the trainable maps from features
to predictions, fitted algebraically in one solve or by gradient descent
when unfrozen. **Transforms** are parameterless feature operations that
connect everything else into a DAG.

Choose among leaky ESNs, gated reservoirs, continuous and spiking dynamics,
oscillators, Legendre memory, native PyTorch GRU/LSTM adapters, and
next-generation delay features. All use the same cell/layer contract.
New families fit that contract without changes downstream. A readout is independent of the
dynamics that produced the features it reads, and a transform operates
only on the feature axis, so new implementations in any category compose
with the existing ones.

<!-- nb-cards: build/layers -->

## Changing a built layer

Live settings are forwarded to the cell: changing `layer.leak_rate` or
`layer.noise` changes the next update. EuSN exposes its integration interval
through both `layer.epsilon` and `layer.dt`. For LIF, changing `layer.dt`,
`layer.tau_mem` or `layer.tau_syn` recomputes both exponential decays; changing
`layer.v_threshold` changes the firing threshold. Time intervals/constants must
be positive and finite. A failed assignment leaves the old settings intact.
An explicit LIF `alpha`/`beta` override remains possible; another time-setting
assignment recomputes them from the time constants.

Sizes, spectral radius and discretized Legendre settings are fixed when the
weights are built. Rebuild the layer to change those settings. A rejected
assignment raises instead of silently creating an ineffective attribute.

To edit frozen recurrent weights, use ordinary PyTorch updates such as
`with torch.no_grad(): cell.weight_hh.copy_(new_weights)`. The sparse CSR cache
refreshes before the next recurrent operation, including parameter replacement.
Weights that require gradients use the dense path. Avoid `.data` edits: they
bypass PyTorch's mutation tracking. CSR is a derived operator; the dense weights
remain the canonical serialized representation.

## Adding a layer

A new reservoir family is one `ReservoirCell` implementation wrapped in a
`BaseReservoirLayer` subclass, which inherits the sequence loop and state
API. The cell contract, in full:

```python
class MyCell(ReservoirCell):
    state_size: int    # shape contract for the state
    output_size: int   # trailing dim of the per-step output

    def init_state(self, batch_size, device, dtype): ...
    def forward(self, inputs, state): ...   # -> (output, new_state)
```

## See also

- [Readouts](../readouts/index.md) — the trainable maps from features to predictions
- [Architectures](../architectures/index.md) — these components assembled into DAGs
- [Initialization](../initialization/index.md) — what builds the weights inside a reservoir
- [The mental model](../../start/concepts.md) — why the components divide this way
