---
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

None of these categories is closed. The echo-state and next-generation
families are the current reservoir implementations; further families,
continuous-time reservoirs among them, fit the same cell/layer contract
without changes anywhere downstream. A readout is independent of the
dynamics that produced the features it reads, and a transform operates
only on the feature axis, so new implementations in any category compose
with the existing ones.

<!-- nb-cards: build/layers -->

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
