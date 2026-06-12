---
description: The layer taxonomy — stateful reservoirs split into cell and layer, readouts fitted algebraically or by gradient, and the parameterless transforms that glue a DAG together.
---

<span class="nb-kicker">Build</span>

# The parts catalog

Three kinds of part build every architecture in the library, and all of
them speak `(batch, time, features)`. **Reservoirs** hold the stateful
dynamics; every reservoir family splits the work the same way — a cell
that owns the single-step update and its parameters, and a layer that
owns the sequence loop and the full state-management API. **Readouts**
are the trainable maps from features to predictions, fitted algebraically
in one solve or by gradient descent when you choose to unfreeze them.
**Transforms** are parameterless feature operations that wire everything
else into a DAG.

The taxonomy is deliberately wider than its current residents. The
echo-state and next-generation families ship today; further families —
continuous-time reservoirs among them — drop into the same cell/layer
contract without touching anything downstream. A readout never learns
which dynamics produced the features it reads, and a transform cares only
about the feature axis, so every page below describes a slot, not a
closed list.

<!-- nb-cards: build/layers -->

## Adding a layer

A new reservoir family is one `ReservoirCell` implementation wrapped in a
`BaseReservoirLayer` subclass that inherits the sequence loop and state
API for free. The cell contract, in full:

```python
class MyCell(ReservoirCell):
    state_size: int    # shape contract for the state
    output_size: int   # trailing dim of the per-step output

    def init_state(self, batch_size, device, dtype): ...
    def forward(self, inputs, state): ...   # -> (output, new_state)
```

Its documentation is one file dropped in this folder — the page appears
in the navigation and in the cards above automatically.

## See also

- [Architectures](../architectures/index.md) — these parts assembled into DAGs
- [Initialization](../initialization/index.md) — what builds the weights inside a reservoir
- [The mental model](../../start/concepts.md) — why the parts divide this way
