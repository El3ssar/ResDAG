---
description: Extend ResDAG — custom topologies, input initializers, readouts, and reservoir cells, each in a dozen lines.
---

<span class="rd-eyebrow">Cookbook</span>

# Custom components

All four extension points, each a small function or class plus, at most,
a decorator: topology, input initializer, readout solver, reservoir cell.

## Custom topology

The direct route is a **matrix topology**: any function taking `n` and
returning an `(n, n)` tensor. Spectral-radius scaling is applied by the
layer afterwards — you only decide which connections exist.

<div class="rd-window" data-title="my_topology.py" markdown>

```python
import torch
from resdag import ESNLayer
from resdag.init.topology import register_matrix_topology

@register_matrix_topology("block_diagonal", blocks=4)
def block_diagonal(n: int, blocks: int = 4, seed=None) -> torch.Tensor:
    w = torch.zeros(n, n)
    size = n // blocks
    for b in range(blocks):
        s = b * size
        w[s : s + size, s : s + size] = torch.randn(size, size)
    return w

reservoir = ESNLayer(500, feedback_size=3, topology="block_diagonal")
```

</div>

Registration buys string access (factories, HPO search spaces, config
files); for one-off use any bare callable works with no decorator —
`topology=block_diagonal` or `topology=(block_diagonal, {"blocks": 8})`.
For graph-structured ideas, return a weighted NetworkX graph instead:

```python
import networkx as nx
from resdag.init.topology import register_graph_topology

@register_graph_topology("my_ring", weight=1.0)
def my_ring(n: int, weight: float = 1.0, seed=None) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_weighted_edges_from((i, (i + 1) % n, weight) for i in range(n))
    return g
```

## Custom input/feedback initializer

The easy path is a plain function `fn(rows, cols, **kwargs) -> matrix`:

```python
from resdag.init.input_feedback import register_input_feedback

@register_input_feedback("first_neuron", scale=1.0)
def first_neuron(rows: int, cols: int, scale: float = 1.0) -> torch.Tensor:
    w = torch.zeros(rows, cols)
    w[0, :] = scale                     # drive only the first neuron
    return w

reservoir = ESNLayer(300, feedback_size=3, feedback_initializer="first_neuron")
```

In-place mutators need no wrapping at all —
`feedback_initializer=torch.nn.init.xavier_uniform_` just works. For
initializers with real internal state, subclass `InputFeedbackInitializer`,
implement `initialize(self, weight, **kwargs)` mutating the tensor in
place, and decorate the class with the same `@register_input_feedback`.

## Custom readout

Subclass `ReadoutLayer` and override one hook: `_fit_impl`. The base
class flattens `(batch, time, features)` to `(N, features)`, validates
shapes, transposes your coefficients into `nn.Linear` layout, and flips
`is_fitted` — you only write the solve. Give it a `name` and
`ESNTrainer.fit(targets={...})` trains it exactly like a `CGReadoutLayer`.

```python
from resdag.layers.readouts import ReadoutLayer

class ExactRidgeReadout(ReadoutLayer):
    alpha = 1e-6                       # promote to an __init__ arg as needed

    def _fit_impl(self, states, targets):
        # states (N, in), targets (N, out) — already flattened by the base
        eye = torch.eye(states.shape[1], device=states.device, dtype=states.dtype)
        coefs = torch.linalg.solve(states.T @ states + self.alpha * eye,
                                   states.T @ targets)        # (in, out)
        intercept = targets.mean(0) - states.mean(0) @ coefs  # (out,)
        return coefs, intercept       # return (coefs, None) to skip the bias
```

## Custom reservoir cell

A cell defines the single-step update; `BaseReservoirLayer` supplies the
time loop and the full state-management API. Implement `state_size`,
`output_size`, `init_state`, and `forward(inputs, state) -> (output,
new_state)` — `inputs[0]` is the feedback slice, the rest are drivers,
each `(batch, features)`:

```python
import torch.nn as nn
from resdag.layers.cells import ReservoirCell
from resdag.layers.reservoirs.base_reservoir import BaseReservoirLayer

class LinearCell(ReservoirCell):
    def __init__(self, hidden: int, input_dim: int):
        super().__init__()
        self.hidden = hidden
        self.W = nn.Linear(input_dim, hidden, bias=False)

    state_size = property(lambda self: self.hidden)     # compact @property
    output_size = property(lambda self: self.hidden)

    def init_state(self, batch_size, device, dtype):
        return torch.zeros(batch_size, self.hidden, device=device, dtype=dtype)

    def forward(self, inputs, state):
        new_state = self.W(inputs[0])
        return new_state, new_state    # output and state coincide here

class LinearReservoir(BaseReservoirLayer):
    def __init__(self, hidden: int, feedback_size: int):
        super().__init__(LinearCell(hidden, feedback_size))
```

If your state isn't 2-D `(batch, state_size)`, also override the cell's
`validate_state` — that's how `NGCell` enforces its 3-D delay buffer.
Custom HPO losses are simpler still: any `(y_true, y_pred, /, **kwargs) -> float`
callable passes straight into `run_hpo(loss=...)` — see the
[loss reference](../reference/hpo/losses.md).

## Related

- [Topologies](topologies.md) — the 17 built-ins your custom graph competes with.
- [Initializers](initializers.md) — built-in input/feedback schemes.
- [NG-RC](ngrc.md) — a complete non-trivial cell built on this exact interface.
