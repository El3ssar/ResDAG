---
description: The architecture of the library and its rationale — the cell/layer split, frozen weights as parameters, the hook-based trainer, and the callable-first init system.
---

<span class="nb-kicker">Theory · Design</span>

# Design of the library

ResDAG's architecture rests on a few decisions applied consistently
throughout the library: a two-level reservoir stack, frozen weights
stored as `Parameter`s, a trainer built on forward pre-hooks, and an
init system that accepts any callable. This page records those
decisions and the reasoning behind them.

## The package map

```text
resdag/
├── core/             ESNModel — symbolic DAG + forecast, state, persistence
├── layers/
│   ├── cells/          single-step updates, one per family: ESNCell, NGCell, …
│   ├── reservoirs/     sequence loop + state API: ESNLayer, NGReservoir, …
│   ├── readouts/       ReadoutLayer (an nn.Linear) + CGReadoutLayer, …
│   └── transforms/     Concatenate, Power, SelectiveExponentiation, …
├── init/
│   ├── topology/       registry + GraphTopology / MatrixTopology
│   ├── input_feedback/ registry + input/feedback initializers
│   ├── graphs/         17 NetworkX graph builders
│   └── matrices/       direct matrix builders (orthogonal)
├── models/           premade factories: classic_esn, ott_esn, …
├── ensemble/         coupled ensembles + aggregators
├── training/         ESNTrainer — hook-based algebraic fitting
├── hpo/              Optuna integration (optional extra)
└── utils/            data prep & I/O, esp_index
```

## The cell/layer split

Reservoirs follow PyTorch's own `LSTMCell`/`LSTM` pattern. The **cell**
— every reservoir family implements one, such as `ESNCell` — owns all
parameters and defines one timestep:
`forward(inputs, state) -> (output, new_state)`. The **layer**
(`BaseReservoirLayer` and its subclasses) owns the time loop and the
entire state-management API — lazy init, reset, get/set, detach
semantics — once, for every cell type. Adding a new reservoir means
writing a single-step update and inheriting the rest.

The split also carries a performance contract: `project_inputs` /
`step`. A leaky ESN's pre-activation separates into an input-dependent
part and a state-dependent part, and only the latter must live inside the
loop. The layer asks the cell to precompute $W_{fb}\,u_t + W_{in}\,d_t +
b$ for the *whole sequence* in one batched matmul, then iterates with
`step`, where `torch.addmm` fuses the recurrent matmul with the
precomputed slice. The loop body then issues roughly three kernel
launches per step (addmm, activation, lerp) instead of about six; at
typical reservoir sizes this determines whether GPU execution
outperforms the CPU. Cells that cannot split (NG-RC's buffer update) return `None`
from `project_inputs` and the layer falls back to per-step `forward`.

## Frozen weights are still `Parameter`s

A reservoir's weights are never trained, so why not store them as
buffers? Three reasons:

- **`state_dict` parity.** A frozen and a trainable reservoir serialize
  identically; checkpoints do not care which experiment produced them.
- **`trainable` is a single flag.** Construction is one code path;
  `trainable=False` calls `requires_grad_(False)` on every parameter,
  and unfreezing is the same call in reverse.
- **Errors surface immediately.** Running SGD against a fully frozen
  model makes `loss.backward()` raise, because no tensor in the graph
  requires grad. With buffers, the optimizer would silently do nothing.

## The trainer is a hook

`ESNTrainer.fit` does three things: reset, teacher-forced warmup, one
forward pass over the training inputs. Notably, it never computes a
topological order. Each readout gets a `forward_pre_hook` that fits it
on the exact tensor about to enter its `forward`; since the symbolic
model executes the DAG in dependency order, every readout is fitted at
the point where its input states become available, and downstream
layers consume already-fitted outputs. Multi-readout DAGs, including
readouts feeding other readouts, train correctly without explicit
graph analysis. The mechanism applies to any readout type that
implements `fit(states, targets)`.

## Registries, but callables first

Topologies and weight initializers resolve through one spec type:
a registry name (`"erdos_renyi"`), a `(name, params)` tuple, **any bare
callable** — `fn(n) -> matrix | graph`, `fn(rows, cols) -> matrix`, or an
in-place `torch.nn.init.*_` function — a `(callable, params)` tuple, or a
configured initializer object. The registry exists for discoverability
and for string-based specification in HPO search spaces; the callable
path exists because reservoir research routinely involves weight
structures that have no registry name yet. A new structure requires
only a function definition; `register_matrix_topology` /
`register_input_feedback` promote it to a named, reusable component.

!!! note "Convention"
    Resolution happens eagerly at layer construction
    (`resolve_topology` / `resolve_initializer` inside
    `ESNCell.__init__`), so an invalid spec raises at construction
    time rather than partway through an experiment.

---

## Future directions

These are under consideration, not commitments: an `integrations` namespace
for third-party couplings; per-layer seeding so a single model can pin
each reservoir's randomness independently; a public `step()` streaming
API for online/real-time inference without sequence tensors; and
additional readout solvers (direct Cholesky, randomized sketching)
behind the same `_fit_impl` contract.

## See also

- [Build](../build/index.md) — composing these components into models
- [Contributing](../project/contributing.md) — how to add a topology, cell, or solver
- [Reservoir dynamics](dynamics.md) — the equations these components implement
