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
│   ├── graphs/         NetworkX graph builders
│   └── matrices/       direct matrix builders (orthogonal, antisymmetric, …)
├── models/           premade factories: classic_esn, ott_esn, …
├── ensemble/         coupled ensembles + aggregators
├── training/         ESNTrainer — hook-based algebraic fitting
├── hpo/              Optuna integration (optional extra)
├── data/             data prep, I/O, window datasets, DataLoader factory
├── datasets/         public time-series generators
├── metrics/          forecast, dynamical, and attractor diagnostics
└── utils/            RNG helpers, esp_index, compatibility data imports
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

Reservoir weights are frozen by default, so why not store them as
buffers? Three reasons:

- **`state_dict` parity.** A frozen and a trainable reservoir serialize
  identically; checkpoints do not care which experiment produced them.
- **`trainable` is a single flag.** Construction is one code path;
  `trainable=False` calls `requires_grad_(False)` on every parameter,
  and unfreezing is the same call in reverse.
- **PyTorch integration.** Optimizers can discover the same parameters after
  unfreezing. A frozen reservoir still propagates gradients to upstream inputs
  that require them; freezing weights does not detach the computation graph.

## The trainer is a hook

`ESNTrainer.fit` resets recurrence once and evaluates a continuous
warmup-plus-training prefix in topological order. Each readout's pre-hook fits
only the training suffix, then its normal forward computes fitted outputs for
the complete prefix. A downstream reservoir consequently warms up with the
fitted upstream mapping and continues through training without replaying any
upstream nodes. Each reservoir consumes its samples once; shared branches are
cached until their last consumer.

Normal PyTorch calls preserve module hooks. Declare dependencies as separate
functional graph nodes or use standard `torch.nn.Sequential` containers.
Opaque custom containers mixing readouts and reservoirs are rejected because
their internal ordering cannot be inferred safely. The common `Model.fit`
also supports internal targets without exposing additional outputs and stages
the fit before committing. See [training](../workflows/train.md).
`fit_stream` has its own independent incremental-readout contract; it cannot
retroactively refit upstream heads after consuming their downstream inputs.

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

These are under consideration, not commitments: a consistent high-level
workflow for custom graphs and presets; explicit feedback routing and delays;
per-layer seed streams; and a public streaming API without sequence tensors.
Direct Cholesky, SVD, conjugate-gradient, incremental, online and sparse
readouts are already available through the same fitting contract.

## See also

- [Build](../build/index.md) — composing these components into models
- [Contributing](../project/contributing.md) — how to add a topology, cell, or solver
- [Reservoir dynamics](dynamics.md) — the equations these components implement
