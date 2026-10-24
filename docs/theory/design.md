---
description: The common model workflow, cell/layer split, parameter storage, single-pass trainer and initializer extension contracts.
---

<span class="nb-kicker">Theory · Design</span>

# Design of the library

Presets and custom DAGs share a `Model` workflow around ordinary PyTorch
modules. Reservoirs split into a single-step cell and a sequence layer;
algebraic fitting traverses shared upstream work once. Initializers and custom
components have explicit extension interfaces.

## The package map

```text
resdag/
├── facade.py         ESN — ready-made Model workflow, eager or lazy construction
├── core/             Model workflow, ESNModel graph, feedback, state and execution
├── artifacts/        portable DAG/state codecs and explicit trusted bundles
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
precomputed slice. This moves input projection out of the sequential recurrence.
Actual kernels and speed depend on device, dtype, layout and compilation; the
split alone establishes no fixed launch count or CPU/GPU crossover.
Cells that cannot split (NG-RC's buffer update) return `None`
from `project_inputs` and the layer falls back to per-step `forward`.

## Dense parameters and compressed buffers

Dense reservoir weights use frozen Parameters by default for three reasons:

- **`state_dict` parity.** A frozen and a trainable reservoir serialize
  identically; checkpoints do not care which experiment produced them.
- **`trainable` is a single flag.** Construction is one code path;
  `trainable=False` calls `requires_grad_(False)` on every parameter,
  and unfreezing is the same call in reverse.
- **PyTorch integration.** Optimizers can discover the same parameters after
  unfreezing. A frozen reservoir still propagates gradients to upstream inputs
  that require them; freezing weights does not detach the computation graph.

Compact sparse reservoirs instead keep compressed weights as buffers, without
a duplicate dense square matrix. They still pass gradients through the supported
Torch operations to upstream inputs. Inspecting a dense copy or making all
connections trainable requires an explicit representation choice; see
[sparse storage](../workflows/sparse-reservoirs.md). Parameter discovery and
checkpoint representation consequently depend on that choice.

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

## Workflow and extension boundaries

`rd.ESN` and `rd.Model` support training-only preprocessing, named signals,
explicit feedback routes/delays, tuning, and complete model artifacts. Seed
streams and custom initializer registration are public controls. Single-step
state is available through `step_stateless`; incremental readouts also support
streaming fitting.

Public extension contracts are documented in the [API reference](../reference/index.md)
and [contributing guide](../project/contributing.md). Private execution planners,
fit-preparation helpers and serialization plumbing are implementation details.
Torch's checkpoint hooks retain their required method names but are omitted
from ordinary method listings. Current compiler/export limits are stated in
[graph tooling](../project/graph-tooling.md) and the [roadmap](../project/roadmap.md).

## See also

- [Build](../build/index.md) — composing these components into models
- [Contributing](../project/contributing.md) — how to add a topology, cell, or solver
- [Reservoir dynamics](dynamics.md) — the equations these components implement
