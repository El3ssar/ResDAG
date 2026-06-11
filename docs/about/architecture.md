# Architecture & design

ResDAG is built around two promises:

1. **Keras-grade composition.** Reservoir components are plain layers wired
   into arbitrary DAGs with `pytorch_symbolic`'s functional API — the same
   call-a-layer-on-a-tensor experience as the Keras functional API, on top of
   PyTorch.
2. **PyTorch citizenship.** A built (and even trained) reservoir model is an
   ordinary `nn.Module`: it moves with `.to(device)`, serializes with
   `state_dict()`, trains with any optimizer, and embeds in a larger network
   like any other submodule.

Every structural decision below serves one of those two promises.

## Package map

```text
resdag
├── core         ESNModel (SymbolicModel + warmup/forecast/state mgmt/persistence)
├── layers
│   ├── cells       single-step updates, own the parameters  (ESNCell, NGCell)
│   ├── reservoirs  sequence loop + state management          (ESNLayer, NGReservoir)
│   ├── readouts    nn.Linear + algebraic fit()               (CGReadoutLayer)
│   └── transforms  parameterless feature ops                 (Concatenate, Power, …)
├── init         registries: graph topologies + input/feedback initializers
├── training     ESNTrainer (hook-based algebraic fitting)
├── models       premade factories (classic_esn, ott_esn, …)
├── ensemble     coupled multi-model forecasting
├── hpo          Optuna integration (optional extra)
└── utils        data prep, ESP diagnostics
```

## Cell / layer split

A *cell* computes one timestep and owns all parameters; a *layer* owns the
time loop and the state. This mirrors `nn.LSTMCell` / `nn.LSTM`, and it is
what lets ESN and NG-RC — completely different dynamics, one with a 2-D
hidden state and one with a 3-D delay buffer — share a single state
management API (`reset_state`, `get_state`, `set_state`,
`set_random_state`). New reservoir types implement the four-method
`ReservoirCell` interface and inherit the rest.

State validation is delegated *downward* (`cell.validate_state`), so each
cell owns its own shape contract instead of the layer enumerating cell types.

## Frozen weights are still `Parameter`s

Reservoir and readout weights are `nn.Parameter` with
`requires_grad_(False)` — not buffers. This is deliberate:

- `state_dict()` round-trips identically whether a model is frozen or
  trainable, so checkpoints don't depend on the training path.
- `trainable=True` is a *flag flip*, not a different class. The same
  architecture can be algebraically fitted, SGD-trained, or both in sequence
  (see [Training paths](training-paths.md)).
- Optimizers see frozen models as having zero trainable parameters, which
  fails loudly instead of silently optimizing nothing.

The complementary contract is **truncated BPTT at call boundaries**: the
stored reservoir state is detached at the end of each `forward`, so dropping
a (frozen or trainable) reservoir into a standard SGD loop just works —
gradients flow through all timesteps within a call, never across calls.

## Why the trainer uses hooks

Classical ESN training must fit each readout on the exact features it will
see at inference — *after* every upstream transform, and *after* any
upstream readout that feeds it. Instead of re-deriving the DAG's topological
order, `ESNTrainer` registers a forward pre-hook on each readout and runs
one forward pass: each hook fires precisely when its readout executes,
receives precisely the tensor the readout is about to consume, and fits
before letting execution continue. Topological order, multi-readout
correctness, and arbitrary-DAG support all fall out of the graph executor
for free. The trainer stays ~100 lines and never touches
`pytorch_symbolic` internals.

## Registries for the research surface

Topologies and input/feedback initializers are where reservoir research
happens, so both are string-keyed registries with a three-way spec
(`"name"`, `("name", {params})`, or a configured object). A new topology is
a decorated NetworkX function; a new initializer is a small subclass. Nothing
else in the library needs to know it exists.

## Assessment: what stays as is

The 0.4 → 0.5 audit deliberately did **not** restructure the package. The
current layout already separates the four axes that vary independently —
dynamics (cells), sequencing (reservoirs), training (readouts/trainer), and
wiring (core + init) — and each extension guide maps to exactly one
directory. Two legacy shims (`resdag.composition`, `resdag.layers.custom`)
remain for backward compatibility and will be removed at 1.0.

## Future directions

Planned, in rough priority order:

- **`resdag.integrations` namespace** — opt-in adapters kept out of the
  core: an sklearn-style `ESNRegressor` (fit/predict over NumPy), and a
  Lightning module wrapper for the SGD path.
- **Per-layer seeding** — today reproducibility relies on the global torch
  seed at construction time; layers should accept a `seed`/`Generator`
  argument so independent components can be seeded independently.
- **Step-level streaming API** — a public single-step `step()` on
  reservoirs for online/streaming inference without building length-1
  sequences.
- **More algebraic solvers** — direct Cholesky and SVD readouts beside CG,
  selected per problem size and conditioning.
