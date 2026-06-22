---
description: Where ResDAG sits in the reservoir-computing software landscape.
---

<span class="nb-kicker">Project</span>

# Related work

Reservoir computing has a healthy software landscape: NumPy-based libraries
with broad method coverage and long histories, PyTorch research codebases
attached to individual papers, and a maintained Julia stack. The closest
neighbours are [ReservoirPy](https://github.com/reservoirpy/reservoirpy)
(NumPy/SciPy), [EchoTorch](https://github.com/nschaetti/EchoTorch) and
[RcTorch](https://github.com/blindedjoy/RcTorch) (PyTorch), and
[ReservoirComputing.jl](https://github.com/SciML/ReservoirComputing.jl)
(Julia). ResDAG takes a different set of trade-offs rather than competing
feature-for-feature.

## Feature comparison

How ResDAG's surface area lines up against the four libraries above. A `✓`
marks a documented, first-class capability; `~` a partial or undocumented
one; `—` its absence. The point is positioning, not a scorecard — each
library leads on the axes it was designed for.

| Capability | ResDAG | ReservoirPy | EchoTorch | RcTorch | ReservoirComputing.jl |
|---|:--:|:--:|:--:|:--:|:--:|
| PyTorch-native (`nn.Module`) | ✓ | — | ✓ | ✓ | — |
| First-class GPU path | ✓ | ~ | ~ | ~ | ~ |
| NG-RC features | ✓ | ✓ | — | — | ✓ |
| Algebraic (ridge) readout | ✓ | ✓ | ✓ | ✓ | ✓ |
| Arbitrary DAG composition | ✓ | ✓ | — | — | — |
| Built-in ensembles | ✓ | — | — | — | — |
| Built-in HPO | ✓ | ✓ | — | ✓ | — |

The algebraic ridge readout is universal — it is the defining training step
of reservoir computing — so the table separates on the rest: how the
reservoir is composed, where it runs, and what is bundled around it.
ReservoirPy is the broadest non-PyTorch neighbour (a `Node`/`Model` graph
API, NG-RC, and `hyperopt` integration); ReservoirComputing.jl pairs NG-RC
with the SciML ecosystem; the PyTorch codebases (EchoTorch, RcTorch) expose
a single configurable reservoir rather than a composition surface, and both
are no longer actively maintained. ResDAG is the one that treats reservoirs,
readouts, and transforms as composable PyTorch parts wired into arbitrary
DAGs.

For wall-clock numbers on identical architectures against ReservoirPy and
ReservoirComputing.jl — training throughput, autoregressive generation, and
forecast skill — see the [benchmarks](../workflows/benchmarks.md).

## What ResDAG optimizes for

**Composition over configuration.** Most reservoir software exposes a
configurable pipeline: one reservoir, one readout, options on each. ResDAG
exposes parts — reservoirs, readouts, transforms as PyTorch modules — and a
functional API to wire them into arbitrary DAGs. Architectures from the
literature (state augmentation, parallel reservoirs, multiple heads) are a
few lines of composition instead of framework features.

**PyTorch citizenship.** Models are ordinary `nn.Module`s: they move with
`.to(device)`, serialize with `state_dict()`, embed in larger networks, and
train with any optimizer. The algebraic one-pass fit and gradient descent
are interchangeable paths over the same parameters, not separate systems.

**GPU throughput.** The sequence loop, the solver's precision strategy,
and the training pass are engineered for CUDA; at research scale the GPU
is an order of magnitude faster than the CPU path, and the regime is
[measured and documented](../workflows/deploy.md), not assumed.

**Documented mathematics.** Every update equation, solver decision, and
timing convention is [written down](../theory/index.md) against the code
that implements it.

## See also

- [Benchmarks](../workflows/benchmarks.md) — head-to-head timings and forecast skill vs ReservoirPy and ReservoirComputing.jl
- [Ecosystem](ecosystem.md) — the projects ResDAG builds on and pairs with
- [Citation](citation.md) — the methods' original papers
