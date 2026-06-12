---
description: Where ResDAG sits in the reservoir-computing software landscape.
---

<span class="nb-kicker">Project</span>

# Related work

Reservoir computing has a healthy software landscape: NumPy-based libraries
with broad method coverage and long histories, research codebases attached
to individual papers, and general sequence-modeling frameworks that include
echo-state baselines. ResDAG takes a different set of trade-offs rather
than competing feature-for-feature.

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

- [Ecosystem](ecosystem.md) — the projects ResDAG builds on and pairs with
- [Citation](citation.md) — the methods' original papers
