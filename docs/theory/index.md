---
description: The mathematical backbone of ResDAG — every equation the code implements, with index origins, shift directions, and centering conventions pinned down.
---

<span class="nb-kicker">Theory</span>

# Theory

The course and the handbook tell you what to call. This track pins down
what the library *computes* — the exact update equations, which index is
zero, which direction the target shift points, what is and is not
centered. Every formula here is cross-checked against the line of code
that implements it, so you can validate the implementation against your
own derivations or a paper's.

Read it like a lab journal: each page is self-contained, equations carry
their code coordinates, and conventions are stated once and never assumed.

| Page | What it pins down |
| --- | --- |
| [Reservoir dynamics](dynamics.md) | The leaky-ESN update, spectral radius scaling, the echo state property, bias and symmetry, the NG-RC feature map |
| [Readout solvers](readout.md) | The ridge objective, centered vs. uncentered normal equations, conjugate gradient on the Gram system, the dtype strategy |
| [Timing conventions](timing.md) | The one-step-ahead contract, the target shift, forecast index maps, driver alignment, statefulness across calls |
| [Design of the library](design.md) | The package map, the cell/layer split, frozen weights as parameters, the hook-based trainer |

If a result in your experiment disagrees with a result on these pages,
one of us has a bug — and these pages cite their sources.

## See also

- [The mental model](../start/concepts.md) — the four ideas, prose first
- [Build](../build/index.md) — the composition handbook these equations live inside
