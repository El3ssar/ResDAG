---
description: The equations ResDAG implements, with index origins, shift directions, and centering conventions stated precisely.
---

<span class="nb-kicker">Theory</span>

# Theory

The Start and Build sections describe the API. This section specifies
what the library *computes*: the exact update equations, which index is
zero, which direction the target shift points, and what is and is not
centered. Each formula is cross-referenced with the code that implements
it, so the implementation can be checked against your own derivations or
a published result.

Each page is self-contained, equations cite their implementing code, and
conventions are stated explicitly rather than assumed.

| Page | What it covers |
| --- | --- |
| [Reservoir dynamics](dynamics.md) | The leaky-ESN update, spectral radius scaling, the echo state property, bias and symmetry, the NG-RC feature map |
| [Readout solvers](readout.md) | The ridge objective, centered vs. uncentered normal equations, conjugate gradient on the Gram system, the dtype strategy |
| [Timing conventions](timing.md) | The one-step-ahead contract, the target shift, forecast index maps, driver alignment, statefulness across calls |
| [Design of the library](design.md) | The package map, the cell/layer split, frozen weights as parameters, the hook-based trainer |

## See also

- [The mental model](../start/concepts.md) — a prose introduction to the concepts these equations formalize
- [Build](../build/index.md) — composing models from the layers these equations describe
