---
description: The projects ResDAG builds on and pairs with — pytorch_symbolic and TSDynamics.
---

<span class="nb-kicker">Project</span>

# Ecosystem

## pytorch_symbolic

ResDAG's composition layer is built on
[pytorch_symbolic](https://github.com/gahaalt/pytorch-symbolic) by Szymon
Mikler. It provides the symbolic tracing that turns layer calls on
placeholder tensors into executable model graphs — the mechanism behind
`Input`, the functional API, and the model class. Without it, the central
premise of this library — reservoir models composed as arbitrary DAGs —
would not exist in its current form. Credit where it is due.

## TSDynamics

[TSDynamics](https://github.com/El3ssar/TSDynamics) is a companion library
of dynamical systems by the same author: a catalog of chaotic ODEs, maps,
and delay systems with a uniform integration interface, plus Lyapunov
analysis utilities.

The two projects are designed to pair: TSDynamics generates the systems,
ResDAG forecasts them. The benchmark systems in this documentation — the
landing-page animation, the forecast figures — are TSDynamics systems, and
the integration will deepen as both libraries grow (shared benchmark
suites, data loaders for its systems, and reservoirs for continuous-time
dynamics).

```python
# the intended pairing
# pip install tsdynamics resdag
data = ...        # trajectory from a TSDynamics system, (batch, time, dim)
model = ...       # a ResDAG model
```

## See also

- [Citation](citation.md) — how to cite ResDAG and the methods it implements
- [Contributing](contributing.md) — development setup
