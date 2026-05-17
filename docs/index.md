---
title: ResDAG — PyTorch-native reservoir computing
description: >-
  ResDAG is a PyTorch-native reservoir computing library. Build Echo State
  Networks, Next-Generation Reservoir Computers, and coupled ensembles on
  GPU, with graph-driven topologies and algebraic ridge-regression training.
hide:
  - navigation
  - toc
---

<div class="resdag-hero" markdown>

# Reservoir computing, the PyTorch way.

<p class="tagline">
ResDAG turns the rich theory of Echo State Networks into a modular,
GPU-accelerated library that composes like any other <code>torch.nn.Module</code>
— and trains in a single algebraic step.
</p>

<div class="cta-row" markdown>
[:material-rocket-launch-outline: Get started](getting-started/index.md){ .md-button .md-button--primary }
[:material-book-open-page-variant: Learn the theory](learn/index.md){ .md-button }
[:material-source-branch: View on GitHub](https://github.com/El3ssar/resdag){ .md-button }
</div>

</div>

## Why ResDAG?

Reservoir computing sits in a strange spot. The theory is beautiful — a random
recurrent network, frozen forever, plus a single linear readout — but most
existing libraries either:

- treat reservoirs as *opaque black boxes* you can't compose, or
- live outside the modern deep-learning ecosystem, forcing you to choose
  between RC and the rest of your PyTorch stack.

**ResDAG fixes both.** Every component is a `torch.nn.Module`. Reservoirs run
on GPU. Models are built with a functional graph API (`pytorch_symbolic`),
so a five-layer multi-readout architecture is as easy to write as a single
ESN. Readouts are trained algebraically by Conjugate Gradient — no gradient
descent, no learning rate to tune — and the whole training loop is one
function call.

<div class="grid cards" markdown>

-   :material-flash:{ .lg .middle } **GPU-native, end to end**

    ---

    Reservoir layers, ridge-regression readouts, NG-RC feature construction
    and ensemble forecasting all run on GPU. No NumPy fallbacks. No copies.

    [:octicons-arrow-right-24: GPU & performance](guides/gpu-and-performance.md)

-   :material-graph-outline:{ .lg .middle } **17 graph topologies, one registry**

    ---

    Erdős–Rényi, Watts–Strogatz, Barabási–Albert, ring-chord, dendrocycle…
    Pick one by name, override its parameters with a tuple, or plug in your
    own graph generator with a single decorator.

    [:octicons-arrow-right-24: Topology system](learn/topologies.md)

-   :material-chart-line:{ .lg .middle } **Forecasting that just works**

    ---

    Two-phase `model.forecast(warmup, horizon=N)` handles state
    synchronization and autoregressive generation for you, with full support
    for input-driven systems and multi-output models.

    [:octicons-arrow-right-24: Forecasting](learn/forecasting.md)

-   :material-cog-transfer-outline:{ .lg .middle } **Composable like any nn.Module**

    ---

    Build a multi-input, multi-readout DAG with `pytorch_symbolic`. Wrap it
    in `ESNModel`. Train it in one call. Save, load, visualize.

    [:octicons-arrow-right-24: Model composition](learn/reservoir-layers.md)

-   :material-tune-variant:{ .lg .middle } **HPO out of the box**

    ---

    First-class Optuna integration with five loss functions designed for
    chaotic systems — including the Expected Forecast Horizon — and real
    multi-process parallelism over journal-file storage.

    [:octicons-arrow-right-24: Hyperparameter optimization](guides/hyperparameter-optimization.md)

-   :material-puzzle-outline:{ .lg .middle } **Extensible by design**

    ---

    Every system — cells, topologies, initializers, readouts, losses — has
    a registry and a base class. Adding your own is a 20-line file plus a
    decorator.

    [:octicons-arrow-right-24: Extend ResDAG](extending/index.md)

</div>

## A complete ESN in 20 lines

```python
import torch
import pytorch_symbolic as ps
from resdag import ESNModel, ESNLayer, CGReadoutLayer, ESNTrainer

# 1.  Build a graph.  Inputs are symbolic; layers are real torch.nn.Modules.
inp = ps.Input((None, 3))                                      # (T, features)
states = ESNLayer(reservoir_size=500,
                  feedback_size=3,
                  spectral_radius=0.9,
                  topology="erdos_renyi")(inp)
out = CGReadoutLayer(500, 3, alpha=1e-6, name="output")(states)

model = ESNModel(inp, out)

# 2.  Train.  No SGD — one algebraic Conjugate-Gradient ridge solve.
ESNTrainer(model).fit(
    warmup_inputs=(warmup,),       # synchronise reservoir state
    train_inputs=(train,),          # teacher-force the dynamics
    targets={"output": targets},    # one key per readout
)

# 3.  Forecast.  Two-phase: warmup + autoregressive generation.
predictions = model.forecast(forecast_warmup, horizon=1000)
```

That's the whole loop. No epochs, no schedulers, no dropout — just the
mathematics of reservoir computing in code that reads like the equations.

## Where to go next

<div class="grid cards" markdown>

-   :material-school-outline:{ .lg .middle } **New to reservoir computing?**

    ---

    Start with the [mental model](getting-started/mental-model.md) — it's
    the 10-minute version of the field. Then walk through the
    [Lorenz tutorial](getting-started/lorenz-walkthrough.md).

-   :material-tools:{ .lg .middle } **Coming from another RC library?**

    ---

    The [Reference](reference/index.md) maps every public symbol with full
    signatures and cross-links. The
    [related-work page](about/related-work.md) compares ResDAG to
    ReservoirPy, EchoTorch, and RcTorch.

-   :material-rocket-outline:{ .lg .middle } **Building something specific?**

    ---

    Browse the [Guides](guides/index.md) — task-oriented recipes for
    chaotic forecasting, input-driven systems, multi-readout architectures,
    coupled ensembles, and HPO.

-   :material-source-pull:{ .lg .middle } **Want to contribute?**

    ---

    The [Extend section](extending/index.md) walks through adding your own
    topology, initializer, cell, readout, or HPO loss — each is a single
    decorated function or class.

</div>

## The honest fine print

ResDAG is in active development (v0.4.0, alpha status). The public API
documented here is the API the maintainers commit to keeping stable —
backward-compatible shims redirect old imports (e.g. `resdag.composition →
resdag.core`) with deprecation warnings. We follow [semantic
versioning](https://semver.org/) once we reach 1.0; until then, minor
versions may rename internals but **not** the public surface listed in
[`resdag/__init__.py`](https://github.com/El3ssar/resdag/blob/main/src/resdag/__init__.py).

---

<small>
Built with :heart: in PyTorch · MIT-licensed · By [Daniel
Estevez-Moya](mailto:kemossabee@gmail.com). Documentation generated by
[MkDocs Material](https://squidfunk.github.io/mkdocs-material/) and
[mkdocstrings](https://mkdocstrings.github.io/).
</small>
