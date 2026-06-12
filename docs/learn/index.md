---
description: The ResDAG course — seven chapters from "what is a reservoir" to a tuned, deployed forecaster.
---

<span class="rd-eyebrow">Learn</span>

# The course

Seven chapters, in order, each building on the last. Read it once top to
bottom and you'll know not just the API but *why* it looks the way it does.
Total time: about an hour, less if you skip the math links.

<div class="grid cards" markdown>

- **01 · [Reservoir computing, in five minutes](reservoir-computing.md)**

    ---

    Why training a random network's *output* beats training the network.
    No equations, one idea.

- **02 · [Quickstart](quickstart.md)**

    ---

    Install, build, train, forecast. Sixty seconds of code, then a tour of
    what each line did.

- **03 · [Anatomy of an ESN](anatomy.md)**

    ---

    The three moving parts — reservoir, readout, model — and the state
    that flows between them.

- **04 · [Building models](building-models.md)**

    ---

    The functional API: layers as building blocks, DAGs as models.
    Premade architectures when you don't want to wire your own.

- **05 · [Training](training.md)**

    ---

    One-pass algebraic fitting, and the two gradient paths beyond it —
    frozen-feature SGD and full BPTT.

- **06 · [Forecasting](forecasting.md)**

    ---

    Warmup, autoregression, exogenous drivers, and how far ahead you can
    actually predict.

- **07 · [Tuning](tuning.md)**

    ---

    What each hyperparameter does to the dynamics, and how to search the
    space with Optuna.

</div>

!!! tip "Already know reservoir computing?"
    Skim [02](quickstart.md) and [03](anatomy.md) for the API mapping, then
    go straight to [04 — Building models](building-models.md). That's where
    ResDAG stops looking like other RC libraries.
