---
description: The daily loop — train the readout, close the forecast loop, tune the knobs that matter, and scale the winner.
---

<span class="nb-kicker">Work</span>

# The daily loop

A reservoir project settles into a rhythm quickly: fit the readout (it
takes seconds), forecast against held-out data (the only score that
matters), tune the handful of knobs that move that score, then scale the
winner — bigger reservoirs, ensembles, GPU, persistence. These four pages
are that loop, in order.

<div class="grid cards" markdown>

- **[Train](train.md)**

    ---

    Three training paths — the one-pass algebraic solve, frozen features
    with a gradient head, full BPTT — and when each earns its cost.

- **[Forecast](forecast.md)**

    ---

    Two-phase forecast anatomy, the pinned driver alignment, coupled
    ensembles, and how far chaos lets you see.

- **[Tune](tune.md)**

    ---

    The knob table, intuition for the big three, and complete
    hyperparameter studies with `run_hpo`.

- **[Scale & deploy](deploy.md)**

    ---

    When the GPU pays off, saving and loading models, and embedding
    frozen reservoirs in larger PyTorch pipelines.

</div>
