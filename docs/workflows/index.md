---
description: Train the readout, run autoregressive forecasts, tune hyperparameters, and scale to larger models and the GPU.
---

<span class="nb-kicker">Work</span>

# Workflows

A reservoir computing project iterates through the same cycle: fit the
readout (an algebraic solve that takes seconds), forecast against
held-out validation data, adjust the hyperparameters that affect forecast
quality, then scale the result with larger reservoirs, ensembles, GPU
execution, and model persistence. These four pages cover that cycle in
order.

<div class="grid cards" markdown>

- **[Train](train.md)**

    ---

    Three training approaches — the one-pass algebraic solve, frozen
    features with a gradient head, and full BPTT — and when each is
    appropriate.

- **[Forecast](forecast.md)**

    ---

    The two-phase forecast procedure, driver alignment during
    autoregression, coupled ensembles, and Lyapunov-time limits on
    forecast horizons.

- **[Tune](tune.md)**

    ---

    The eight hyperparameters that drive forecast quality, intuition for
    spectral radius, leak rate, and ridge regularization, and complete
    hyperparameter studies with `run_hpo`.

- **[Scale & deploy](deploy.md)**

    ---

    When to move to the GPU, saving and loading models, and embedding
    frozen reservoirs in larger PyTorch pipelines.

</div>
