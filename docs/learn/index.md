# Learn

Theory and design rationale behind ResDAG — written like short textbook chapters.
Each page explains *why* a subsystem exists, then *what* it does, then a minimal code
skippet.

## Suggested reading order

1. [What is reservoir computing?](what-is-reservoir-computing.md) — history and core idea
2. [Mental model](../getting-started/mental-model.md) — optional 10-minute recap
3. [Reservoir layers](reservoir-layers.md) — cells vs layers
4. [Echo State Property](echo-state-property.md) — stability and spectral radius
5. [Readouts](readouts.md) — ridge regression, not SGD
6. [Two-phase training](two-phase-training.md) — warmup + fit
7. [Forecasting](forecasting.md) — autoregressive generation
8. [Graph topologies](topologies.md) — recurrent weight structure
9. [Input & feedback initializers](input-feedback-initializers.md) — how inputs enter the reservoir
10. [Next-Generation RC](ngrc.md) — delay embeddings without recurrence
11. [Coupled ensembles](ensembles.md) — diversity + shared feedback
12. [Chaos & loss functions](chaos-and-losses.md) — why MSE fails and what to optimize instead

## When to skip ahead

| Your goal | Jump to |
|-----------|---------|
| Run code now | [Get started](../getting-started/index.md) |
| API details | [Reference](../reference/index.md) |
| Copy-paste recipes | [Guides](../guides/index.md) |
| Register a custom topology | [Extend](../extending/index.md) |
