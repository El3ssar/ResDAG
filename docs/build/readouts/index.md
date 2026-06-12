---
description: Trainable maps from reservoir states to outputs — fitted algebraically in one pass or by gradient descent.
---

<span class="nb-kicker">Build</span>

# Readouts

A readout maps collected reservoir states to outputs. Every readout is an
ordinary linear layer underneath, so any reservoir family's states can feed
any readout, and both training paths remain available: a one-pass algebraic
solve, or gradient descent with any PyTorch optimizer.

The solver is a pluggable contract: a new readout implements
`_fit_impl(states, targets)` and inherits shape handling, validation, and
parameter copy-back from the `ReadoutLayer` base class. `CGReadoutLayer`,
the current solver, fits ridge regression by conjugate gradient.

<!-- nb-cards: build/readouts -->

## See also

- [Train](../../workflows/train.md) — both training paths in practice
- [Theory · Readout solvers](../../theory/readout.md) — the ridge problem and the conjugate-gradient solve
