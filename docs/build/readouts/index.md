---
title: "Readouts"
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
parameter copy-back from the `ReadoutLayer` base class. Several solvers ship
with the library — the default, `RidgeReadoutLayer`, fits ridge regression with
a direct Cholesky solve; `CGReadoutLayer` solves the same problem iteratively by
conjugate gradient; and `SVDReadoutLayer`, `PinvReadoutLayer`, and
`IncrementalRidgeReadoutLayer` cover rank-deficient, pseudo-inverse, and streaming
cases. [Theory · Readout solvers](../../theory/readout.md#choosing-a-solver)
tabulates which to reach for.

<!-- nb-cards: build/readouts -->

## Choose and configure a readout

All readout-bearing factories accept `readout` and `readout_kwargs` separately
from reservoir options:

```python
import resdag as rd

model = rd.classic_esn(
    reservoir_size=300, feedback_size=3, output_size=3,
    readout="svd", readout_alpha=1e-6,
    readout_kwargs={"rcond": 1e-12, "trainable": True},
)
```

Names are `ridge`, `cholesky`, `cg`, `svd`, `pinv`, `incremental`, `rls`, and
`fista`; a `ReadoutLayer` subclass also works. `readout_kwargs` passes additional
constructor options and can override `alpha`, `bias`, and `name`. Input/output
widths are fixed by the architecture. The pseudoinverse solver has no ridge
penalty, so `readout_alpha` does not apply to `pinv`.

`trainable=True` enables gradient updates to the same readout after an algebraic
fit. Reservoir trainability is controlled independently by the factory's
`trainable` argument. For streaming fits, choose `readout="incremental"` and
call `ESNTrainer.fit_stream`.

SVD and pseudoinverse solves operate directly on the design matrix; they do not
allocate an unused feature-by-feature Gram matrix. Direct ridge and incremental
ridge form their Gram in the solve dtype (`float64` by default) on every
device. Setting `gram_dtype=torch.float32` explicitly can speed up a
well-conditioned CUDA fit, but can lose accuracy for correlated states. CG's
automatic Gram dtype remains input precision on CUDA and solve precision on CPU.

## See also

- [Train](../../workflows/train.md) — both training paths in practice
- [Theory · Readout solvers](../../theory/readout.md) — the ridge problem and the conjugate-gradient solve
