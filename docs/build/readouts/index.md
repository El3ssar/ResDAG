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
`solve(states, targets)` and inherits shape handling, validation, and
parameter copy-back from the `ReadoutLayer` base class. Several solvers ship
with the library — `rd.ESN` defaults to `AutoReadoutLayer`, checking numerical
suitability while preserving the L2 objective and requested penalty. Lower-level
model factories retain `RidgeReadoutLayer`, a direct Cholesky solve;
`CGReadoutLayer` solves the ridge problem iteratively by
conjugate gradient; and `SVDReadoutLayer`, `PinvReadoutLayer`, and
`IncrementalRidgeReadoutLayer` cover rank-deficient, pseudo-inverse, and streaming
cases. [Theory · Readout solvers](../../theory/readout.md#choosing-a-solver)
tabulates which to reach for.

Start with the [practical solver guide](../../guides/readout-solvers.md) for
automatic fitting, diagnostics, explicit algorithms and MPS device policies.

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

Names are `auto`, `ridge`, `cholesky`, `cg`, `lsmr`, `svd`, `pinv`, `incremental`, `rls`, and
`fista`; a `ReadoutLayer` subclass also works. `readout_kwargs` passes additional
constructor options and can override `alpha`, `bias`, and `name`. Input/output
widths are fixed by the architecture. The pseudoinverse solver has no ridge
penalty, so `readout_alpha` does not apply to `pinv`.

With `readout="auto"`, `readout_kwargs` accepts `solver`, `solve_dtype`, `rcond`
and `cpu_fallback` as well as the common readout options. Its `solver` choices
are `"auto"`, `"primal"`, `"dual"`, `"qr"` and `"svd"`. The `use_float64` and
`gram_dtype` arguments below belong to other readout classes, not Auto.

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

For an explicit iterative fit without a feature-by-feature Gram matrix, choose
[`readout="lsmr"`](../../reference/lsmr-readout.md). It uses SciPy on CPU in
float64, preserves the ridge penalty and unpenalized intercept, and reports
per-output convergence. Accelerator inputs use a warned CPU bridge. It retains
the feature matrix, so this is distinct from streaming over a long sequence.

## Stable fits and streaming checkpoints

With an intercept, batch solvers center features and targets before forming
products. Incremental ridge merges centered chunk covariances and running means;
it avoids subtracting large raw moments. Chunking preserves the same ridge
objective, with ordinary floating-point differences.

`use_float64=True` keeps incremental accumulators and RLS inverse-correlation
state in float64 even when model parameters are converted with `.float()` or
`.to(dtype=torch.float32)`. With `use_float64=False`, working buffers follow the
parameter dtype. MPS cannot store float64 statistics: keep those readouts on CPU,
or explicitly opt out for operations supported by your installed PyTorch.
Ordinary ridge models fitted on CPU still support float32 MPS inference.

Incremental checkpoints now carry a statistics-format version. Legacy raw-moment
checkpoints load and upgrade with a warning when data was accumulated. Already
lost precision cannot be recovered: reset accumulators and refit highly offset
or ill-conditioned streams before continuing. New checkpoints retain only one
feature-by-feature Gram buffer; centered statistics do not double that storage.

Fresh RLS `partial_fit` starts from the same zero coefficient prior as
`fit_online`; `delta` controls its strength. FISTA's optional power estimate is
checked with backtracking before accepting a proximal update, including when
the initial power vector misses a nonzero eigenspace.

## See also

- [Train](../../workflows/train.md) — both training paths in practice
- [Theory · Readout solvers](../../theory/readout.md) — the ridge problem and the conjugate-gradient solve
