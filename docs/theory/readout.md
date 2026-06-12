---
description: The ridge objective with unpenalized intercept, centered vs. uncentered normal equations, conjugate gradient on the Gram system, and the mixed-precision strategy.
---

<span class="nb-kicker">Theory · Readout</span>

# Readout solvers

Training a readout is one linear-algebra problem: ridge regression of
collected states onto targets. This page states the objective
`CGReadoutLayer` actually solves, the normal equations it forms, and the
precision choices that make the solve fast on a GPU without going wrong.

## The objective

Let $X \in \mathbb R^{N \times F}$ be the flattened states ($N = B\,T$
samples, $F$ = `in_features`) and $Y \in \mathbb R^{N \times M}$ the
targets. With `bias=True` (the default) the layer solves ridge regression
with an **unpenalized intercept**:

$$
\min_{W,\;c}\; \lVert X W + \mathbf 1 c^\top - Y \rVert_F^2
\;+\; \alpha\, \lVert W \rVert_F^2
$$

Only the slope $W \in \mathbb R^{F \times M}$ is shrunk; the intercept
$c \in \mathbb R^{M}$ absorbs the means at no cost. Penalizing $c$ would
make the fit depend on where your data sits relative to the origin —
ridge should be translation-invariant, and this formulation is.

## Centered normal equations

Eliminating $c$ analytically reduces the problem to ridge on centered
data. With sample means $\bar x \in \mathbb R^{F}$, $\bar y \in \mathbb
R^{M}$, the code forms the centered Gram and right-hand side *without
materializing a centered copy of $X$*:

$$
\left(X^\top X - N\,\bar x\,\bar x^\top + \alpha I\right) W
= X^\top Y - N\,\bar x\,\bar y^\top
$$

— in `_solve_ridge_cg`: `XtX = X.T @ X - n * (X_mean.T @ X_mean)` and
`rhs = X.T @ y - n * (X_mean.T @ y_mean)`, the rank-one mean corrections
applied to the raw products. After the solve the intercept is recovered
exactly:

$$
c = \bar y - W^\top \bar x
$$

(`intercept = y_mean - X_mean @ coefs`). With `bias=False` the layer
instead solves the **raw, uncentered** normal equations
$(X^\top X + \alpha I)\,W = X^\top Y$ — no centering, no intercept.

!!! warning "Changed in 0.5"
    Pre-0.5 the solver centered the data even when `bias=False`, then had
    no intercept to apply at predict time — every prediction was
    systematically shifted by $\bar y - W^\top \bar x$. Since 0.5,
    `bias=False` means uncentered ridge: centering is only valid when the
    intercept that undoes it exists.

---

## Why conjugate gradient on the Gram system

The normal equations are an $F \times F$ symmetric positive-definite
system, independent of $N$. CG exploits both facts:

- **Memory.** Only the $F \times F$ Gram matrix is ever materialized. The
  $N \times F$ state matrix is read exactly twice (once for $X^\top X$,
  once for $X^\top Y$) and never decomposed or augmented — for $N$ in
  the hundreds of thousands, a direct solver on the design matrix would
  not fit where this does.
- **GPU-friendly iterations.** Each CG step is one matvec
  `XtX @ w + alpha * w` plus a handful of vector ops — exactly what a GPU
  pipelines well. All $M$ output columns are solved simultaneously
  (block CG with per-column step sizes), with per-column convergence
  tracked as $\lVert r_j \rVert^2 < \texttt{tol}^2$.

The convergence test is checked **every 10 iterations**, not every
iteration: reading a scalar predicate off the GPU forces a device-to-host
sync that serializes the pipeline, and the price of overshooting is at
most nine extra cheap $F \times F$ matvecs. Iteration count and tolerance
are the layer's `max_iter=100` and `tol=1e-5`.

## The dtype strategy

Two precision decisions, made independently because their costs differ by
orders of magnitude:

- **Gram formation** — the heavy $(N, F)$ matmuls — runs in `gram_dtype`.
  Default `None` resolves automatically: `float64` on CPU, where it is
  nearly free, and the **input dtype on CUDA**, because float64 matmuls
  run at 1/32–1/64 throughput on consumer GPUs — the classic reason ESN
  training used to measure *slower* on GPU than CPU. Pass
  `gram_dtype=torch.float64` to force full precision everywhere (only
  worth it for badly scaled states; prefer normalizing the data).
- **CG iterations** — on the small $(F, F)$ system — run in `float64`
  whenever `use_float64=True` (default). This is cheap on every device
  and stabilizes the part of the computation where rounding actually
  accumulates. The result is cast back to the input dtype at the end.

## What `fit()` does around the solve

The base `ReadoutLayer.fit` (a subclass of `torch.nn.Linear`) owns
everything except the algebra: it flattens `(B, T, F)` inputs to
`(B·T, F)`, validates that states and targets agree on $N$ and that the
target width matches `out_features`, delegates to `_fit_impl`, then copies
the solution into the standard linear-layer parameters — `weight.copy_(coefs.T)`,
`bias.copy_(intercept)`. A fitted readout **is** a regular
`nn.Linear`: same forward, same `state_dict`, nothing bespoke left behind.

That is what makes the algebraic and gradient worlds interchangeable. The
same layer can be fitted in one CG solve today and trained with Adam
tomorrow (`trainable=True` flips `requires_grad`); a checkpoint does not
record which path produced the weights, because there is nothing to
record.

## Next

[**Timing conventions**](timing.md) — which target row each state row is
regressed onto, and where forecast index 0 lives.
