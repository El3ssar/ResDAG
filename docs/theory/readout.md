---
description: The ridge objective with unpenalized intercept, centered vs. uncentered normal equations, conjugate gradient on the Gram system, and the mixed-precision strategy.
---

<span class="nb-kicker">Theory · Readout</span>

# Readout solvers

ResDAG fits readouts algebraically: training reduces to one linear-algebra
problem — ridge regression of collected states onto targets. Every readout
solver in the library minimizes the *same* objective; they differ only in
how they solve it. The default, `RidgeReadoutLayer`, forms the normal
equations and factorizes them directly (Cholesky); `CGReadoutLayer` solves
the identical system iteratively by conjugate gradient. This page states the
shared objective, the normal equations it forms, the precision choices that
keep the solve fast on a GPU, and — at the end — [how to choose a
solver](#choosing-a-solver) when the default doesn't fit.

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
R^{M}$, every solver forms the centered Gram and right-hand side *without
materializing a centered copy of $X$* — this reduction is shared, only the
final solve of the resulting $F\times F$ system differs by solver:

$$
\left(X^\top X - N\,\bar x\,\bar x^\top + \alpha I\right) W
= X^\top Y - N\,\bar x\,\bar y^\top
$$

In `_solve_ridge_cg` this is `XtX = X.T @ X - n * (X_mean.T @ X_mean)`
and `rhs = X.T @ y - n * (X_mean.T @ y_mean)`: the rank-one mean
corrections applied to the raw products. After the solve the intercept is
recovered exactly:

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

## Solving the system: direct vs. iterative

The normal equations are an $F \times F$ symmetric positive-definite system,
independent of $N$. There are two families of solver for it, and ResDAG
ships both.

**Direct factorization** (`RidgeReadoutLayer`, the default;
`CholeskyReadoutLayer`). At reservoir-computing widths ($F \approx
100$–$2000$) the $(F, F)$ Gram factorizes exactly and almost instantly: one
`torch.linalg.cholesky` plus a `cholesky_solve` returns the *exact* ridge
solution (up to floating point). This is both faster and more accurate than
an iterative solve at these sizes, which is why it is the default. Passing
`solver='solve'` swaps the Cholesky for a general LU (`torch.linalg.solve`),
tolerating `alpha == 0` on a full-rank Gram.

**Conjugate gradient** (`CGReadoutLayer`). The same SPD system can be solved
iteratively, which pays off when $F$ is very large or you want a matrix-free
solve. CG exploits both structural facts:

- **Memory.** Only the $F \times F$ Gram matrix is ever materialized. The
  $N \times F$ state matrix is read exactly twice (once for $X^\top X$,
  once for $X^\top Y$) and never decomposed or augmented, so the solver's
  working memory is independent of $N$. A direct factorization of the
  design matrix scales with $N$ and becomes impractical when $N$ reaches
  the hundreds of thousands.
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
  run at 1/32–1/64 throughput on consumer GPUs. This is why a naive
  float64 implementation can make ESN training slower on GPU than on
  CPU. Pass
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
target width matches `out_features`, delegates the solve to the
subclass's `_fit_impl`, then copies the solution into the standard
linear-layer parameters — `weight.copy_(coefs.T)`,
`bias.copy_(intercept)`. A fitted readout is a regular `nn.Linear`: same
forward pass, same `state_dict`.

This makes algebraic fitting and gradient training interchangeable. The
same layer can be fitted with a single algebraic solve or trained with an
optimizer such as Adam (`trainable=True` flips `requires_grad`), and a
checkpoint does not record which path produced the weights, because the
layer carries no solver-specific state.

## Choosing a solver

Six readout classes (all in `resdag.layers.readouts`, re-exported from
`resdag.layers` and the top level) solve the *same* ridge objective above;
they differ only in the numerical method. Pick by conditioning, whether you
need `alpha = 0`, problem size, and whether the data streams:

| Readout | Method | `alpha = 0`? | Rank-deficient Gram? | Cost / notes | Reach for it when |
| --- | --- | --- | --- | --- | --- |
| **`RidgeReadoutLayer`** *(default)* | Direct Cholesky (`solver='cholesky'`) or LU (`solver='solve'`) of the normal equations | `'solve'` only | No (fails / unstable) | One factorization; **exact**, fastest at $F \le 2000$ | Almost always — well-conditioned readout, `alpha > 0` |
| `CholeskyReadoutLayer` | Single-shot Cholesky ridge | No | No | Equivalent to `RidgeReadoutLayer(solver='cholesky')`; the streaming-path direct solver | You want the direct Cholesky solver by name (e.g. behind `IncrementalRidgeReadout`) |
| `CGReadoutLayer` | Iterative conjugate gradient on the Gram | Yes | Tolerates it | Matrix-free, memory-lean in $N$; **approximate** (can under-converge) | Very large $F$, or a memory-bound / matrix-free solve |
| `SVDReadoutLayer` | SVD with Tikhonov filter factors | Yes | **Yes** (robust) | An SVD of the design; most robust, most expensive | Rank-deficient Gram — high-degree NG-RC maps, more features than samples, or `alpha = 0` |
| `PinvReadoutLayer` | `torch.linalg.lstsq` / `pinv` with an `rcond` cutoff | Yes | Yes | Least squares with pinv semantics | You want pseudo-inverse / lstsq behavior directly |
| `IncrementalRidgeReadout` | Streaming ridge — `partial_fit` accumulates the Gram sufficient statistics chunk-by-chunk, `finalize` solves once | No | No | $O(F^2)$ memory, independent of total samples | The state matrix doesn't fit in memory — the DataLoader / long-sequence path ([Streaming](../workflows/streaming.md)) |

Rules of thumb:

- **Default to `RidgeReadoutLayer`.** It is exact, fastest at typical widths,
  and matches CG to `< 1e-8` on well-conditioned fits. The premade factories
  (`classic_esn`, `ott_esn`, `power_augmented`) use it via `readout="ridge"`;
  pass `readout="cg"` for the legacy iterative solver, or a `ReadoutLayer`
  subclass for any of the others.
- **`alpha = 0`?** A Cholesky needs `alpha > 0` for guaranteed positive
  definiteness. Use `RidgeReadoutLayer(solver='solve')` on a full-rank Gram,
  or `SVDReadoutLayer` / `PinvReadoutLayer` otherwise.
- **Rank-deficient Gram** (NG-RC with high `p`, or `F > N`) → `SVDReadoutLayer`
  (filter factors) or `PinvReadoutLayer` (lstsq). A Cholesky/LU solve will
  fail or be unstable here.
- **Streaming / too-big-for-memory** → `IncrementalRidgeReadout`, which keeps
  only the $F\times F$ sufficient statistics.

!!! note "Standardizing readout inputs"
    Badly-scaled features (e.g. an unnormalized driver concatenated into the
    readout) hurt conditioning. The
    [`Standardize`][resdag.layers.transforms.Standardize] transform
    (`resdag.layers.Standardize`) applies a per-feature z-score with a fixed,
    exact inverse — drop it into the graph before a readout, or normalize the
    data upstream, rather than forcing `gram_dtype=torch.float64`.

## Next

[**Timing conventions**](timing.md) — which target row each state row is
regressed onto, and where forecast index 0 lives.
