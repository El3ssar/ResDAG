---
description: Stable centered ridge regression, explicit solver choices, working precision and streaming statistics.
---

<span class="nb-kicker">Theory · Readout</span>

# Readout solvers

A readout learns a linear map from reservoir features to targets. The friendly
`rd.ESN` preset uses `AutoReadoutLayer`: it checks numerical suitability and
chooses among primal/dual Cholesky, augmented QR and direct SVD without increasing
the requested L2 penalty. Lower-level model factories retain a direct ridge
default; pass `readout="auto"` to select the same policy explicitly. See the
[practical solver guide](../guides/readout-solvers.md) for examples, diagnostics,
device fallback and forcing an algorithm.

## The ridge objective

Let $X\in\mathbb R^{N\times F}$ contain $N=B\,T$ feature rows and
$Y\in\mathbb R^{N\times M}$ contain targets. Batch and incremental ridge with
`bias=True` minimize

$$
\lVert XW+\mathbf1c^\top-Y\rVert_F^2+\alpha\lVert W\rVert_F^2.
$$

The intercept $c$ is unpenalized. Center both matrices first:

$$
X_c=X-\mathbf1\bar x^\top,\qquad Y_c=Y-\mathbf1\bar y^\top.
$$

Then solve

$$
(X_c^\top X_c+\alpha I)W=X_c^\top Y_c,
\qquad c=\bar y-W^\top\bar x.
$$

Centering **before** multiplying matters. The algebraically equivalent
$X^\top X-N\bar x\bar x^\top$ subtracts two potentially enormous nearly equal
numbers. It can lose the meaningful variation even in float64. ResDAG forms
products from centered matrices, including for explicit-noise ridge fits and CG.
With `bias=False`, it uses raw products and no intercept: centering would change
that different objective.

SVD and pseudoinverse solvers work directly with the centered design matrix.
They avoid squaring its condition number through normal equations and do not
allocate an unused feature-by-feature Gram matrix.

## Choosing a solver

| Readout / factory key | Method | Useful when |
| --- | --- | --- |
| `AutoReadoutLayer` / `auto` | Checked primal/dual Cholesky, augmented QR or design-matrix SVD | A common L2 objective with automatic numerical checks and explicit overrides |
| `RidgeReadoutLayer` / `ridge` | Cholesky, or explicit `solver="solve"` for LU | Regularized, well-conditioned batch fitting |
| `CholeskyReadoutLayer` / `cholesky` | Direct Cholesky ridge | You want that method by name |
| `CGReadoutLayer` / `cg` | Iterative conjugate gradient on a full Gram matrix | Iterative fitting is preferable for your feature width and tolerance |
| `LSMRReadoutLayer` / `lsmr` | SciPy iterative least squares through matrix/vector products | An explicit CPU float64 fit without a feature-by-feature matrix |
| `SVDReadoutLayer` / `svd` | Design-matrix SVD with ridge filter factors | Correlated or rank-deficient features; more features than samples |
| `PinvReadoutLayer` / `pinv` | Least squares / pseudoinverse with an `rcond` cutoff | Unregularized least squares with explicit rank tolerance |
| `IncrementalRidgeReadoutLayer` / `incremental` | Merge chunk statistics, then solve once | The complete feature sequence does not fit in memory |
| `RLSReadout` / `rls` | Sequential inverse-correlation updates | Online adaptation, optionally with forgetting |
| `FISTAReadoutLayer` / `fista` | Accelerated proximal iterations | Sparse coefficients through L1 / elastic-net regularization |

These algorithms do not all minimize the same objective. Pseudoinverse fitting
has no ridge penalty. RLS applies its initial `delta` prior to an augmented bias
feature too, and forgetting changes sample weights. FISTA adds an L1 penalty;
`l1_ratio` controls the balance with its quadratic penalty. Check each class's
reference when comparing coefficients.

Auto always keeps the L2 objective and requested `alpha`: FISTA, RLS and
pseudoinverse are never automatic substitutes for a positive-ridge fit. It tries
the smaller normal system, checks full inverse-norm conditioning and estimated
Gram roundoff, and checks a normalized optimality residual. QR/SVD recovery does
not increase the penalty. A forced Auto algorithm never switches to another
algorithm; unsuitable forced paths warn or raise. The immutable `last_solve`
report identifies the actual method, placement, precision, residuals and reasons
for recovery.

For `alpha=0`, a Cholesky or LU solve needs adequate rank; use SVD or pseudoinverse
for rank-deficient designs. Cholesky's recovery path can add rescue jitter or
project a numerically indefinite Gram, emitting a warning. This is a numerical
fallback, not evidence that the original requested problem was well-conditioned.
This rescue behavior belongs to the explicit ridge classes, not Auto's
unchanged-penalty recovery policy.

SVD computes $X_c=U\,\mathrm{diag}(s)V^\top$ and, for positive ridge,

$$
W=V\,\mathrm{diag}\!\left(\frac{s}{s^2+\alpha}\right)U^\top Y_c.
$$

The filter is evaluated in scaled form to avoid overflowing $s^2$ for large
finite singular values. It preserves `alpha`; this numerical improvement does
not guarantee accurate decomposition or coefficients for arbitrary data scales.
For zero penalty, `rcond` discards small singular values and defines the numerical
minimum-norm solution. Auto uses a dtype/dimension-aware default cutoff; the
explicit SVD class defaults to `1e-15`. No cutoff truncates positive-ridge SVD.
Both SVD and Auto reject nonfinite or negative penalties/cutoffs.

Very weak regularization on dependent features can produce large cancelling
coefficients in finite precision, even with small training and optimality
residuals. Auto warns and records the final severe-conditioning risk when
detected. Rescaling, an explicitly stronger penalty or higher precision can help;
held-out autonomous forecast accuracy remains a separate validation target.

CG still stores an $F\times F$ Gram matrix: it is **not matrix-free**. It solves
output columns together with separate step sizes, warns on under-convergence,
and checks its absolute residual tolerance every ten iterations. Increasing
`max_iter` or changing `tol` trades work against the quality of that approximate
solve. Compare predictions as well as residuals before choosing it for speed.

LSMR instead applies the design and its transpose through a linear operator,
centering bounded row blocks when fitting an intercept. Its damping is
`sqrt(alpha)`, preserving the same ridge objective. It uses relative stopping
tests, per-output target scaling, and fails explicitly on iteration or condition
limits. The feature matrix remains resident and accelerator fitting requires a
warned CPU bridge; see the [LSMR guide](../reference/lsmr-readout.md).

FISTA uses an exact Gram eigenvalue by default. Optional `power_iters>0` estimates
curvature more cheaply, but an estimate plus a fixed margin is not a guaranteed
upper bound. Quadratic-majorizer backtracking verifies each proximal update. A
power vector that misses a nonzero eigenspace therefore cannot turn a real fitting
problem into a false converged zero solution.

## Working precision and device placement

Auto uses `solve_dtype`: float64 on CPU/CUDA by default, and an initial native
float32 attempt on MPS. Unsafe or unavailable MPS decompositions use an explicit
CPU float64 bridge, with a warning and the actual device recorded in the report.
`cpu_fallback=False` raises instead of bridging. The fitted parameters keep their
ordinary model dtype/device, and a forced algorithm remains forced across a
bridge. Unlike the explicit solvers below, Auto does not accept `use_float64`
or `gram_dtype`; choose the options of the selected readout class.

Direct and incremental ridge form their Gram in the solve dtype by default:
float64 with `use_float64=True`. CG retains a different historical default:
solve precision on CPU, input precision on CUDA. Explicit `gram_dtype` overrides
these choices. Lower-precision products can be faster on some accelerators, but a
later float64 solve cannot recover information already lost during multiplication.

Incremental accumulation and RLS recursion retain their configured working
precision across parameter conversions. For example, `.float()` makes their
weights float32 while retaining float64 statistics when `use_float64=True`.
With `use_float64=False`, statistics follow the parameter dtype. Values are moved
directly into the destination working dtype, without an intermediate narrowing
cast that would destroy accumulated precision.

MPS cannot store float64 statistics. Moving such an incremental or RLS readout to
MPS raises before changing its parameters. Keep it on CPU for float64 training,
or explicitly use `use_float64=False` for operations supported by your installed
PyTorch. This does not guarantee every MPS solver operation is available. Ordinary
ridge readouts fitted on CPU can still be moved to float32 MPS for inference.

Choose precision using held-out forecast quality and measured runtime. Feature
scaling fitted on training data helps conditioning; it does not replace a stable
solver or justify discarding precision automatically.

## Stable streaming and checkpoints

Incremental ridge merges each chunk's centered cross-products and means with the
Chan covariance update. For chunks with counts $n_a,n_b$ and mean differences
$\Delta_x,\Delta_y$, the cross-product correction is

$$
C_{xy}=C_{xy,a}+C_{xy,b}
+\frac{n_an_b}{n_a+n_b}\Delta_x\Delta_y^\top.
$$

The feature covariance uses the same formula with $\Delta_y=\Delta_x$. This keeps
$O(F^2+FM)$ statistics independently of total sequence length, avoids subtracting
large raw moments at finalization, and preserves the batch ridge objective up to
floating-point differences between chunk orders.

`XtX` and `Xty` now contain **centered** cross-products when `bias=True`, and raw
products when `bias=False`. Checkpoints include running means and a statistics
format version. Legacy raw-moment checkpoints upgrade automatically, with a
warning when samples were accumulated. Conversion cannot reconstruct precision
already lost in an old checkpoint: reset and refit highly offset or ill-conditioned
streams before continuing. No second feature-by-feature covariance is retained.

Fresh RLS `partial_fit` starts from zero weights, matching `fit_online` and
`reset_state`. An unobserved feature therefore does not retain a random coefficient
prior. Existing saved RLS weights and inverse-correlation state remain loadable.

## Fitting, gradients and customization

`ReadoutLayer.fit` flattens sequence inputs, validates shapes and finite values,
asks `solve` for coefficients and intercept, then copies them into standard
`nn.Linear` parameters. The fitted layer remains an ordinary PyTorch module.
Set `trainable=True` to refine its parameters with a gradient optimizer.

Custom solvers return coefficients shaped `(in_features, out_features)` and a
vector intercept when `bias=True`; return `None` for an absent bias. See
[custom solvers](../build/readouts/cg-readout.md#custom-solvers) for a complete
example that handles both cases.

Continue with [streaming](../workflows/streaming.md),
[training](../workflows/train.md), or [forecast timing](timing.md).
