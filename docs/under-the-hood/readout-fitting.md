# Readout fitting

## The objective

`CGReadoutLayer.fit(states, targets)` solves ridge regression. With state
matrix $X \in \mathbb{R}^{N \times F}$ (sequences flattened to
$N = B \cdot T$ rows) and targets $Y \in \mathbb{R}^{N \times O}$:

$$
\min_{W,\,c} \;\; \lVert X W + \mathbf{1}c^\top - Y \rVert_F^2 \;+\; \alpha \lVert W \rVert_F^2
$$

The intercept $c$ is **not penalized** — standard practice, since penalizing
it would pull predictions toward zero regardless of the target mean.

## Intercept via centering

With `bias=True` (default) the solver centers the data and recovers the
intercept afterwards:

$$
(X_c^\top X_c + \alpha I)\, W = X_c^\top Y_c,
\qquad
c = \bar{y} - \bar{x}^\top W
$$

where $X_c = X - \mathbf{1}\bar{x}^\top$, $Y_c = Y - \mathbf{1}\bar{y}^\top$.
The centered Gram matrix is formed as
$X^\top X - N\,\bar{x}\bar{x}^\top$ — one pass, no copy of $X$.

With `bias=False` the **uncentered** normal equations
$(X^\top X + \alpha I) W = X^\top Y$ are solved and no intercept exists.
Only use this when your states and targets are genuinely zero-mean (e.g.
NG-RC with `include_constant=True` already carries a constant feature that
absorbs the intercept).

!!! warning "Changed in 0.5"
    Before 0.5, `bias=False` still solved the centered problem and then threw
    the intercept away, shifting every prediction by $\bar{y} - \bar{x}^\top W$.
    If you used `bias=False` with non-centered data on ≤ 0.4, refit.

## Why conjugate gradient

The normal equations are solved per output column with the conjugate-gradient
method on the $F \times F$ system:

- **Memory** — only the Gram matrix ($F \times F$) is materialized; $X$ is
  touched once. For a 5 000-unit reservoir over 100 000 timesteps that is
  100 MB instead of a 4 GB design-matrix factorization.
- **GPU** — the inner loop is matrix–vector products, which is exactly what
  the GPU is good at. No host round-trips.
- **Precision** — the solve runs in `float64` by default
  (`use_float64=False` to opt out) and casts back to the layer dtype. The
  Gram matrix squares the condition number of $X$, so `float32` normal
  equations lose roughly half the significant digits of the direct problem.

Knobs: `alpha` (regularization), `max_iter` and `tol` (CG stopping), all set
at layer construction.

## What `fit()` does around the solve

`ReadoutLayer.fit` (the base class) owns the bookkeeping; subclasses only
implement the algebraic solve (`_fit_impl`):

1. Flatten `(B, T, F)` → `(B·T, F)` for both states and targets.
2. Validate sample counts and `out_features`.
3. Call `_fit_impl(states, targets)` → `(coefs, intercept)`.
4. Copy `coefs.T` into `self.weight`, intercept into `self.bias`
   (`nn.Linear` layout — the fitted readout *is* a regular linear layer).

Because the result lands in standard `nn.Linear` parameters, a fitted
readout behaves identically under `state_dict()`, `to(device)`, ONNX export,
or any downstream PyTorch code. Writing a custom solver means subclassing
`ReadoutLayer` and overriding `_fit_impl` — see
[Add a readout](../extending/custom-readout.md).

## Algebraic fit vs. SGD

`trainable=False` (default) freezes the readout for autograd and relies on
`fit()`; `trainable=True` leaves the same parameters open to any PyTorch
optimizer instead — see [Training paths](../about/training-paths.md). The two
are interchangeable because the readout is a plain linear map either way; you
can even `fit()` first and fine-tune with SGD afterwards.
