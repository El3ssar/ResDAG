# Readouts & ridge regression

!!! info "Why this exists"
    Reservoir states are high-dimensional; the target map is usually **linear** in
    those states. Fitting $W_{\mathrm{out}}$ by ridge regression is fast, stable, and
    has one main hyperparameter (`alpha`) — unlike SGD on the full RNN.

## The optimization problem

Given state matrix $H \in \mathbb{R}^{N \times d}$ (all timesteps stacked) and targets
$Y \in \mathbb{R}^{N \times m}$, ridge regression solves:

$$
\min_{W, b} \| H W + b - Y \|_F^2 + \alpha \|W\|_F^2
$$

$\alpha > 0$ shrinks weights and handles collinearity when $d$ is large (typical for
reservoirs with hundreds of units).

## Why conjugate gradient?

Direct normal equations cost $O(d^3)$ — painful for $d = 5000$. `CGReadoutLayer` solves
the ridge system with **conjugate gradient** on the Gram matrix, upcasting to `float64`
by default for numerical stability, then stores coefficients in the layer.

Plain-English CG loop:

1. Start with $W = 0$.
2. Measure residual $r = X^\top y - (X^\top X + \alpha I) W$.
3. Walk search directions that are conjugate w.r.t. the ridge Hessian until residual norm
   is below `tol` or `max_iter` is hit.

Implementation: `CGReadoutLayer._solve_ridge_cg` in `src/resdag/layers/readouts/cg_readout.py`.

## Using `CGReadoutLayer`

```python
from resdag.layers.readouts import CGReadoutLayer

readout = CGReadoutLayer(
    in_features=500,
    out_features=3,
    alpha=1e-6,       # only hyperparameter that usually matters
    name="output",    # must match ESNTrainer targets key
)
```

| Parameter | Effect |
|-----------|--------|
| `alpha` | Larger → smoother, more biased fit; smaller → closer fit, risk of overfitting |
| `max_iter` / `tol` | CG stopping criteria |
| `use_float64` | Disable for huge reservoirs if memory-bound |

Readouts are fitted automatically by [`ESNTrainer`](two-phase-training.md), not by
`loss.backward()`.

## See also

- [Two-phase training](two-phase-training.md)
- [Reference: readouts](../reference/layers/readouts.md)
- [Extend: custom readout](../extending/custom-readout.md)
