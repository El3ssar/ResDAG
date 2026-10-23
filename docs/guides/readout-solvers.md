---
description: Fit with automatic numerical checks, force an algorithm, inspect diagnostics and control device fallback.
---

# Choose and inspect a readout

A readout learns how to turn reservoir features into predictions. Start with
the automatic readout: `rd.ESN` uses it by default. You can choose every
algorithm and precision setting explicitly when an experiment needs them.

When a feature-by-feature matrix would be too large, the optional
[`readout="lsmr"` route](../reference/lsmr-readout.md) performs an iterative CPU
float64 fit without forming a Gram matrix. Select it explicitly; automatic
selection below remains unchanged. Its convergence controls and device bridge
are documented separately.

## Start with an automatic fit

```python
import torch
import resdag as rd

t = torch.arange(400, dtype=torch.float64) * 0.05
train = torch.stack((t.sin(), t.cos()), dim=-1)

model = rd.ESN(
    reservoir_size=32, input_size=2, washout=30,
    alpha=1e-5, seed=7, dtype=torch.float64,
)
model.fit(train)
prediction = model.forecast(horizon=40)
assert prediction.shape == (40, 2)
assert torch.isfinite(prediction).all()
```

Scaling is learned from the training data, and forecasts return in the original
units. Numerical solver selection does not tune `alpha` or other model settings.
Use held-out autonomous forecasts to choose those settings.

## What automatic selection does

All automatic paths solve the same L2 problem:

$$
\min_{W,b}\;\lVert XW+b-Y\rVert_F^2+\alpha\lVert W\rVert_F^2.
$$

With `bias=True`, features and targets are centered and an unpenalized intercept
is recovered. With `bias=False`, the solver uses the original matrices and no
intercept. `alpha` is the penalty on the **sum** of squared errors; the solver
does not divide that sum by the number of samples.

| Path | When automatic fitting tries it |
| --- | --- |
| Primal Cholesky | At least as many samples as features; solves a feature-by-feature system |
| Dual Cholesky | More features than samples and `alpha > 0`; solves a sample-by-sample system |
| Augmented QR | A tall normal-equations fit fails its numerical checks |
| Design-matrix SVD | A wide unregularized fit, or when a previous path is unsuitable |

The Cholesky checks include full inverse-norm conditioning, estimated product
roundoff and a relative optimality residual. A diagonal ratio or a small residual
alone can miss bad conditioning. QR also checks its triangular factor and the
relative scale of the data and regularization blocks. Unsafe matrix products
that can erase small but representable coefficients are rejected.

Automatic recovery keeps the requested `alpha`. It never adds rescue jitter,
never truncates positive-ridge singular values and never substitutes a sparse
or online-learning objective. These guards reduce numerical failure; they do
not guarantee the best forecast or universally accurate coefficients.

## See what actually happened

Readout layers are ordinary PyTorch modules, so inspect the fitted graph:

```python
readout = next(
    layer for layer in model.graph.modules()
    if isinstance(layer, rd.AutoReadoutLayer)
)
report = readout.last_solve
assert report is not None
print(report.solver, report.solve_device, report.solve_dtype)
print(report.alpha, report.relative_optimality_residual)
```

For a custom graph with several readouts, inspect each one separately.

| Report field | Meaning |
| --- | --- |
| `requested_solver`, `solver` | Requested policy and the algorithm that completed the solve |
| `alpha` | Requested, unchanged L2 penalty |
| `input_device`, `input_dtype` | Placement and precision of the feature tensor received by this readout |
| `solve_device`, `solve_dtype` | Placement and precision actually used for fitting |
| `n_samples`, `n_features` | Flattened training matrix dimensions |
| `condition_estimate` | Estimated conditioning of the regularized normal system; methods use different norm estimates |
| `roundoff_estimate` | Estimated normal-equations roundoff; `None` for the final QR/SVD path |
| `relative_optimality_residual` | Scale-relative error in the ridge first-order equations |
| `relative_prediction_residual` | Training prediction error divided by the centered target norm when fitting an intercept, otherwise the raw target norm |
| `fallback_reasons` | Reasons previous paths were rejected or a CPU bridge was needed |
| `precision_warning` | Explanation when the final positive-ridge decomposition remains severely ill-conditioned, otherwise `None` |

Residuals describe the **working-precision** solution before it is cast back to
the input and parameter dtype. They use the features/targets seen by the readout,
including workflow preprocessing and centering for an intercept. They are not
forecast errors measured in the original physical units. Strong regularization can legitimately produce a
large prediction residual. A tiny optimality residual does not prove that the
coefficients, or an autonomous forecast, are accurate.

`last_solve` starts as `None` and records the last successful algebraic solve.
It is immutable Python metadata and is not included in `state_dict()`. Loading
weights into a fresh layer restores its fitted flag, but does not manufacture a
past solve report. Moving a layer leaves the report describing its past solve.

## Force an algorithm

Choose `"primal"`, `"dual"`, `"qr"` or `"svd"` through `solver`:

```python
forced = rd.ESN(
    reservoir_size=32, input_size=2, washout=30,
    alpha=1e-5, seed=7, dtype=torch.float64,
    readout_kwargs={"solver": "svd", "solve_dtype": torch.float64},
)
forced.fit(train)
assert torch.isfinite(forced.forecast(horizon=10)).all()
```

A forced algorithm stays selected. Unsafe normal equations warn when they can
be computed; unrepresentable products or failed factorizations raise instead.
Forced QR raises when rank, conditioning or data scale makes that path
unreliable. Unregularized wide QR is unsupported: use SVD for that minimum-norm
problem. If a device bridge is allowed, it preserves the forced algorithm.

Keeping Auto and setting `readout_kwargs={"solver": "svd"}` retains Auto's
diagnostics and device policy. Selecting `readout="svd"` instead constructs the
standalone `SVDReadoutLayer`, with its own precision options and no Auto CPU
bridge. Fit that explicit class on a supported device, or fit on CPU and move
the fitted model for inference.

For a preset graph from a lower-level factory, request automatic selection
explicitly. Those factories retain their `"ridge"` default:

```python
graph = rd.classic_esn(
    reservoir_size=32, feedback_size=2, output_size=2,
    readout="auto", readout_alpha=1e-5,
    readout_kwargs={"solver": "svd"}, seed=7, dtype=torch.float64,
)
custom = rd.Model(graph=graph, washout=30)
custom.fit(train)
assert custom.forecast(horizon=10).shape == (10, 2)
```

In a functional DAG, attach `rd.AutoReadoutLayer(features, outputs, alpha=...)`
with the same `layer(x)` syntax as other readouts. Set `trainable=True` when you
want an optimizer to refine the fitted linear parameters. Algebraic `fit` does
not differentiate through the solve; subsequent forward calls preserve ordinary
PyTorch gradients and fitting preserves the parameter objects.

## Precision and Apple MPS

`AutoReadoutLayer` defaults to float64 fitting on CPU and CUDA. On MPS it first
tries native float32 normal equations. Cholesky substitution uses native
triangular solves rather than assuming every PyTorch Cholesky helper is
available. The fitted weights stay on their model device and keep their dtype.

If MPS needs QR/SVD, or an accelerator lacks a required operation, the default
policy explicitly fits on CPU in float64. It emits a `RuntimeWarning`, records
the reason and actual solve device/dtype, and returns coefficients to the input
device. MPS cannot store float64 tensors, so requesting float64 fitting there
also requires this bridge. Operation availability depends on the installed
PyTorch version.

Disable the bridge when an experiment must remain on its chosen device:

```python
no_bridge = rd.ESN(
    reservoir_size=32, input_size=2, washout=30,
    alpha=1e-5, seed=7, dtype=torch.float32,
    readout_kwargs={"cpu_fallback": False},
)
```

Move the whole model and training tensors to the same device as usual. Fitting
raises with an explanation if the selected device needs a disabled CPU bridge.
Set `solve_dtype=torch.float32` or `torch.float64` to choose working precision
explicitly. Precision options do not change the model's parameter dtype. Measure
runtime and held-out forecasts for your actual shapes and device; there is no
universal solver speed ranking.

## Very small penalties and SVD limits

For SVD, a positive penalty uses the filter `s / (s² + alpha)`. The implementation
evaluates it in scaled form, avoiding overflow when squaring large finite
singular values, while preserving the penalty. When `alpha` is a normal
representable working-dtype scalar, small singular values are divided directly
by `alpha` to avoid magnifying subnormal rounding through repeated division by
`sqrt(alpha)`.

That filter improvement does not remove finite-precision limits in centering,
decomposition, projection or parameter storage. With nearly dependent features
and a very small positive `alpha`, large opposing coefficients can cancel on
the training data. Both SVD and augmented least squares can then have inaccurate
coefficients despite small training and optimality residuals. Automatic fitting
warns and records `precision_warning` when its final conditioning check detects
this risk. Rescale features, explicitly choose a stronger penalty or use higher
working precision; validate autonomous forecasts independently.

For `alpha=0`, SVD returns a minimum-norm least-squares fit after dropping
singular values at or below `rcond * s_max`. Auto defaults `rcond` to working
epsilon times `max(n_samples, n_features)`; the explicit `SVDReadoutLayer`
defaults to `1e-15`. `rcond=0` retains every nonzero computed singular value,
which can amplify noise or require coefficients outside the parameter dtype.
The cutoff does not apply to positive-ridge SVD. `alpha` and an explicit `rcond`
must be finite and nonnegative; invalid live settings are rejected when solving.

## Updating solver settings from the previous preset default

`readout_kwargs` belongs to the selected readout class. Auto uses
`solve_dtype`, not the older direct solver's `use_float64`/`gram_dtype` pair.
Its solver names also differ: `"primal"` means primal **Cholesky**, while the
explicit ridge solver's `"solve"` means **LU**.

```python
# Automatic selection with an explicitly chosen working dtype.
automatic = rd.ESN(
    reservoir_size=32, input_size=2,
    readout_kwargs={"solve_dtype": torch.float32},
)

# Select the direct ridge class when you require its LU algorithm/options.
direct = rd.ESN(
    reservoir_size=32, input_size=2, readout="ridge",
    readout_kwargs={
        "solver": "solve", "use_float64": False, "gram_dtype": torch.float32,
    },
)
```

Passing `use_float64` or `gram_dtype` to Auto raises an unexpected-keyword error;
there is no silent translation or discarded setting. Explicit readout choices
such as `"svd"`, `"cg"` and `"ridge"` retain their own constructor options.

Continue with the [solver mathematics](../theory/readout.md),
[readout API reference](../reference/layers.md), or
[custom solver extension](../build/readouts/cg-readout.md#custom-solvers).
