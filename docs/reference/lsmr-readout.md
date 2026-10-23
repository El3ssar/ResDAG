---
title: Low-memory LSMR readout
description: An explicit SciPy iterative ridge fit without a feature-by-feature Gram matrix.
---

# Low-memory LSMR readout

Choose `readout="lsmr"` when the readout's feature-by-feature matrix would be too
large to form. It solves the same ridge problem as the other L2 readouts, using
SciPy's established iterative solver. Automatic solver selection stays unchanged;
LSMR is an explicit choice.

```python
import torch
import resdag as rd

t = torch.arange(400, dtype=torch.float64) * 0.05
train = torch.stack((t.sin(), t.cos()), dim=-1)
model = rd.ESN(
    32, input_size=2, washout=30, seed=7, dtype=torch.float64,
    readout="lsmr", alpha=1e-5,
    readout_kwargs={
        "atol": 1e-10, "btol": 1e-10,
        "max_iter": 500, "chunk_size": 128,
    },
).fit(train)
prediction = model.forecast(horizon=40)
assert prediction.shape == (40, 2)
assert torch.isfinite(prediction).all()

head = next(layer for layer in model.graph.modules()
            if isinstance(layer, rd.LSMRReadoutLayer))
for column in head.last_solve.columns:
    print(column.output, column.stop_reason, column.iterations)
```

Custom DAGs use `rd.LSMRReadoutLayer(features, outputs, ...)` like any other
readout. Built-in vanilla and OTT models also retain this solver and its settings
when tuning rebuilds candidates. `model.save(...)` and `rd.load(...)` preserve
fitted weights, options and reservoir memory. Past solve diagnostics are runtime
metadata and start as `None` after loading.

## What memory it saves

LSMR calls a `scipy.sparse.linalg.LinearOperator` that computes products with the
feature matrix and its transpose. It never forms `X.T @ X`, an augmented identity,
or an SVD/QR decomposition. When fitting an intercept, it centers row blocks on
demand rather than retaining a second complete centered design.

The feature matrix still exists. Fitting needs a CPU float64 representation of
it, up to `min(chunk_size, samples) * features` temporary centered entries, fitted
coefficients and linear-size solver vectors. A short series may fit inside one
block. Reducing `chunk_size` lowers centering scratch storage but can increase
operator overhead. Reservoir projections and collected states are still produced
by the ordinary batch fitting path; this option does not make the whole DAG
streaming. See [streaming training](../workflows/streaming.md) when the sequence's
feature history is the limiting allocation.

## Numerical controls

The fitted objective is

$$
\min_{W,b}\;\lVert XW+b-Y\rVert_F^2+\alpha\lVert W\rVert_F^2.
$$

The data-fit term uses the **sum** of squared errors, rather than their mean. SciPy receives
`damp=sqrt(alpha)`. The intercept is unpenalized: supervised feature and target
means are removed during the solve, then restored in the fitted bias. With
`bias=False`, no centering or intercept is applied. Each target column is scaled
internally and restored afterwards so changing small target units does not
accidentally make the zero solution pass a relative stopping test.

| Option | Default | Meaning |
| --- | --- | --- |
| `alpha` | `1e-6` | Finite non-negative ridge penalty; never increased automatically |
| `atol`, `btol` | `1e-8` | Non-negative SciPy backward-error stopping tolerances |
| `conlim` | `1e8` | Estimated condition limit; zero disables this limit |
| `max_iter` | `1000` | Iteration budget per output; `None` uses SciPy's `min(samples, features)` |
| `chunk_size` | `1024` | Maximum rows centered at once |
| `cpu_fallback` | `True` | Permit the warned CPU solve for accelerator inputs |

Iteration exhaustion, a condition-limit stop, or non-finite results raise before
new coefficients are published. A failed first fit stays unfitted; a failed
refit retains its previous fitted weights. Increase the iteration budget when a
valid but difficult problem needs more iterations. For very ill-conditioned data,
compare against a [direct SVD solve](../guides/readout-solvers.md) and evaluate
held-out forecasts. Convergence is a numerical stopping result, not a guarantee
of accurate coefficients or long autonomous predictions.

`last_solve` records input placement, the CPU float64 solve, matrix dimensions
and a report per output. Per-output norms describe the normalized-target solve;
`target_scale` explains the conversion back to original units. Condition numbers
are estimates. Different solvers' conditioning estimates need not use the same
norm or matrix.

## Devices and PyTorch integration

When tuning with a memory budget, the default planner still reserves general
direct-solver workspaces. A custom `memory_estimator` can describe LSMR's
tighter requirements; see [memory and sparse models](../workflows/sparse-reservoirs.md#readout-and-sequence-memory).

Fitting runs on CPU in float64. Accelerator feature tensors are copied to CPU
with a `RuntimeWarning`; pass `cpu_fallback=False` to reject that transfer.
Fitted coefficients return to the layer's device and dtype. This explicit bridge
can cost time and host memory, so measure it for your workload.

After fitting, forward evaluation is the usual Torch linear layer. It can sit
inside a larger PyTorch model, passes gradients to earlier layers, and supports
`trainable=True` when fitted readout weights should also receive gradient updates.
The algebraic fit itself is not differentiated through SciPy.

## API

::: resdag.layers.readouts.LSMRReadoutLayer
    options:
      show_root_heading: true

::: resdag.layers.readouts.LSMRSolveReport
    options:
      show_root_heading: true

::: resdag.layers.readouts.LSMRColumnReport
    options:
      show_root_heading: true

See SciPy's [LSMR documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsmr.html)
for stopping codes and its
[LinearOperator interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html)
for the matrix products used by the solver.
