---
description: Family-agnostic trainable maps — the name contract, ridge regression by conjugate gradient, bias and precision semantics, and the _fit_impl hook for custom solvers.
---

<span class="nb-kicker">Build · Readouts</span>

# CGReadoutLayer

A readout is the trainable map from features to predictions, and it is
deliberately family-agnostic: it sees `(batch, time, features)` and never
asks which reservoir produced them. Any reservoir's states — or any
transform of them — feed any readout.

`ReadoutLayer` is the base class: an `nn.Linear` applied independently per
timestep (3-D input is flattened to `(batch*time, features)` and back),
carrying a `name` for multi-readout DAGs, frozen by default, and exposing
a `fit(states, targets)` interface for algebraic training.
`CGReadoutLayer`, the current concrete solver, performs ridge regression
by conjugate gradient on the normal equations rather than gradient
descent:

<div class="nb-specimen" data-label="cg_readout.py" markdown>

```python
from resdag import CGReadoutLayer

readout = CGReadoutLayer(
    in_features=500,   # feature dim of whatever feeds it
    out_features=3,
    name="output",     # the key you pass in targets={...} when fitting
    alpha=1e-6,        # ridge strength; tune on a log scale
    bias=True,         # False solves the raw, uncentered problem
    max_iter=100,      # CG iteration cap
    tol=1e-5,          # CG convergence tolerance
)
```

</div>

In practice `alpha` has the largest effect on fit quality of any training
parameter; sweep it logarithmically (1e-8 to 1e-2) before tuning anything
else. `ESNTrainer.fit(targets={"output": y})` matches targets to readouts
by `name`: the key in the targets dict must equal the layer's `name`.

## Bias semantics

With `bias=True` the solver centers states and targets and recovers an
unpenalized intercept afterwards — the standard ridge-with-intercept
formulation. With `bias=False` it solves the raw, uncentered normal
equations, because centering without an intercept at predict time would
shift every prediction.

## Precision

Two parameters control numerical precision. `use_float64=True` (the
default) runs the CG iterations in float64. `gram_dtype` controls the
Gram-matrix products, which dominate the cost: it defaults to float64 on
CPU and to the input dtype on CUDA, because consumer GPUs execute float64
at a fraction of float32 throughput — forcing full precision there can
make ESN training slower on GPU than on CPU. Pass
`gram_dtype=torch.float64` to force float64 everywhere; the numerics are
covered in [Readout solvers](../../theory/readout.md).

---

## Custom solvers

`fit()` owns the bookkeeping — flattening, sample-count and shape
validation, copy-back into the `nn.Linear` parameters — and delegates the
algebra to `_fit_impl`. A new solver overrides one method:

```python
import torch
from resdag import ReadoutLayer

class LstsqReadout(ReadoutLayer):
    def _fit_impl(self, states, targets):
        # already flattened: states (N, in_features), targets (N, out_features)
        coefs = torch.linalg.lstsq(states, targets).solution
        return coefs, None    # (in, out) matrix; None leaves the bias untouched
```

Return a coefficient matrix of shape `(in_features, out_features)` and an
optional intercept; the base class transposes into the `(out, in)` layout
`nn.Linear` expects and sets `is_fitted`. Because every readout is an
ordinary `nn.Linear` underneath, gradient training remains available:
pass `trainable=True` and fit it with any optimizer instead.

## See also

- [Readout solvers](../../theory/readout.md) — the ridge problem and its numerics
- [Architectures](../architectures/index.md) — multi-readout DAGs keyed by name
- [Layers reference](../../reference/layers.md) — full signatures
