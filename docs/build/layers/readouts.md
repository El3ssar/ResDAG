---
description: Family-agnostic trainable maps — the name contract, ridge regression by conjugate gradient, bias and precision semantics, and the _fit_impl hook for custom solvers.
---

<span class="nb-kicker">Build · Layers</span>

# Readouts

A readout is the trainable map from features to predictions, and it is
deliberately family-agnostic: it sees `(batch, time, features)` and never
asks which reservoir produced them. Any reservoir's states — or any
transform of them — feed any readout.

`ReadoutLayer` is the base: an `nn.Linear` applied independently per
timestep (3-D input is flattened to `(batch*time, features)` and back),
carrying a `name` for multi-readout DAGs, frozen by default, and exposing
a `fit(states, targets)` interface for algebraic training. The shipped
solver is `CGReadoutLayer` — ridge regression by conjugate gradient on
the normal equations, not gradient descent:

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

`alpha` is the single most consequential training knob in the library —
sweep it logarithmically (1e-8 to 1e-2) before touching anything else.
The `name` is a contract: `ESNTrainer.fit(targets={"output": y})` finds
this layer by that string, nothing more.

## Bias semantics

With `bias=True` the solver centers states and targets and recovers an
unpenalized intercept afterwards — textbook ridge-with-intercept. With
`bias=False` it solves the raw, uncentered normal equations, because
centering without an intercept at predict time would shift every
prediction.

## Precision

Two knobs, briefly: `use_float64=True` (default) runs the small CG
iterations in float64, while `gram_dtype` controls the heavy Gram-matrix
matmuls — automatically float64 on CPU and the input dtype on CUDA, where
float64 throughput on consumer GPUs is the classic reason ESN training
measures slower on GPU than CPU. Pass `gram_dtype=torch.float64` to force
full precision everywhere; the numerics are dissected in
[Readout solvers](../../theory/readout.md).

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
`nn.Linear` expects and flips `is_fitted`. Because every readout is an
ordinary `nn.Linear` underneath, the gradient world stays open — pass
`trainable=True` and train it with any optimizer instead.

## See also

- [Readout solvers](../../theory/readout.md) — the ridge problem and its numerics
- [Architectures](../architectures/index.md) — multi-readout DAGs keyed by name
- [Layers reference](../../reference/layers.md) — full signatures
