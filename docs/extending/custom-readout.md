# Add a readout

Subclass `ReadoutLayer` and implement `_fit_impl`. The base class flattens
`(batch, time, features)`, validates shapes, and writes coefficients into the
underlying linear layer.

```python
import torch
from resdag.layers.readouts.base import ReadoutLayer


class RidgeReadout(ReadoutLayer):
    def __init__(self, in_features, out_features, alpha=1e-6, **kwargs):
        super().__init__(in_features, out_features, **kwargs)
        self.alpha = alpha

    def _fit_impl(self, states, targets):
        n = states.shape[0]
        A = states.T @ states + self.alpha * torch.eye(states.shape[1], device=states.device)
        coefs = torch.linalg.solve(A, states.T @ targets)
        intercept = targets.mean(0) - states.mean(0) @ coefs
        return coefs, intercept
```

The shipped `CGReadoutLayer` is one ridge implementation; additional solvers can be
added as separate classes. Register readouts in the graph with a unique `name` for
`ESNTrainer`.

See [`ReadoutLayer`](../reference/layers/readouts.md).
