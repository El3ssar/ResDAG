# Add a readout

Override `_fit_impl` on `ReadoutLayer`; the base class handles flattening and copying
weights into `nn.Linear`.

```python
import torch
from resdag.layers.readouts.base import ReadoutLayer


class RidgeTorchReadout(ReadoutLayer):
    def __init__(self, in_features, out_features, alpha=1e-6, **kwargs):
        super().__init__(in_features, out_features, **kwargs)
        self.alpha = alpha

    def _fit_impl(self, states, targets):
        # states: (N, in_features), targets: (N, out_features)
        n = states.shape[0]
        A = states.T @ states + self.alpha * torch.eye(states.shape[1], device=states.device)
        b = states.T @ targets
        coefs = torch.linalg.solve(A, b)
        intercept = targets.mean(0) - states.mean(0) @ coefs
        return coefs, intercept
```

Register the layer in your graph with a unique `name` for `ESNTrainer`.

For production use, prefer built-in [`CGReadoutLayer`](../reference/layers/readouts.md).

## See also

- [Readouts (Learn)](../learn/readouts.md)
