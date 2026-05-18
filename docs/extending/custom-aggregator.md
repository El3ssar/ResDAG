# Add an ensemble aggregator

`CoupledEnsembleESNModel` expects `nn.Module` that maps
`(N, batch, timesteps, features) → (batch, timesteps, features)`.

```python
import torch
import torch.nn as nn


class TrimmedMean(nn.Module):
    def __init__(self, trim: int = 1):
        super().__init__()
        self.trim = trim

    def forward(self, stacked: torch.Tensor) -> torch.Tensor:
        # stacked: (N, batch, time, features)
        sorted_vals, _ = stacked.sort(dim=0)
        if 2 * self.trim >= sorted_vals.shape[0]:
            return stacked.mean(0)
        inner = sorted_vals[self.trim : -self.trim]
        return inner.mean(0)
```

```python
from resdag.models import coupled_ensemble_esn

ensemble = coupled_ensemble_esn(
    n_models=7,
    feedback_size=3,
    output_size=3,
    aggregate=TrimmedMean(trim=1),
    reservoir_size=300,
)
```

See [`OutliersFilteredMean`](https://github.com/El3ssar/resdag/blob/main/src/resdag/ensemble/aggregators/outliers_filtered_mean.py) for norm-based masking.

## See also

- [Ensemble reference](../reference/ensemble.md)
