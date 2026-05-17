# Add an HPO loss

Implement a callable matching `LossProtocol`: positional `y_true`, `y_pred`, optional
keyword args, return a scalar **to minimize**.

```python
import numpy as np
from numpy.typing import NDArray


def my_horizon_loss(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    /,
    threshold: float = 0.25,
) -> float:
    err = np.linalg.norm(y_true - y_pred, axis=-1).mean(axis=0)  # (T,)
    valid = int(np.argmax(err >= threshold)) if (err >= threshold).any() else len(err)
    return -float(valid)


# Pass directly — no registry required
from resdag.hpo import run_hpo

run_hpo(..., loss=my_horizon_loss, loss_params={"threshold": 0.3})
```

Monitor without optimizing:

```python
run_hpo(..., loss="efh", monitor_losses=[my_horizon_loss])
```

To add to `LOSSES` in library code, extend `resdag.hpo.losses.LOSSES` in a fork or
wrap `get_loss`.

## See also

- [Chaos & losses](../learn/chaos-and-losses.md)
- [`LossProtocol`](../reference/hpo/losses.md)
