---
description: API reference for resdag.facade — the ESN scikit-learn-style estimator that owns the whole fit → forecast workflow.
---

<span class="nb-kicker">Reference</span>

# Facade

A single object that owns the whole reservoir-computing workflow — build,
warmup, algebraic fit, autoregressive forecast — with numpy in and numpy out.
`ESN` is the estimator-shaped entry point for users who want
`fit().forecast()` without wiring the `pytorch_symbolic` graph themselves; its
`.model` attribute exposes the underlying [`ESNModel`][resdag.core.ESNModel]
the moment you outgrow it.

```python
import numpy as np
from resdag import ESN

series = np.cumsum(np.random.randn(2000, 3), axis=0)   # (time, features)
esn = ESN(reservoir_size=300, spectral_radius=0.9).fit(series)
prediction = esn.forecast(horizon=200)                 # numpy (200, 3)
full_model = esn.model                                 # the composable ESNModel
```

See [Start · The mental model](../start/concepts.md#4-forecasting-is-autoregressive-feedback)
for how `fit`/`forecast` map onto ResDAG's warmup + `ESNTrainer.fit` / rollout
phases.

::: resdag.facade
    options:
      members:
        - ESN
