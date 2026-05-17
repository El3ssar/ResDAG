# Multi-readout models

Fit **several readout heads** on one graph in a single `ESNTrainer.fit` call. Hooks
ensure each readout is solved before downstream layers consume its output.

## Example

```python
import pytorch_symbolic as ps
import torch
from resdag.core import ESNModel
from resdag.layers import ESNLayer
from resdag.layers.readouts import CGReadoutLayer

fb = ps.Input((80, 4))
res = ESNLayer(300, feedback_size=4)(fb)

pos = CGReadoutLayer(300, 3, name="position")(res)
vel = CGReadoutLayer(300, 3, name="velocity")(res)

model = ESNModel(fb, [pos, vel])

from resdag.training import ESNTrainer

ESNTrainer(model).fit(
    warmup_inputs=(warmup,),
    train_inputs=(train,),
    targets={
        "position": target_pos,
        "velocity": target_vel,
    },
)
```

Each key in `targets` must match a `CGReadoutLayer(..., name=...)`.

## Forecasting note

`model.forecast()` uses **only the first output** as feedback. If that head is not
4-dimensional (matching feedback), either:

- reorder outputs so the feedback-sized head is first, or
- build a dedicated low-dimensional readout for autoregression.

## See also

- [Two-phase training](../learn/two-phase-training.md)
- [Training example](https://github.com/El3ssar/resdag/blob/main/examples/09_training.py)
