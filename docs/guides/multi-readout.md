# Multi-readout models

Several readout layers can share one reservoir graph. `ESNTrainer` fits each readout
in forward order via hooks; `targets` keys must match readout `name` values.

```python
import pytorch_symbolic as ps
from resdag.core import ESNModel
from resdag.layers import ESNLayer
from resdag.layers.readouts import CGReadoutLayer
from resdag.training import ESNTrainer

fb = ps.Input((1, 4))
res = ESNLayer(400, feedback_size=4)(fb)

pos = CGReadoutLayer(400, 3, name="position")(res)
vel = CGReadoutLayer(400, 3, name="velocity")(res)

model = ESNModel(fb, [pos, vel])

ESNTrainer(model).fit(
    warmup_inputs=(warmup,),
    train_inputs=(train,),
    targets={"position": target_pos, "velocity": target_vel},
)
```

For `forecast`, the first output tensor is used as feedback; its feature dimension
must match `feedback_size`.

See [`ReadoutLayer`](../reference/layers/readouts.md).
