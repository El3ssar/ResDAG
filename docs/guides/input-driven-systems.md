# Input-driven systems

Use when an **exogenous signal** (control input, covariate, forcing) drives the reservoir
alongside autoregressive feedback.

## Build the graph

```python
import pytorch_symbolic as ps
import torch
from resdag.core import ESNModel
from resdag.layers import ESNLayer
from resdag.layers.readouts import CGReadoutLayer

feedback = ps.Input((100, 1))
driver = ps.Input((100, 5))

reservoir = ESNLayer(
    reservoir_size=200,
    feedback_size=1,
    input_size=5,
    spectral_radius=0.9,
)(feedback, driver)

readout = CGReadoutLayer(200, 1, name="output")(reservoir)
model = ESNModel([feedback, driver], readout)
```

Convention: **first** symbolic input is always feedback; additional inputs are drivers.

## Train

```python
from resdag.training import ESNTrainer

ESNTrainer(model).fit(
    warmup_inputs=(warmup_fb, warmup_drv),
    train_inputs=(train_fb, train_drv),
    targets={"output": target},
)
```

## Forecast with known future drivers

During autoregression, feedback comes from the model; drivers must be supplied for
each forecast step:

```python
pred = model.forecast(
    (warmup_fb, warmup_drv),
    forecast_inputs=(future_drv,),  # (batch, horizon, 5)
    horizon=200,
)
```

## Gotchas

- `input_size` on `ESNLayer` must match driver feature dimension.
- If you only have drivers during training, you still need `forecast_inputs` at predict time.
- Multi-output models: first output dimension must match feedback size ([forecasting](../learn/forecasting.md)).

## See also

- [Model composition example](https://github.com/El3ssar/resdag/blob/main/examples/03_model_composition.py)
- [`ESNLayer`](../reference/layers/reservoirs.md)
