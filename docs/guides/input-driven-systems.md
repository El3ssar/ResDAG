# Input-driven systems

Exogenous inputs enter the reservoir as additional arguments to `ESNLayer.forward`.
The first symbolic input remains feedback.

```python
import pytorch_symbolic as ps
import torch
from resdag.core import ESNModel
from resdag.layers import ESNLayer
from resdag.layers.readouts import CGReadoutLayer
from resdag.training import ESNTrainer

feedback = ps.Input((1, 1))
driver = ps.Input((1, 5))

reservoir = ESNLayer(400, feedback_size=1, input_size=5, spectral_radius=0.9)(
    feedback, driver
)
readout = CGReadoutLayer(400, 1, name="output")(reservoir)
model = ESNModel([feedback, driver], readout)

ESNTrainer(model).fit(
    warmup_inputs=(warmup_fb, warmup_drv),
    train_inputs=(train_fb, train_drv),
    targets={"output": target},
)

pred = model.forecast(
    (f_warmup_fb, f_warmup_drv),
    forecast_inputs=(future_drv,),
    horizon=val.shape[1],
)
```

Use `prepare_esn_data` on each channel or stack features if they share the same
time base. `forecast_inputs` supplies drivers only; feedback is autoregressive.

## Driver alignment

`forecast_inputs` must be the driver series **continuing exactly where the
warmup drivers ended**. If the full driver series is `d` and warmup used
`d[:, :T]`, pass `d[:, T:T+horizon-1]` — or `d[:, T:T+horizon]`, whose last
step is accepted but unused, convenient when slicing the same window as the
validation targets. The model pairs each autoregressive feedback with the
driver of the *same* timestep, matching how the readout was trained
(`target = feedback shifted by 1`). See
[Timing & alignment](../under-the-hood/timing-and-alignment.md) for the full
index-by-index walkthrough.
