# Your first ESN

A minimal sine-wave forecast: build a model, fit the readout, predict ahead.

## 1. Imports and data

```python
import torch
from resdag import classic_esn
from resdag.training import ESNTrainer

torch.manual_seed(0)
t = torch.linspace(0, 4 * torch.pi, 400)
data = torch.sin(t).view(1, -1, 1)  # (batch=1, time, features=1)
```

## 2. Splits

We need warmup (state sync), training (fit readout), and a segment to forecast from.

```python
warmup = data[:, :200, :]
train = data[:, 200:350, :]
target = data[:, 201:351, :]  # one-step-ahead targets
f_warmup = data[:, 350:380, :]
```

## 3. Model

`classic_esn` wires reservoir → readout for you ([API](../reference/models.md)).

```python
model = classic_esn(
    reservoir_size=200,
    feedback_size=1,
    output_size=1,
    spectral_radius=0.9,
)
```

## 4. Train

The readout layer is named `"output"` by default — `targets` keys must match.

```python
trainer = ESNTrainer(model)
trainer.fit(
    warmup_inputs=(warmup,),
    train_inputs=(train,),
    targets={"output": target},
)
```

## 5. Forecast

```python
pred = model.forecast(f_warmup, horizon=50)
print(pred.shape)  # torch.Size([1, 50, 1])
```

## What each step did

| Step | Concept page |
|------|----------------|
| Reservoir + readout graph | [Reservoir layers](../learn/reservoir-layers.md) |
| `ESNTrainer.fit` | [Two-phase training](../learn/two-phase-training.md) |
| `model.forecast` | [Forecasting](../learn/forecasting.md) |

## Next

[Lorenz walkthrough](lorenz-walkthrough.md) for a chaotic 3D system with `ott_esn`.
