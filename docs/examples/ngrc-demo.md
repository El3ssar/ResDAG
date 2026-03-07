# NG-RC Demo

This example demonstrates Next Generation Reservoir Computing ([Gauthier et al., 2021](https://www.nature.com/articles/s41467-021-25801-2)) on the Lorenz attractor. NG-RC uses no recurrent weights — just delay embedding and polynomial monomials.

---

## Why NG-RC for Lorenz?

The Lorenz attractor is 3-dimensional with relatively simple nonlinear structure. NG-RC excels here because:

1. Delay embedding captures the attractor's geometry (Takens' theorem)
2. Degree-2 polynomials suffice for the quadratic Lorenz equations
3. No spectral radius or topology to tune

---

## Generate Data

```python
import numpy as np
import torch


def lorenz(n_steps=10000, dt=0.01, seed=0):
    rng = np.random.default_rng(seed)
    xyz = np.zeros((n_steps, 3))
    xyz[0] = rng.standard_normal(3) * 0.1
    for i in range(n_steps - 1):
        x, y, z = xyz[i]
        dxyz = np.array([10*(y-x), x*(28-z)-y, x*y-(8/3)*z])
        xyz[i+1] = xyz[i] + dt * dxyz
    return torch.tensor(xyz, dtype=torch.float32).unsqueeze(0)


data    = lorenz()
warmup  = data[:, :200,   :]
train   = data[:, 200:1200, :]
target  = data[:, 201:1201, :]
f_warm  = data[:, 1200:1700, :]
val     = data[:, 1700:2700, :]
```

---

## Build NG-RC Model

```python
import pytorch_symbolic as ps
from resdag import ESNModel
from resdag.layers import NGReservoir, CGReadoutLayer
from resdag.training import ESNTrainer

# NG-RC: k=2, s=1, p=2 → feature_dim = 1 + 6 + 28 = 35
layer = NGReservoir(input_dim=3, k=2, s=1, p=2)
print(f"Feature dimension: {layer.feature_dim}")   # 35
print(f"Warmup length:     {layer.warmup_length}") # 1 step

inp  = ps.Input((100, 3))
feat = NGReservoir(input_dim=3, k=2, s=1, p=2)(inp)
out  = CGReadoutLayer(layer.feature_dim, 3, alpha=1e-6, name="output")(feat)
model = ESNModel(inp, out)
```

---

## Train and Forecast

```python
trainer = ESNTrainer(model)
trainer.fit(
    warmup_inputs=(warmup,),
    train_inputs=(train,),
    targets={"output": target},
)

preds = model.forecast(f_warm, horizon=1000)
rmse  = torch.sqrt(torch.mean((preds - val) ** 2)).item()
print(f"NG-RC RMSE: {rmse:.4f}")
```

---

## Exploring Parameters

### Effect of Polynomial Degree

```python
import pandas as pd

results = []
for p in [1, 2, 3]:
    layer_p = NGReservoir(input_dim=3, k=2, s=1, p=p)
    feat_p  = CGReadoutLayer(layer_p.feature_dim, 3, alpha=1e-6, name="output")
    inp_p   = ps.Input((100, 3))
    model_p = ESNModel(inp_p, feat_p(layer_p(inp_p)))

    trainer_p = ESNTrainer(model_p)
    trainer_p.fit(
        warmup_inputs=(warmup,),
        train_inputs=(train,),
        targets={"output": target},
    )
    preds_p = model_p.forecast(f_warm, horizon=1000)
    rmse_p = torch.sqrt(torch.mean((preds_p - val) ** 2)).item()
    results.append({"p": p, "feat_dim": layer_p.feature_dim, "rmse": rmse_p})

df = pd.DataFrame(results)
print(df)
```

### Effect of Delay Taps

```python
results_k = []
for k in [1, 2, 3, 4]:
    layer_k = NGReservoir(input_dim=3, k=k, s=1, p=2)
    feat_k  = CGReadoutLayer(layer_k.feature_dim, 3, alpha=1e-6, name="output")
    inp_k   = ps.Input((100, 3))
    model_k = ESNModel(inp_k, feat_k(layer_k(inp_k)))

    trainer_k = ESNTrainer(model_k)
    trainer_k.fit(
        warmup_inputs=(warmup,),
        train_inputs=(train[:, layer_k.warmup_length:, :],),  # skip unfilled steps
        targets={"output": target[:, layer_k.warmup_length:, :]},
    )
    preds_k = model_k.forecast(f_warm, horizon=1000)
    rmse_k = torch.sqrt(torch.mean((preds_k - val) ** 2)).item()
    results_k.append({"k": k, "feat_dim": layer_k.feature_dim, "rmse": rmse_k})

df_k = pd.DataFrame(results_k)
print(df_k)
```

---

## ESN vs NG-RC Comparison

```python
from resdag.models import ott_esn

# ESN
esn = ott_esn(reservoir_size=300, feedback_size=3, output_size=3)
ESNTrainer(esn).fit(
    warmup_inputs=(warmup,), train_inputs=(train,), targets={"output": target}
)
esn_preds = esn.forecast(f_warm, horizon=1000)
esn_rmse  = torch.sqrt(torch.mean((esn_preds - val) ** 2)).item()

# NG-RC
ngrc = NGReservoir(input_dim=3, k=2, p=2)
inp_n = ps.Input((100, 3))
ngrc_model = ESNModel(inp_n, CGReadoutLayer(ngrc.feature_dim, 3, name="output")(ngrc(inp_n)))
ESNTrainer(ngrc_model).fit(
    warmup_inputs=(warmup,), train_inputs=(train,), targets={"output": target}
)
ngrc_preds = ngrc_model.forecast(f_warm, horizon=1000)
ngrc_rmse  = torch.sqrt(torch.mean((ngrc_preds - val) ** 2)).item()

print(f"ESN  RMSE: {esn_rmse:.4f}  (N=300, ~{300*303} params in reservoir)")
print(f"NG-RC RMSE: {ngrc_rmse:.4f}  (feat_dim={ngrc.feature_dim}, 0 reservoir params)")
```

!!! tip "When NG-RC wins"
    For low-dimensional, clean chaotic systems (Lorenz, Rössler, double pendulum),
    NG-RC often matches or beats ESN with far fewer parameters and no random initialization.
    For high-dimensional or noisy systems, ESN is typically more robust.
