# Hyperparameter Tuning Example

This example shows how to use `run_hpo()` to automatically find the best hyperparameters for a Lorenz attractor forecasting task.

---

## Setup

```bash
pip install "resdag[hpo]"
```

---

## Define the Components

```python
import numpy as np
import torch
from resdag.hpo import run_hpo, get_best_params
from resdag.models import ott_esn
from resdag.training import ESNTrainer


# --- Generate data ---
def lorenz(n=12000, dt=0.01, seed=0):
    rng = np.random.default_rng(seed)
    xyz = np.zeros((n, 3))
    xyz[0] = rng.standard_normal(3) * 0.1
    for i in range(n - 1):
        x, y, z = xyz[i]
        xyz[i+1] = xyz[i] + dt * np.array([10*(y-x), x*(28-z)-y, x*y-(8/3)*z])
    return torch.tensor(xyz, dtype=torch.float32).unsqueeze(0)


data     = lorenz()
warmup   = data[:, :500,   :]
train    = data[:, 500:4000, :]
target   = data[:, 501:4001, :]
f_warmup = data[:, 4000:4500, :]
val      = data[:, 4500:5500, :]


# --- Model creator ---
def model_creator(reservoir_size, spectral_radius, alpha, topology):
    return ott_esn(
        reservoir_size=reservoir_size,
        feedback_size=3,
        output_size=3,
        spectral_radius=spectral_radius,
        readout_alpha=alpha,
        topology=topology,
    )


# --- Search space ---
def search_space(trial):
    return {
        "reservoir_size":  trial.suggest_int("reservoir_size", 100, 1000, step=100),
        "spectral_radius": trial.suggest_float("spectral_radius", 0.7, 1.3),
        "alpha":           trial.suggest_float("alpha", 1e-9, 1e-3, log=True),
        "topology":        trial.suggest_categorical(
            "topology",
            ["erdos_renyi", "watts_strogatz", "barabasi_albert"],
        ),
    }


# --- Data loader ---
def data_loader(trial):
    return {
        "warmup":   warmup,
        "train":    train,
        "target":   target,
        "f_warmup": f_warmup,
        "val":      val,
    }
```

---

## Run the Study

```python
study = run_hpo(
    model_creator=model_creator,
    search_space=search_space,
    data_loader=data_loader,
    n_trials=50,
    loss="efh",                       # Expected Forecast Horizon
    loss_params={"threshold": 0.4},
    monitor_losses=["standard"],      # also log standard RMSE
    storage="lorenz_hpo.log",         # save to file
    seed=42,
    verbosity=1,
)

print(f"\nBest EFH:    {-study.best_value:.1f} steps")
print(f"Best params: {study.best_params}")
```

---

## Analyze Results

```python
import pandas as pd

df = study.trials_dataframe()
df["efh"] = -df["value"]  # convert negative to positive
df = df.sort_values("efh", ascending=False)

print(df[["params_reservoir_size", "params_spectral_radius",
          "params_alpha", "params_topology", "efh"]].head(10))
```

---

## Train Best Model

```python
best = study.best_params
best_model = model_creator(**best)

trainer = ESNTrainer(best_model)
trainer.fit(
    warmup_inputs=(warmup,),
    train_inputs=(train,),
    targets={"output": target},
)

# Full evaluation
preds = best_model.forecast(f_warmup, horizon=1000)
rmse  = torch.sqrt(torch.mean((preds - val) ** 2)).item()
data_std = data.std().item()
print(f"Best model RMSE: {rmse:.4f}  (normalized: {rmse/data_std:.4f})")
```

---

## Multi-Worker Parallel HPO

Scale to many trials with parallel workers:

```python
study = run_hpo(
    model_creator=model_creator,
    search_space=search_space,
    data_loader=data_loader,
    n_trials=500,
    loss="efh",
    n_workers=8,                    # 8 parallel processes
    storage="lorenz_hpo_500.log",   # required for multi-worker
    seed=0,
    verbosity=1,
    catch_exceptions=True,
    clip_value=10.0,
    prune_on_clip=True,
)
```

---

## Optuna Visualization

With [Optuna's visualization tools](https://optuna.readthedocs.io/en/stable/reference/visualization/):

```python
import optuna

# Load study from file
study = optuna.load_study(
    study_name=None,   # use first study in file
    storage=optuna.storages.JournalStorage(
        optuna.storages.JournalFileBackend("lorenz_hpo.log")
    ),
)

# Optimization history
fig = optuna.visualization.plot_optimization_history(study)
fig.show()

# Parameter importances
fig = optuna.visualization.plot_param_importances(study)
fig.show()

# Contour plot for most important params
fig = optuna.visualization.plot_contour(
    study,
    params=["spectral_radius", "reservoir_size"],
)
fig.show()
```

---

## Tips

- Start with `n_trials=30–50` for a quick scan, then expand to `n_trials=200+` for refinement
- Use `loss="efh"` for chaotic systems — it's more robust than single-run metrics
- Log `monitor_losses=["standard"]` to understand the landscape from multiple perspectives
- Use `catch_exceptions=True` with large reservoirs that might OOM
- Save to a file (`storage="*.log"`) so you can resume or analyze after the fact
