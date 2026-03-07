# HPO Overview

`run_hpo()` is the main entry point for hyperparameter optimization. It wraps the entire build → train → evaluate pipeline in an Optuna study.

---

## Installation

```bash
pip install "resdag[hpo]"
```

---

## Core API

```python
from resdag.hpo import run_hpo

study = run_hpo(
    model_creator=my_model_creator,   # Callable(**hparams) → ESNModel
    search_space=my_search_space,      # Callable(trial) → dict of hparams
    data_loader=my_data_loader,        # Callable(trial) → data dict
    n_trials=100,                      # number of trials
    loss="efh",                        # optimization objective
    n_workers=1,                       # parallel workers
    storage=None,                      # Optuna storage backend
    seed=42,                           # reproducibility
)

print(f"Best params: {study.best_params}")
print(f"Best value:  {study.best_value}")
```

---

## Defining the Search Space

The `search_space` function takes an Optuna `Trial` object and returns a dictionary of hyperparameters:

```python
def my_search_space(trial):
    return {
        "reservoir_size":  trial.suggest_int("reservoir_size", 100, 2000, step=100),
        "spectral_radius": trial.suggest_float("spectral_radius", 0.5, 1.5),
        "leak_rate":       trial.suggest_float("leak_rate", 0.1, 1.0),
        "alpha":           trial.suggest_float("alpha", 1e-8, 1e-2, log=True),
        "topology":        trial.suggest_categorical(
            "topology", ["erdos_renyi", "watts_strogatz", "barabasi_albert"]
        ),
    }
```

---

## Defining the Model Creator

The `model_creator` receives the dictionary returned by `search_space` as keyword arguments:

```python
from resdag.models import ott_esn

def my_model_creator(reservoir_size, spectral_radius, leak_rate, alpha, topology):
    return ott_esn(
        reservoir_size=reservoir_size,
        feedback_size=3,
        output_size=3,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        readout_alpha=alpha,
        topology=topology,
    )
```

---

## Defining the Data Loader

The `data_loader` receives the trial (for trial-specific data augmentation) and returns a fixed dictionary:

```python
# Pre-loaded data
warmup  = data[:, :500,   :]
train   = data[:, 500:3000, :]
target  = data[:, 501:3001, :]
f_warmup = data[:, 3000:3500, :]
val     = data[:, 3500:4500, :]

def my_data_loader(trial):
    return {
        "warmup":   warmup,     # reservoir synchronization
        "train":    train,      # readout training
        "target":   target,     # training targets
        "f_warmup": f_warmup,   # forecast warmup
        "val":      val,        # validation data (ground truth)
    }
```

### Data Loader Keys

| Key | Shape | Description |
|---|---|---|
| `"warmup"` | `(batch, warmup_steps, feat)` | Warmup data for state synchronization |
| `"train"` | `(batch, train_steps, feat)` | Training input data |
| `"target"` | `(batch, train_steps, feat)` | Training targets (often `train` shifted by 1) |
| `"f_warmup"` | `(batch, f_warmup_steps, feat)` | Forecast warmup (for validation) |
| `"val"` | `(batch, horizon_steps, feat)` | Ground truth for validation |

For input-driven models, additionally provide:
- `"warmup_drivers"` — dict mapping driver name → tensor
- `"train_drivers"` — dict mapping driver name → tensor
- `"f_warmup_drivers"` — dict mapping driver name → tensor
- `"forecast_drivers"` — dict mapping driver name → tensor

---

## Complete Example

```python
import torch
import numpy as np
from resdag.hpo import run_hpo
from resdag.models import ott_esn


def lorenz(n=12000, dt=0.01, seed=0):
    rng = np.random.default_rng(seed)
    xyz = np.zeros((n, 3))
    xyz[0] = rng.standard_normal(3)
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


def model_creator(reservoir_size, spectral_radius, alpha):
    return ott_esn(
        reservoir_size=reservoir_size,
        feedback_size=3,
        output_size=3,
        spectral_radius=spectral_radius,
        readout_alpha=alpha,
    )


def search_space(trial):
    return {
        "reservoir_size":  trial.suggest_int("reservoir_size", 100, 1000, step=100),
        "spectral_radius": trial.suggest_float("spectral_radius", 0.7, 1.2),
        "alpha":           trial.suggest_float("alpha", 1e-8, 1e-3, log=True),
    }


def data_loader(trial):
    return {
        "warmup":   warmup,
        "train":    train,
        "target":   target,
        "f_warmup": f_warmup,
        "val":      val,
    }


study = run_hpo(
    model_creator=model_creator,
    search_space=search_space,
    data_loader=data_loader,
    n_trials=50,
    loss="efh",
    seed=42,
)

print("Best parameters:", study.best_params)
print("Best EFH:",        study.best_value)
```

---

## Additional Parameters

```python
study = run_hpo(
    model_creator=...,
    search_space=...,
    data_loader=...,
    n_trials=100,
    loss="efh",
    # --- Optional ---
    loss_params={"threshold": 0.5},    # extra kwargs for the loss function
    targets_key="output",              # readout name (default "output")
    monitor_losses=["horizon", "standard"],  # extra losses to log (not optimized)
    n_workers=4,                       # parallel processes
    storage="study.log",               # journal file (for multi-worker)
    seed=42,                           # study + per-trial seeding
    device="cuda",                     # move models/data to device
    verbosity=1,                       # 0=silent, 1=normal, 2=verbose
    catch_exceptions=True,             # catch failures, return penalty
    clip_value=1e6,                    # upper-bound clamp on loss
    prune_on_clip=True,                # prune trial if loss exceeds clip_value
    sampler=None,                      # custom Optuna sampler
)
```

---

## Inspecting Results

```python
from resdag.hpo import get_best_params, get_study_summary

# Get best parameters as a dict
best = get_best_params(study)
print(best)

# Full summary DataFrame
summary = get_study_summary(study)
print(summary.head(10))

# Optuna's own interface
study.trials_dataframe()       # pandas DataFrame of all trials
study.best_trial               # best Trial object
study.best_params              # best hyperparameters
study.best_value               # best objective value
```

---

## Reproducing the Best Model

```python
best_params = study.best_params
best_model = model_creator(**best_params)

from resdag.training import ESNTrainer
trainer = ESNTrainer(best_model)
trainer.fit(
    warmup_inputs=(warmup,),
    train_inputs=(train,),
    targets={"output": target},
)

predictions = best_model.forecast(f_warmup, horizon=1000)
```
