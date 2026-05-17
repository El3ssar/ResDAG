# Hyperparameter optimization

Requires `pip install resdag[hpo]`.

## Minimal study

```python
import torch
from resdag.hpo import run_hpo, get_study_summary
from resdag.models import ott_esn


def model_creator(reservoir_size, spectral_radius, leak_rate):
    return ott_esn(
        reservoir_size=reservoir_size,
        feedback_size=3,
        output_size=3,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
    )


def search_space(trial):
    return {
        "reservoir_size": trial.suggest_int("reservoir_size", 200, 600, step=100),
        "spectral_radius": trial.suggest_float("spectral_radius", 0.5, 1.2),
        "leak_rate": trial.suggest_float("leak_rate", 0.1, 1.0),
    }


def data_loader(trial):
    data = ...  # (1, T, 3) tensor
    return {
        "warmup": data[:, :100, :],
        "train": data[:, 100:600, :],
        "target": data[:, 101:601, :],
        "f_warmup": data[:, 600:700, :],
        "val": data[:, 700:900, :],
    }


study = run_hpo(
    model_creator=model_creator,
    search_space=search_space,
    data_loader=data_loader,
    n_trials=50,
    loss="efh",
    loss_params={"threshold": 0.2},
    storage="study.log",  # recommended for n_workers > 1
)

print(get_study_summary(study))
```

## Loss keys

Use registry names: `efh`, `forecast_horizon`, `lyapunov`, `standard`, `soft_horizon`
(see [chaos & losses](../learn/chaos-and-losses.md)).

## Parallel workers

```python
study = run_hpo(..., n_workers=4, storage="study.log")
```

Journal storage avoids SQLite locking across processes.

## Gotchas

- `data_loader` must return all five keys: `warmup`, `train`, `target`, `f_warmup`, `val`.
- `target` is **one-step-ahead** of `train` along time.
- `model_creator` must accept every key from `search_space`.

## See also

- [`run_hpo`](../reference/hpo/run.md)
- [Example 10](https://github.com/El3ssar/resdag/blob/main/examples/10_hpo.py)
