# Hyperparameter optimization

Requires `pip install resdag[hpo]`.

```python
import torch
from resdag.hpo import run_hpo, get_study_summary
from resdag.models import ott_esn
from resdag.utils.data import prepare_esn_data


def lorenz63(n_steps=30_000):
    # ... integrate as in chaotic-systems guide ...
    return xyz.unsqueeze(0)


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
        "reservoir_size": trial.suggest_int("reservoir_size", 400, 1000, step=200),
        "spectral_radius": trial.suggest_float("spectral_radius", 0.5, 1.2),
        "leak_rate": trial.suggest_float("leak_rate", 0.1, 1.0),
    }


def data_loader(trial):
    data = lorenz63()
    warmup, train, target, f_warmup, val = prepare_esn_data(
        data,
        warmup_steps=3_000,
        train_steps=18_000,
        val_steps=6_000,
        discard_steps=3_000,
        normalize=True,
    )
    return {
        "warmup": warmup,
        "train": train,
        "target": target,
        "f_warmup": f_warmup,
        "val": val,
    }


study = run_hpo(
    model_creator=model_creator,
    search_space=search_space,
    data_loader=data_loader,
    n_trials=50,
    loss="efh",
    storage="study.log",
)
print(get_study_summary(study))
```

Loss keys: `efh`, `forecast_horizon`, `lyapunov`, `standard`, `soft_horizon` — see
[`LOSSES`](../reference/hpo/losses.md).
