# Hyperparameter optimization

Requires `pip install resdag[hpo]`.

`run_hpo` is the high-level entry point. You supply three callables —
`model_creator`, `search_space`, `data_loader` — plus a loss name from the
registry, and it returns a completed Optuna `Study`.

<figure markdown>
  ![HPO scatter plot](../assets/figures/hpo_scatter.png){ width="640" }
  <figcaption>Two-parameter sweep over <code>spectral_radius</code> and
  <code>leak_rate</code>, coloured by score (higher = better). The star
  marks the best trial. With a real Optuna study you would slice this
  plot per parameter pair, plot the score history, and inspect parameter
  importance.</figcaption>
</figure>

```python
import torch
from resdag.hpo import run_hpo, get_study_summary
from resdag.models import ott_esn
from resdag.utils.data import prepare_esn_data


def lorenz63(n_steps=30_000):
    # ... integrate as in the chaotic-systems guide ...
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
    loss="efh",          # Expected Forecast Horizon
    storage="study.log", # journal-file storage — survives crashes
    n_workers=4,         # real OS processes, journal-coordinated
)
print(get_study_summary(study))
```

## Loss registry

The loss is selected by string key (or by passing a callable directly):

| Key | Function | Use case |
|---|---|---|
| `"efh"` | `expected_forecast_horizon` | **Default for chaos.** Smooth, differentiable proxy for forecast horizon. |
| `"forecast_horizon"` | `forecast_horizon` | Contiguous valid-step count. |
| `"lyapunov"` | `lyapunov_weighted` | Exponentially time-weighted error. |
| `"standard"` | `standard_loss` | Geometric-mean error baseline. |
| `"soft_horizon"` | `soft_valid_horizon` | Hill-gate survival probability variant. |

Full signatures and formulas in [`resdag.hpo.losses`](../reference/hpo/losses.md).

## Parallelism

`n_workers > 1` spawns real OS processes coordinated through the file
storage backend. `JournalFileStorage` (the default when you pass
`storage="study.log"`) tolerates concurrent writes cleanly; SQLite works
too with WAL mode enabled automatically. BLAS/OpenMP threads are
throttled to 1 per worker before forking to avoid oversubscription.

## Monitor losses

You can log additional losses on every trial without optimising on them —
useful for understanding what the search is actually selecting:

```python
study = run_hpo(
    ...,
    loss="efh",
    monitor_losses=["standard", "lyapunov"],
    monitor_params={"lyapunov_weighted": {"lyapunov_t": 50}},
)
```

Each monitor value is stored on `trial.user_attrs["monitor_<fn_name>"]`.

## See also

- [`run_hpo` reference](../reference/hpo/run.md)
- [Example 10](https://github.com/El3ssar/resdag/blob/main/examples/10_hpo.py)
