# Advanced HPO Usage

---

## Multi-Worker Parallel Optimization

Run HPO across multiple CPU cores using real OS processes:

```python
study = run_hpo(
    model_creator=model_creator,
    search_space=search_space,
    data_loader=data_loader,
    n_trials=200,
    loss="efh",
    n_workers=8,                       # 8 parallel processes
    storage="my_study.log",            # journal file (required for n_workers > 1)
    seed=42,
)
```

!!! important "Storage is required for multi-worker"
    With `n_workers > 1`, a `storage` path must be provided so all workers can share trial results.
    Use `"study.log"` for the recommended journal file backend.

### How Multi-Worker Works

1. `n_trials` is evenly distributed across `n_workers` processes
2. Each process runs its own Optuna study connected to the shared storage
3. BLAS/OpenMP threads are throttled before forking to avoid thread oversubscription
4. After all processes complete, results are aggregated from storage

---

## Storage Backends

| Backend | Syntax | Notes |
|---|---|---|
| In-memory | `storage=None` | Single worker only; lost after process ends |
| Journal file | `storage="study.log"` | Recommended for multi-worker; simple, robust |
| SQLite | `storage="study.db"` or `storage="sqlite:///study.db"` | WAL mode enabled automatically |
| Full URL | `storage="postgresql://..."` | Production use |

```python
# Journal file (recommended)
study = run_hpo(..., storage="lorenz_hpo.log")

# SQLite
study = run_hpo(..., storage="lorenz_hpo.db")

# Load existing study
import optuna
existing = optuna.load_study(
    study_name="resdag_hpo",
    storage="lorenz_hpo.log",
)
```

---

## Custom Samplers

Replace the default TPE sampler with any Optuna sampler:

```python
import optuna

# CMA-ES (good for continuous search spaces)
sampler = optuna.samplers.CmaEsSampler(seed=42)

# Grid search (for small discrete spaces)
sampler = optuna.samplers.GridSampler({
    "reservoir_size": [100, 300, 500, 1000],
    "spectral_radius": [0.7, 0.9, 1.0, 1.1],
})

# Random (baseline)
sampler = optuna.samplers.RandomSampler(seed=42)

study = run_hpo(..., sampler=sampler)
```

---

## Exception Handling

By default, trial exceptions propagate and stop the study. Enable graceful exception handling:

```python
study = run_hpo(
    ...,
    catch_exceptions=True,   # catch failures, return penalty_value
    clip_value=1e6,           # clamp loss at this value
    prune_on_clip=True,       # prune (skip) trials that exceed clip_value
)
```

This is useful when:
- Some hyperparameter combinations cause numerical instability
- GPU runs out of memory for large reservoirs
- You want to continue optimization despite occasional failures

---

## Seeding for Reproducibility

```python
study = run_hpo(
    ...,
    seed=42,   # seeds the Optuna sampler AND per-trial torch.manual_seed(seed + trial.number)
)
```

This ensures:
- The HPO sampler produces the same sequence of hyperparameter proposals
- Each trial's reservoir initialization is deterministic (given its trial number)

---

## Device Placement

Move models and data to a specific device:

```python
study = run_hpo(
    ...,
    device="cuda:0",   # or "cpu", "cuda", "mps"
)
```

Data tensors in the data loader dict are automatically moved to the specified device.

---

## Verbosity

```python
# Silent (useful in production or large runs)
study = run_hpo(..., verbosity=0)

# Normal (default: trial results, best value)
study = run_hpo(..., verbosity=1)

# Verbose (all Optuna output + resdag debug info)
study = run_hpo(..., verbosity=2)
```

---

## Naming Studies

```python
from resdag.hpo import make_study_name

name = make_study_name("lorenz_ott", seed=42, loss="efh")
print(name)  # "lorenz_ott_efh_s42"

study = run_hpo(
    ...,
    storage="all_studies.db",
    # study_name=name  (pass to underlying optuna.create_study)
)
```

---

## Inspecting a Study After the Fact

```python
import optuna
from resdag.hpo import get_study_summary, get_best_params

# Load from file
study = optuna.load_study(
    study_name="my_study",
    storage="study.log",
)

# Summary DataFrame
df = get_study_summary(study)
print(df[["reservoir_size", "spectral_radius", "alpha", "value"]].sort_values("value"))

# Best parameters
best = get_best_params(study)

# Optuna visualization (requires plotly)
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_param_importances(study)
optuna.visualization.plot_contour(study, params=["spectral_radius", "reservoir_size"])
```

---

## Example: Full Production HPO Run

```python
import torch
from resdag.hpo import run_hpo
from resdag.models import ott_esn

# --- Data (pre-loaded) ---
data = torch.load("lorenz_data.pt")
warmup   = data[:, :500,   :]
train    = data[:, 500:4000, :]
target   = data[:, 501:4001, :]
f_warmup = data[:, 4000:4500, :]
val      = data[:, 4500:5500, :]


def model_creator(reservoir_size, spectral_radius, leak_rate, alpha, topology):
    return ott_esn(
        reservoir_size=reservoir_size,
        feedback_size=3,
        output_size=3,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        readout_alpha=alpha,
        topology=topology,
    )


def search_space(trial):
    return {
        "reservoir_size":  trial.suggest_int("reservoir_size", 200, 2000, step=100),
        "spectral_radius": trial.suggest_float("spectral_radius", 0.7, 1.3),
        "leak_rate":       trial.suggest_float("leak_rate", 0.3, 1.0),
        "alpha":           trial.suggest_float("alpha", 1e-9, 1e-3, log=True),
        "topology":        trial.suggest_categorical(
            "topology",
            ["erdos_renyi", "watts_strogatz", "barabasi_albert", "complete"],
        ),
    }


def data_loader(trial):
    return {"warmup": warmup, "train": train, "target": target,
            "f_warmup": f_warmup, "val": val}


study = run_hpo(
    model_creator=model_creator,
    search_space=search_space,
    data_loader=data_loader,
    n_trials=500,
    loss="efh",
    loss_params={"threshold": 0.4, "n_samples": 5},
    monitor_losses=["horizon", "standard"],
    n_workers=16,
    storage="lorenz_hpo_500.log",
    seed=0,
    device="cuda",
    verbosity=1,
    catch_exceptions=True,
    clip_value=10.0,
    prune_on_clip=True,
)

print(f"\nBest EFH: {-study.best_value:.1f} steps")
print(f"Best params: {study.best_params}")
```
