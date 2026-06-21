---
description: API reference for resdag.hpo — run_hpo, the five forecast-quality loss functions, and study utilities.
---

<span class="nb-kicker">Reference</span>

# HPO

Optuna-backed hyperparameter optimization: the `run_hpo` entry point, five
loss functions built for judging forecasts of (possibly chaotic) dynamics,
and helpers for inspecting finished studies.

!!! note "Optional dependency"
    `run_hpo` and the study utilities require `optuna`
    (`pip install "resdag[hpo]"`). The loss functions import without it.

::: resdag.hpo
    options:
      members:
        - run_hpo

---

## Reproducible studies

Pass `seed=` to `run_hpo` to make a study reproducible end to end. The seed is
applied in two places:

1. **The Optuna sampler** — so the *sequence of sampled hyperparameters* is
   identical across runs.
2. **Each trial**, where the per-trial seed `seed + trial.number` seeds PyTorch,
   NumPy, and Python's `random`, *and* is threaded into your `model_creator`.

That last step is the one that matters for the reservoir. The recurrent
**topology** and the **input/feedback initializers** build their own RNGs
(`np.random.default_rng(...)` / `torch.Generator`) and do **not** read NumPy's
legacy global state — so seeding `np.random` alone leaves the reservoir weights
varying run-to-run. To close the gap, make your creator accept a `seed` keyword
and forward it to the reservoir (the premade factories forward `**kwargs`
straight to `ESNLayer`, which fixes the whole reservoir):

```python
from resdag.models import ott_esn

def model_creator(reservoir_size, spectral_radius, seed=None):
    # `seed` reaches ESNLayer/ESNCell, fixing the topology, the
    # feedback/input initializers, and the random bias.
    return ott_esn(
        reservoir_size=reservoir_size,
        feedback_size=3,
        output_size=3,
        spectral_radius=spectral_radius,
        seed=seed,
    )

study_a = run_hpo(model_creator, search_space, data_loader, n_trials=20, seed=42)
study_b = run_hpo(model_creator, search_space, data_loader, n_trials=20, seed=42)
assert study_a.best_value == study_b.best_value  # identical
```

`run_hpo` only injects `seed` when `model_creator` declares a `seed` parameter
(or `**kwargs`); creators without one are called unchanged, and an explicit
`seed` returned by `search_space` always wins. Reproducibility is exact for
single-worker (`n_workers=1`) studies; with `n_workers > 1`, trial *scheduling*
across processes is non-deterministic, so the set of evaluated configurations is
reproducible but their order — and therefore `best_value` — may differ.

---

## Pruning (early stopping)

Each trial forecasts **once** and is scored at a handful of growing
forecast-horizon checkpoints, reporting the prefix loss at every one. A diverging
configuration already shows a large loss at an early checkpoint — so an Optuna
**pruner** can terminate it before paying for the full horizon. Pass `pruner=` to
turn this on:

```python
from resdag.hpo import run_hpo

study = run_hpo(
    model_creator, search_space, data_loader,
    n_trials=100,
    pruner="median",   # stop trials worse than the running median at each checkpoint
)

# Pruned trials are recorded but excluded from `best_value`:
print(get_study_summary(study))   # the summary lists the Pruned count
```

`pruner` accepts a registry key, a fully-configured
[`optuna.pruners.BasePruner`](https://optuna.readthedocs.io/en/stable/reference/pruners.html)
instance, or `None`:

| Key | Pruner | When to reach for it |
|---|---|---|
| `"none"` (default) | `NopPruner` | No pruning. |
| `"median"` | `MedianPruner` | Good default — prunes trials worse than the running median. |
| `"asha"` | `SuccessiveHalvingPruner` (ASHA) | Aggressive, scales well with many parallel workers. |
| `"hyperband"` | `HyperbandPruner` | ASHA across several budgets; strong when the right budget is unknown. |
| `"threshold"` | `ThresholdPruner` | Prune on an absolute loss bound (default `upper=1.0`; pass a configured instance for a different bound). |

Need to tune a pruner's knobs? Build the instance and pass it in:

```python
import optuna

study = run_hpo(
    model_creator, search_space, data_loader, n_trials=100,
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
)
```

Pruning is honored in both single- and multi-process (`n_workers > 1`) runs — the
resolved pruner is attached to every worker's study. It also composes with the
[`clip_value` / `prune_on_clip`](#hpo) bound: the clip path prunes on the *raw*
loss exceeding a ceiling, while the pruner reacts to the *intermediate* horizon
reports; both can be active at once.

---

::: resdag.hpo.losses
    options:
      members:
        - LOSSES
        - LossProtocol
        - get_loss
        - expected_forecast_horizon
        - forecast_horizon
        - lyapunov_weighted
        - standard_loss
        - soft_valid_horizon

---

::: resdag.hpo.utils
    options:
      members:
        - get_study_summary
        - make_study_name
        - get_best_params
