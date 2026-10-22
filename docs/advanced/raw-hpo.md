---
description: Advanced five-array studies with raw ESNModel factories and run_hpo.
---

<span class="nb-kicker">Advanced</span>

# Raw HPO runner

Use this lower-level runner when you want to prepare every training and
forecast tensor yourself and build raw `ESNModel` graphs. The usual
[`model.tune(...)` workflow](../workflows/tune.md) handles chronological
validation, preprocessing, named ports and installing a fitted winner.
`run_hpo` returns an Optuna study and leaves winner reconstruction to you.
Its worker and scoring protocols are separate from the common tuner.

## Parameters to explore

Useful search dimensions include:

| Parameter | Controls | Range | First move |
| --- | --- | --- | --- |
| `reservoir_size` | model capacity | 100–2000 | double it until gains stall |
| `spectral_radius` | memory horizon, stability | 0.8–1.2 | sweep ±0.1 around 0.9 |
| `leak_rate` | state update timescale | 0.05–1.0 | lower it for slow signals |
| `bias_scaling` | diversity of unit operating points | 0.0–2.0 | leave at 1.0; `0.0` recovers a zero-bias reservoir |
| `topology` | recurrent connectivity structure | registry names | default (dense random) first, graphs after |
| `feedback_initializer` | how the signal enters | registry names | default first; scale before structure |
| `alpha` | readout ridge regularization | 1e-8–1e-2 | step by ×100 on a log scale |
| warmup length | state synchronization | 100–500 steps | raise it until forecasts stop changing |

**Spectral radius** scales the recurrent matrix so its largest eigenvalue
magnitude hits the target, which sets how long an input echoes through
the state. Small values forget fast and stay stable; values near one
remember longer and produce richer transients at the edge of instability
— for driven, leaky reservoirs the best value often sits slightly above
1.0. Factories default to 0.9; a bare `ESNLayer` leaves the matrix unscaled
unless you pass a value.

**Leak rate** blends the previous state into the new one:
$x_t = (1-a)\,x_{t-1} + a\,\tilde{x}_t$. At `1.0` the state is fully
replaced each step (the standard ESN); smaller values integrate slowly,
matching the reservoir's timescale to the signal's. Slow, smooth signals
call for small leak rates; `leak_rate=0.1` with a spectral radius near
1.0 is a standard configuration for long-period oscillations.

**Alpha** sets the bias–variance tradeoff of a ridge readout such as
`CGReadoutLayer`. Too small, and the readout amplifies state noise and
forecasts diverge within a few steps; too large, and predictions
over-smooth toward the mean. The optimum shifts with reservoir size and
training length, which is why it belongs in every search space.

---

## Searching with `run_hpo`

Three callables define a study: a model factory, a search space, and a
data loader. Each trial builds a fresh model, fits the readout, forecasts
over the validation window, and scores it.

<div class="nb-specimen" data-label="study.py" markdown>

```python
import resdag as rd
from resdag.hpo import run_hpo

data = ...                                  # (1, 2300, 3) — (batch, time, features)
warmup, train, target, f_warmup, val = rd.utils.prepare_esn_data(
    data, warmup_steps=200, train_steps=1600, val_steps=300
)

def model_creator(reservoir_size, spectral_radius, leak_rate, alpha, seed=None):
    """One fresh model per trial. Must accept every key of search_space."""
    return rd.classic_esn(
        reservoir_size=reservoir_size, feedback_size=3, output_size=3,
        spectral_radius=spectral_radius, leak_rate=leak_rate,
        readout_alpha=alpha, seed=seed,
    )

def search_space(trial):
    return {
        "reservoir_size": trial.suggest_int("reservoir_size", 100, 500, step=100),
        "spectral_radius": trial.suggest_float("spectral_radius", 0.5, 1.2),
        "leak_rate": trial.suggest_float("leak_rate", 0.2, 1.0),
        "alpha": trial.suggest_float("alpha", 1e-8, 1e-2, log=True),
    }

def data_loader(trial):
    """Must include these five keys, each (batch, time, features)."""
    return {"warmup": warmup, "train": train, "target": target,
            "f_warmup": f_warmup, "val": val}

study = run_hpo(
    model_creator=model_creator,
    search_space=search_space,
    data_loader=data_loader,
    n_trials=100,
    loss="efh",
    seed=42,
    storage="study.log",                    # journal file: resumable
)
print(study.best_params)
```

</div>

The data contract is validated per trial: the dict must contain
`"warmup"`, `"train"`, `"target"`, `"f_warmup"`, `"val"`, all 3-D
tensors. The forecast horizon defaults to `val.shape[1]`. Input-driven
models add `drivers_keys=["driver"]` and supply `warmup_driver`,
`train_driver`, `f_warmup_driver`, and `forecast_driver` entries — the
last one sliced with the [pinned alignment](../workflows/forecast.md).

!!! note "Invalid trials and debugging"
    `catch_exceptions=True` continues after an invalid evaluation, but the
    default `error_policy="fail"` records it as an Optuna `FAIL` trial so it
    cannot win. Details remain in the trial's `"error"` attribute. A study
    with no valid completed evaluation raises clearly. Pass
    `catch_exceptions=False` to see the original exception immediately.
    The explicit `error_policy="penalty"` retains legacy numeric penalties;
    it still cannot return an invalid best model. Low-level `TrialRunner`
    and `build_objective` keep their old penalty default for compatibility.

The actual forwarded model seed is recorded as `trial.user_attrs["model_seed"]`
when the creator accepts a seed; `"trial_seed"` records the trial RNG seed.
Use that model seed when reconstructing the winner. Forecast evaluation uses
temporary evaluation mode, restoring each module's earlier training flag.

`transfer_from` reuses evaluated scores and requires matching objective
directions. Only use it for the same data, scaling and objective. For a new
experiment, pass prior configurations through `warm_start` to evaluate them
again.

### Losses

| Key | Behavior |
| --- | --- |
| `"efh"` | Expected forecast horizon — soft survival sum (default, chaotic systems) |
| `"forecast_horizon"` | Hard count of contiguous below-threshold steps |
| `"lyapunov"` | Error weighted by $e^{-t/\tau}$ with `lyapunov_t` |
| `"standard"` | Mean geometric-mean error — stable or periodic signals |
| `"soft_horizon"` | Hill-gated survival horizon — smoother search landscape |
| `"attractor_dimension"` | Correlation-dimension mismatch for long-term attractor fidelity |

All are minimized; the horizon losses are negated internally, so lower
study values mean longer usable forecasts. `loss_params` passes kwargs
(thresholds, metrics) to the chosen loss; `monitor_losses=["standard"]`
logs extra metrics on every trial without optimizing on them.

With `torch_scoring=True` (the default), the five horizon/error losses score
directly on the forecast device; each checkpoint transfers only its final
scalar. The attractor-dimension loss uses NumPy. Custom losses can accept torch
tensors for the same device-local path, or use the automatic NumPy fallback.
Checkpoint pruning currently happens after the full trajectory is generated,
so it saves remaining scoring work, rather than forecast generation.

### Multi-worker studies

```python
study = run_hpo(..., n_trials=200, n_workers=8, storage="study.log", seed=42)
```

Workers are separate OS processes coordinating through the storage file.
Journal storage (`"study.log"`) is append-only and handles concurrent
writers; a `.db` path uses SQLite with WAL mode. BLAS threads are
throttled to one per worker before forking, and re-running with the same
storage and study name resumes where the study stopped.

Seed semantics differ by mode. Single-worker runs are fully
reproducible when the model creator accepts and forwards `seed`, as above:
`seed` seeds the sampler, and each trial reseeds torch/numpy/Python with
`seed + trial.number`. Multi-worker runs derive a
sampler seed per worker (`seed + i·7919`), but trial numbers are claimed
in arrival order — so per-trial seeds, and hence the exact trial
sequence, vary run to run even with a fixed `seed`.
An explicit `sampler=` is preserved in each worker with an independent copy
and a call to its `reseed_rng()` hook. Custom samplers must be picklable for
CUDA/spawn studies; their RNG policy determines reproducibility.

---

## Reading a study

`run_hpo` returns a plain optuna study, so optuna's visualization module
works on it directly: `optuna.visualization` for interactive plotly
figures, `optuna.visualization.matplotlib` for static ones like the four
below, drawn from a 70-trial study maximizing valid horizon on Lorenz.

```python
from optuna.visualization import plot_optimization_history   # plotly, interactive
from optuna.visualization.matplotlib import plot_contour     # static

plot_optimization_history(study)
```

<figure markdown>
![Optimization history of a 70-trial study](../assets/figures/hpo_history.png)
<figcaption>Per-trial objective with the running best.</figcaption>
</figure>

<figure markdown>
![Hyperparameter importances](../assets/figures/hpo_importances.png)
<figcaption>Relative importance of each hyperparameter for the
objective.</figcaption>
</figure>

<figure markdown>
![Per-parameter slice plots of the objective](../assets/figures/hpo_slice.png)
<figcaption>Slice plots: the objective against each parameter
individually.</figcaption>
</figure>

<figure markdown>
![Contour of the objective over spectral radius and leak rate](../assets/figures/hpo_contour.png)
<figcaption>The spectral-radius × leak-rate landscape around the
optimum.</figcaption>
</figure>

## Next

- [**Common tuning workflow**](../workflows/tune.md) — tune a preset or custom `Model`
- [**Scale & deploy**](../workflows/deploy.md) — run the best configuration at scale
- [Reference · HPO](../reference/hpo.md) — every `run_hpo` parameter
