---
description: The knobs that move forecast quality — what each controls, where to start, and complete hyperparameter studies with run_hpo.
---

<span class="nb-kicker">Work · Tune</span>

# Tune

Eight knobs account for nearly all forecast-quality variance. Move them
by hand first — each has a direction you can reason about — then hand the
interactions to `run_hpo`.

| Knob | Controls | Range | First move |
| --- | --- | --- | --- |
| `reservoir_size` | model capacity | 100–2000 | double it until gains stall |
| `spectral_radius` | memory horizon, stability | 0.8–1.2 | sweep ±0.1 around 0.9 |
| `leak_rate` | state update timescale | 0.05–1.0 | lower it for slow signals |
| `bias_scaling` | diversity of unit operating points | 0.0–2.0 | leave at 1.0; `0.0` recovers pre-0.5 dynamics |
| `topology` | recurrent connectivity structure | registry names | default (dense random) first, graphs after |
| `feedback_initializer` | how the signal enters | registry names | default first; scale before structure |
| `alpha` | readout ridge regularization | 1e-8–1e-2 | step by ×100 on a log scale |
| warmup length | state synchronization | 100–500 steps | raise it until forecasts stop changing |

**Spectral radius** scales the recurrent matrix so its largest eigenvalue
magnitude hits the target, which sets how long an input echoes through
the state. Small values forget fast and stay stable; values near one
remember longer and produce richer transients at the edge of instability
— for driven, leaky reservoirs the best value often sits slightly above
1.0. This is the single most effective knob for forecast horizon.
Factories default to 0.9; a bare `ESNLayer` leaves the matrix unscaled
unless you pass a value.

**Leak rate** blends the previous state into the new one:
$x_t = (1-a)\,x_{t-1} + a\,\tilde{x}_t$. At `1.0` the state is fully
replaced each step (the standard ESN); smaller values integrate slowly,
matching the reservoir's timescale to the signal's. Slow, smooth signals
want small leaks — pairing `leak_rate=0.1` with a radius near 1.0 is the
classic recipe for long oscillations.

**Alpha** is the bias–variance dial on the only part that trains. Too
small and the readout amplifies state noise — forecasts blow up within a
few steps; too large and everything over-smooths toward the mean. The
optimum shifts with reservoir size and training length, which is why it
belongs in every search space.

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

def model_creator(reservoir_size, spectral_radius, leak_rate, alpha):
    """One fresh model per trial. Must accept every key of search_space."""
    return rd.classic_esn(
        reservoir_size=reservoir_size, feedback_size=3, output_size=3,
        spectral_radius=spectral_radius, leak_rate=leak_rate,
        readout_alpha=alpha,
    )

def search_space(trial):
    return {
        "reservoir_size": trial.suggest_int("reservoir_size", 100, 500, step=100),
        "spectral_radius": trial.suggest_float("spectral_radius", 0.5, 1.2),
        "leak_rate": trial.suggest_float("leak_rate", 0.2, 1.0),
        "alpha": trial.suggest_float("alpha", 1e-8, 1e-2, log=True),
    }

def data_loader(trial):
    """Must return exactly these five keys, each (batch, time, features)."""
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
last one sliced with the [pinned alignment](forecast.md).

!!! warning "Silent failures by design"
    `catch_exceptions=True` (the default) converts any per-trial error
    into `penalty_value` (1e10) and stashes the message in the trial's
    `"error"` attribute, so one bad configuration cannot kill a
    night-long study. A study full of 1e10s is a study full of unread
    exceptions — pass `catch_exceptions=False` while debugging.

### Losses

| Key | Behavior |
| --- | --- |
| `"efh"` | Expected forecast horizon — soft survival sum (default, chaotic systems) |
| `"forecast_horizon"` | Hard count of contiguous below-threshold steps |
| `"lyapunov"` | Error weighted by $e^{-t/\tau}$ with `lyapunov_t` |
| `"standard"` | Mean geometric-mean error — stable or periodic signals |
| `"soft_horizon"` | Hill-gated survival horizon — smoother search landscape |

All are minimized; the horizon losses are negated internally, so lower
study values mean longer usable forecasts. `loss_params` passes kwargs
(thresholds, metrics) to the chosen loss; `monitor_losses=["standard"]`
logs extra metrics on every trial without optimizing on them.

### Many workers, one study

```python
study = run_hpo(..., n_trials=200, n_workers=8, storage="study.log", seed=42)
```

Workers are real OS processes coordinating through the storage file —
journal (`"study.log"`) is append-only and built for this; a `.db` path
gets SQLite with WAL mode. BLAS threads are throttled to one per worker
before forking, and re-running with the same storage and study name
resumes where the study stopped.

Seed semantics differ by mode. Single-worker runs are fully
reproducible: `seed` seeds the sampler, and each trial reseeds
torch/numpy with `seed + trial.number`. Multi-worker runs derive a
sampler seed per worker (`seed + i·7919`), but trial numbers are claimed
in arrival order — so per-trial seeds, and hence the exact trial
sequence, vary run to run even with a fixed `seed`.

<figure markdown>
![HPO study scatter](../assets/figures/hpo_scatter.png)
<figcaption>A study's trials scattered over two hyperparameters — the
basin around the optimum is broad, which is typical: reservoirs are
forgiving once the big three are in range.</figcaption>
</figure>

## Next

- [**Scale & deploy**](deploy.md) — run the winning configuration at size
- [Reference · HPO](../reference/hpo.md) — every `run_hpo` parameter
