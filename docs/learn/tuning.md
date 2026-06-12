---
description: The ESN knob panel — what each hyperparameter does to the dynamics — and automated search over thousands of trials with run_hpo.
---

<span class="rd-eyebrow">Learn · 07</span>

# Tuning

By the end of this page you'll know which knob to reach for when a forecast
misbehaves, and how to hand the whole panel to Optuna when intuition runs
out. Remember the superpower from [chapter 01](reservoir-computing.md):
training takes milliseconds, so thousands of full retrainings are a search
strategy, not a luxury.

## The knob panel

| Knob | What it controls | Typical range | First thing to try |
|---|---|---|---|
| `reservoir_size` | Capacity: richness of the state features | 100–2000 | 500 |
| `spectral_radius` | Memory length vs dynamical stability | 0.5–1.5 | 0.9 |
| `leak_rate` | Reservoir timescale | 0.05–1.0 | 1.0; lower for slow signals |
| `bias_scaling` | Symmetry breaking of tanh dynamics | 0.0–1.5 | 1.0 (default) |
| `topology` | Structure of the recurrent matrix | [registry names](../cookbook/topologies.md) | `"erdos_renyi"` |
| `feedback_initializer` | How the signal enters the reservoir | [registry names](../cookbook/initializers.md) | default uniform |
| readout `alpha` | Ridge regularization of the fit | 1e-9–1e-2, log scale | 1e-6 |
| warmup length | Steps to wash out the initial state | 100–5000 | a few hundred |

## The big three

**Spectral radius is memory versus stability.** It rescales the recurrent
matrix's largest eigenvalue magnitude. Low values (0.5–0.8) make
perturbations fade fast: stable, short memory, good for signals where only
the recent past matters. Pushing toward 1.0 — sometimes slightly above —
lengthens memory at the cost of living nearer the edge of instability,
where reservoirs are often at their most expressive. If forecasts blow up,
lower it; if they look like they've forgotten the signal's structure, raise
it.

**Leak rate is the reservoir's clock.** At `leak_rate=1.0` the state is
fully replaced each step; at 0.1 it's a slow blend, $h_t = 0.9\,h_{t-1} +
0.1\,(\text{update})$. Match it to your signal: fast chaotic oscillations
want 1.0, slow drifts sampled at high rates want 0.1–0.5. A leaky reservoir
also needs a proportionally longer warmup — the state takes longer to
forget.

**Alpha is the overfitting valve.** Tiny `alpha` lets the readout chase the
training states exactly — and amplify state noise into divergence during
autoregression, when the model feeds on its own output. Larger `alpha`
gives a smoother readout that survives longer horizons at some cost in
one-step accuracy. It interacts with everything else, so sweep it (log
scale) whenever you change the reservoir.

## Searching with Optuna

Requires the extra: `pip install resdag[hpo]`. You provide three callables
— build a model from hyperparameters, define the search space, load the
data — and `run_hpo` does the rest: train, forecast, score, repeat.

<div class="rd-window" data-title="hpo.py" markdown>

```python
import resdag as rd
from resdag.hpo import run_hpo

def model_creator(reservoir_size, spectral_radius, leak_rate, readout_alpha):
    return rd.ott_esn(
        reservoir_size=reservoir_size,
        feedback_size=3,
        output_size=3,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        readout_alpha=readout_alpha,
    )

def search_space(trial):
    return {
        "reservoir_size": trial.suggest_int("reservoir_size", 200, 1000, step=100),
        "spectral_radius": trial.suggest_float("spectral_radius", 0.5, 1.2),
        "leak_rate": trial.suggest_float("leak_rate", 0.1, 1.0),
        "readout_alpha": trial.suggest_float("readout_alpha", 1e-9, 1e-3, log=True),
    }

def data_loader(trial):
    data = load_my_series()                      # (batch, time, 3)
    warmup, train, target, f_warmup, val = rd.utils.prepare_esn_data(
        data, warmup_steps=500, train_steps=10_000, val_steps=2_000,
    )
    return {"warmup": warmup, "train": train, "target": target,
            "f_warmup": f_warmup, "val": val}

study = run_hpo(
    model_creator=model_creator,
    search_space=search_space,
    data_loader=data_loader,
    n_trials=200,
    loss="efh",
    seed=42,
)
print(study.best_params, study.best_value)
```

</div>

<figure markdown>
![HPO scatter](../assets/figures/hpo_scatter.png)
<figcaption>A sweep over spectral radius and leak rate, colored by score.
The good region is a ridge, not a point — which is why multivariate TPE
(the default sampler) beats grid search here.</figcaption>
</figure>

**The loss decides what "good" means.** Five are built in, selected by
string key (or pass your own callable):

| Key | One-liner |
|---|---|
| `"efh"` | Expected forecast horizon — smooth survival-probability proxy. Default, best for chaos. |
| `"forecast_horizon"` | Hard count of contiguous steps with error below a threshold. |
| `"lyapunov"` | Time-weighted error decaying exponentially at the Lyapunov time. |
| `"standard"` | Geometric-mean error over the whole window — the plain baseline. |
| `"soft_horizon"` | Hill-gated survival variant of `efh` — sharper gate, still smooth. |

!!! tip "Parallel search"
    `run_hpo(..., n_workers=8, storage="study.log")` spawns real OS
    processes coordinated through Optuna's journal-file storage — crash
    safe, and resumable by running the same call again. Use a journal file
    (not in-memory) whenever `n_workers > 1`. More patterns in the
    [HPO recipe](../cookbook/hpo.md).

## Where next

That's the course. From here:

<div class="grid cards" markdown>

- **[Cookbook](../cookbook/index.md)**

    ---

    Task-sized recipes: custom topologies, ensembles, NG-RC, multi-readout,
    GPU, persistence.

- **[Under the hood](../under-the-hood/index.md)**

    ---

    The math behind the API: reservoir equations, the ridge solver, and
    the timing conventions everything else relies on.

</div>
