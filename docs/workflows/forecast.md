---
description: Two-phase autoregressive forecasting — warmup, feedback, driver alignment, coupled ensembles, and Lyapunov-time expectations.
---

<span class="nb-kicker">Work · Forecast</span>

# Forecast

`forecast(observed, horizon=H)` predicts the `H` samples immediately after
an observed history. Training learns `x[t] → x[t+1]`, so the last warmup readout
already estimates the first unseen sample. Forecast slot zero uses it; each
later prediction feeds the previous prediction back into the reservoir.

## Anatomy

```python
preds = model.forecast(f_warmup, horizon=2000)  # (batch, 2000, features)
error = ((preds - val) ** 2).mean()             # prepared val aligns directly

full = model.forecast(f_warmup, horizon=2000, return_warmup=True)
# (batch, warmup_steps - 1 + 2000, features): no repeated boundary prediction
```

States reset by default. Pass `reset=False` to continue current states.
`horizon` must be a positive integer; `horizon=1` returns the immediate unseen
prediction. Multi-output models return a tuple; the first output supplies
feedback and must match the feedback input width.

An explicit `initial_feedback` supplies a distinct *additional observation*
after all warmup observations. The first output predicts the sample after that
seed. This is useful for continuing a rollout with empty warmup:

```python
more = model.forecast(
    f_warmup[:, :0],
    initial_feedback=preds[:, -1:],
    horizon=100,
    reset=False,
)
```

The final returned prediction has not been consumed yet. Passing it as the next
seed consumes it exactly once. With an explicit seed, including historical
predictions gives `warmup_steps + horizon` outputs.

## Driver alignment

Training pairs `(feedback_t, driver_t) → feedback_{t+1}`. Warmup supplies the
driver needed for forecast slot zero. The remaining `H-1` outputs need drivers
beginning immediately after the warmup driver window:

```python
N, horizon = 200, 300
preds = model.forecast(
    (x[:, :N], d[:, :N]),
    forecast_inputs=d[:, N:N+horizon-1],
    horizon=horizon,
)
truth = x[:, N:N+horizon]
```

A one-step forecast needs no future driver. Explicit seeds need `H` future
drivers, beginning at the seed's own time index. Extra trailing driver samples
are accepted and ignored. [Theory · Timing](../theory/timing.md) derives these
indices and explains continuation for driven models.

To reproduce earlier low-level runs, use `alignment="legacy"` in both
`prepare_esn_data` and `forecast`. That mode retains the old skipped boundary
sample, `H` driver requirement, and `warmup_steps + horizon` combined length.
See [Migration](../migrating.md).

---

## Windowed reconstruction

`windowed_forecast()` fills the gaps in a sparsely-observed trajectory.
You hold the *full* ground-truth series but only trust it in periodic
windows; between them the model free-runs. Each cycle re-synchronizes the
reservoir on a short teacher-forced window, then forecasts across the
unobserved gap — and because the Echo State Property re-anchors the state
on every observed window, per-window error does not compound across
windows, so gaps can be filled indefinitely. It is a loop over `forecast()` (inference uses `no_grad=True` by default): the reservoir is reset once (on
the first window, and only when `reset=True`) and its state is carried
across the whole pass.

```text
[== warmup W ==][~~ gap P ~~][= teacher F =][~~ gap P ~~][= F =] ...
   observed        filled        observed       filled       obs
 (teacher-forced) (forecast)  (teacher-forced) (forecast)
```

<div class="nb-specimen" data-label="windowed_forecast.py" markdown>

```python
recon, mask = model.windowed_forecast(
    series,            # (batch, T, feedback_dim) ground truth
    predict_len=200,   # P: autonomous steps per gap
    teacher_len=40,    # F: real steps teacher-forced to re-sync
    warmup_len=200,    # initial sync window (defaults to teacher_len)
    return_mask=True,  # mask[t] True where observed, False where forecast
)
# recon: real values on observed steps, forecasts on the gaps.
# Score the steps the model never saw:
gap_rmse = ((recon[:, ~mask] - series[:, ~mask]) ** 2).mean().sqrt()
```

</div>

The returned `mask` is a 1-D boolean of length `T`: `True` on observed
(teacher-forced) steps and any untouched trailing remainder, `False` on
the forecast gaps. The gaps are *exactly* what you score —
`recon[:, ~mask]` against `series[:, ~mask]` — the values the model had to
infer without ever seeing them. Observed segments are copied verbatim, so
`recon[:, mask]` equals `series[:, mask]`.

For **driven models**, pass one full-timeline driver series per driver
*positionally*; each is sliced per cycle internally (unlike `forecast()`,
which takes separate warmup and forecast driver tuples):

```python
recon = model.windowed_forecast(feedback, driver, predict_len=100, teacher_len=50)
```

Keep `predict_len` short relative to the system's predictability horizon
(a few Lyapunov times for chaotic systems): sparse sampling still rebuilds
the attractor, but phase tracking degrades before the attractor *shape*
does. For multi-output models the first output is fed back, as in
`forecast()`, and only that channel is reconstructed.

---

## Coupled ensembles

A coupled ensemble runs N independently initialized models, trained
separately, in a single forecast loop: at every autoregressive step each
member receives the *same* aggregated output (mean, median, or a custom
module) as its next feedback. The shared signal couples the members and
suppresses any single member's error growth, and the spread between
members provides an uncertainty estimate.

```python
import torch
import resdag as rd

ensemble = rd.coupled_ensemble_esn(
    n_models=5,
    model_factory=rd.classic_esn,   # default: rd.ott_esn
    aggregate="mean",               # "median" | nn.Module: (N, B, T, F) -> (B, T, F)
    seed=42,                        # sub-model i is built under seed + i
    reservoir_size=300, feedback_size=3, output_size=3,
)

ensemble.fit((warmup,), (train,), {"output": target})   # same contract as ESNTrainer
preds = ensemble.forecast(f_warmup, horizon=400)        # (batch, 400, 3)

agg, members = ensemble.forecast(f_warmup, horizon=400, return_individuals=True)
spread = torch.stack(members).std(dim=0)                # (batch, 400, 3)
```

`fit` trains each member independently on the same data — diversity comes
purely from the random initializations (`n_workers > 1` fits members in a
thread pool). If a member diverges mid-forecast, an outlier-filtering
aggregator excludes it from the shared signal:

```python
from resdag.ensemble.aggregators import OutliersFilteredMean

ensemble = rd.coupled_ensemble_esn(
    n_models=10,
    aggregate=OutliersFilteredMean(method="z_score", threshold=2.0),
    reservoir_size=300, feedback_size=3, output_size=3,
)
```

## How far you can forecast

For chaotic systems the ceiling is the system's, not the model's: any
initial error grows like $e^{\lambda t}$, so even a near-perfect one-step
predictor diverges after a few Lyapunov times. A well-tuned ESN tracks
the true trajectory for several Lyapunov times; after that the pointwise
forecast decorrelates, but the trajectory remains on the attractor, so
the forecast stays statistically consistent with the system after it
stops being pointwise accurate.

<figure markdown>
![True Lorenz attractor beside the attractor traced by the autonomous ESN forecast](../assets/figures/lorenz_phase.png)
<figcaption>The true Lorenz attractor and the attractor traced by 4,000
autonomous forecast steps. Past the valid horizon the forecast
decorrelates from the true trajectory but continues to reconstruct the
attractor's geometry.</figcaption>
</figure>

If forecasts diverge well before this point, tune the hyperparameters
before increasing reservoir size. To quantify how far a forecast tracks —
and to report it in the hardware- and system-independent unit of Lyapunov
times — see [Work · Lyapunov time](lyapunov.md).

## Locking the forecast — golden regression fixtures

The forecast path is the library's most-revised code, so its behaviour is
pinned by committed *golden trajectories*. For every forecast-capable premade
model (`classic_esn`, `ott_esn`, `power_augmented`) plus an NG-RC forecaster, a
fully-seeded warmup → forecast on a canonical Lorenz-63 series is checked in
under `tests/fixtures/golden_forecast/` and asserted by
`tests/test_models/test_golden_forecast.py`:

- the first forecast steps must match the golden within a tight tolerance — the
  structural guard that catches any silent shift in the autoregressive loop
  (slot-0, the warmup/forecast seam, the flat single-step engine);
- the Valid Prediction Time against the true continuation must clear a committed
  floor — a coarser "forecast quality didn't collapse" guard.

If you intentionally change the forecast path (or the canonical data
generators), regenerate the goldens and review the diff:

```bash
python tools/regen_golden_forecasts.py --check   # verify only — write nothing (CI-style drift check)
python tools/regen_golden_forecasts.py --all     # rewrite every fixture
```

## Next

- [**Lyapunov time**](lyapunov.md) — score forecast quality and report horizons in Lyapunov times
- [**Tune**](tune.md) — hyperparameter optimization for forecast quality
- [Theory · Timing](../theory/timing.md) — derivation of every index in the forecast loop
