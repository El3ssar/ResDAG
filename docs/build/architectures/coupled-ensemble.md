---
description: N independently trained sub-models coupled through a shared aggregated feedback signal during autoregression, trained and run through a single fit/forecast interface.
---

<span class="nb-kicker">Build · Architecture</span>

# coupled_ensemble_esn

N independently trained ESN models that forecast jointly: at every
autoregressive step, each sub-model receives the same aggregated output
(the ensemble mean, by default) as its next feedback input. The sub-models
differ only in their random initializations; the shared feedback signal
keeps their trajectories synchronized during forecasting.

## Wiring

The architecture is N replicas of a sub-model plus an aggregator. The
shared feedback loop exists only at forecast time:

```text
              ┌─→ sub-model 1 ─┐
feedback ──→──┼─→     ...     ─┼─→ aggregate ─→ ŷ_t
     ▲        └─→ sub-model N ─┘       │
     └─────────────────────────────────┘
        ŷ_t is every sub-model's next feedback
```

The factory builds the sub-models by calling `model_factory(**model_kwargs)`
N times (default factory: [ott_esn](ott-esn.md)) and returns a
`CoupledEnsembleESNModel`. It is not an `ESNModel`, but its `fit` and
`forecast` follow the same conventions: warmup/train tuples, a targets dict
keyed by readout name, `forecast(warmup, horizon=...)`. It also provides
`save`, `load`, `warmup`, and `reset_reservoirs`, applied across all
sub-models.

## Use

```python
import torch
import resdag as rd

series = torch.cumsum(0.1 * torch.randn(1, 1401, 3), dim=1)

ensemble = rd.coupled_ensemble_esn(
    n_models=5, seed=42,
    reservoir_size=300, feedback_size=3, output_size=3,   # → each ott_esn
)
ensemble.fit(
    (series[:, :200],), (series[:, 200:1200],),
    {"output": series[:, 201:1201]},
    n_workers=4,                       # independent readout fits, one per sub-model
)
preds = ensemble.forecast(series[:, 1200:1400], horizon=100)   # (1, 100, 3)

preds, runs = ensemble.forecast(                # per-sub-model trajectories
    series[:, 1200:1400], horizon=100, return_individuals=True,
)
```

Sub-models train independently on the same data; coupling exists only in
the forecast loop. `seed` makes construction deterministic: sub-model *i*
is built under `torch.manual_seed(seed + i)`, and `seed + i` is forwarded
to the factory when it accepts a `seed` argument.

## Aggregators

`aggregate` takes `"mean"`, `"median"`, or any `nn.Module` mapping a
stacked `(N, batch, time, features)` tensor to `(batch, time, features)`.
`OutliersFilteredMean` drops members whose output norm is an outlier
(Z-score or IQR) before averaging — useful when one diverging sub-model
would otherwise drag the shared feedback off the attractor:

```python
ensemble = rd.coupled_ensemble_esn(
    n_models=10, reservoir_size=300, feedback_size=3, output_size=3,
    aggregate=rd.OutliersFilteredMean(method="z_score", threshold=2.0),
)
```

Non-finite predictions are excluded from both robust statistics and the
average. If every member is invalid at a batch/timestep position, the
aggregator raises `ValueError` identifying that position instead of feeding
an invalid or fabricated value back into the models. If finite members all
fail the outlier threshold, their finite-only mean is the fallback.

Members must have the same input widths, device and precision. These are
checked at construction. A custom aggregator must return a tensor with the
same `(batch, time, features)` shape as one member's output.

## Parameters

| Parameter | Default | Notes |
| --- | --- | --- |
| `n_models` | required | ensemble size |
| `model_factory` | `ott_esn` | any callable returning an `ESNModel` — premade or your own |
| `aggregate` | `"mean"` | `"mean"`, `"median"`, or an `nn.Module` over `(N, batch, time, features)` |
| `seed` | `None` | per-sub-model `torch.manual_seed(seed + i)`; forwarded when the factory takes `seed` |
| `**model_kwargs` | — | forwarded verbatim to every `model_factory` call |

## Reference

The design is related to the parallel-reservoirs scheme of J. Pathak et
al., Phys. Rev. Lett. **120**, 024102 (2018), in which multiple reservoirs
jointly forecast one system through shared signals. This factory couples
complete models through one aggregated feedback signal rather than through
spatially overlapping reservoirs; it is not an implementation of that
paper.

## See also

- [ott_esn](ott-esn.md) — the default sub-model architecture.
- [Forecast](../../workflows/forecast.md) — the timing convention `forecast` and its drivers follow.
- [Models reference](../../reference/models.md) — `coupled_ensemble_esn` and `CoupledEnsembleESNModel` in full.
