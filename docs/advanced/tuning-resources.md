---
description: Plan tuning memory and optionally shortlist candidates before full evaluation.
---

<span class="nb-kicker">Advanced · Tune</span>

# Plan tuning resources

Ordinary tuning evaluates every candidate with the full configured seeds and
forecast starting points. These optional controls help when a search has large
models or many candidates. They do not change a model's architecture or precision.

Start with the [tuning workflow](../workflows/tune.md) if you have not tuned a model yet.

## Give tuning a memory budget

```python
model.tune(
    development,
    preset="balanced",
    horizon=100,
    n_jobs=4,
    memory_budget="8GiB",
)
```

The tuner estimates each candidate before constructing it. It admits fewer
candidates at once when their combined estimates would exceed the budget.
A candidate that cannot fit by itself is skipped and recorded in the tuning
report. A giant lazy ESN base can also be rejected before its initial reservoir
is allocated; choose a smaller base or provide a suitable construction recipe.

The budget is a **planning estimate**, not an operating-system, RSS or GPU
allocator limit. Other processes, cached allocations, Python objects, backend
workspaces and custom layers can use additional memory. Leave headroom.
The tuner does not resize a reservoir, lower precision or change your model
to make it fit.

`"8GB"` means 8,000,000,000 bytes. `"8GiB"` means 8 × 1,073,741,824 bytes.
Integer byte counts and values such as `"1.5GiB"` are also accepted.

### Separate accelerator memory

```python
from resdag.hpo import MemoryBudget

budget = MemoryBudget(
    limit="16GiB",
    device_limit="8GiB",
    worker_overhead="512MiB",
)
model.tune(development, horizon=100, n_jobs=4, memory_budget=budget)
```

`limit` covers estimated host **plus** device memory and one worker-overhead
reserve per active candidate. `device_limit` adds a separate device-only check.
The default overhead reserve is 256 MiB per candidate. This is conservative
logical accounting; host/device accounts can overlap on unified-memory systems.

Automatic estimates account for shared tensor storage, retained/staged model
copies, complete refit data, feature traces, forecast outputs and 8-byte readout
workspaces. Accelerator estimates also reserve host space for a stable CPU solve
bridge. Seeds and starting points run sequentially within a candidate, so the
estimate does not multiply its peak by their count.

Later forecast starting points can have more observed history than the original
fit. Their warmup traces are included too. History-replay execution also reserves
its growing input history and concatenation copies. A custom factory that changes
which execution mode the architecture needs must include those requirements in
its own estimate.

## Optionally shortlist candidates

```python
from resdag.hpo import Screening

model.tune(
    development,
    preset="balanced",
    horizon=100,
    screening=Screening(),
    memory_budget="8GiB",
)
```

Balanced tuning normally checks all 24 candidates with three initialization
seeds and three forecast starting points. With this optional screen, it first
checks all 24 using one seed and one starting point. The best eight then receive
the full evaluation. The winner must come from that full evaluation.

Screening saves work, but can discard a configuration that would perform well
across other seeds or starting points. Leave it disabled when every candidate
needs the full comparison.

Customize the screen explicitly:

```python
screen = Screening(
    seed_count=1,
    origin_count=2,
    keep_fraction=0.5,
    cohort_size=12,
)
```

This screens cohorts of 12 and promotes the best six from each cohort. A partial
cohort rounds promotion upward, retaining at least one candidate. Screening
counts cannot exceed the full evaluation's seed/origin counts. `cohort_size=None`
uses the requested new-trial budget as one cohort.

## Supply an estimate for a custom recipe

A custom factory can change the model's topology. With a memory budget, provide
`memory_estimator(params, context)` so the tuner can check the candidate before
calling its factory. Without a memory budget, no estimate is required.

The context contains immutable shapes, dtype/device, full refit sample counts,
shared dataset/model storage counts and readout widths. It contains no live
model or data tensors. Return a `MemoryEstimate`, or an integer for a complete
host-only estimate. The result replaces the automatic estimate; it is not an
extra surcharge. Worker overhead is added separately by `MemoryBudget`.

For a custom recipe that still builds a classic dense ESN, reuse the standard
estimator instead of writing another memory formula:

```python
from dataclasses import replace

import torch
import resdag as rd
from resdag.hpo.resources import estimate_memory

time = torch.arange(600) * 0.05
development = torch.stack((time.sin(), time.cos()), dim=-1)[:500]

def build(params, seed):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        x = rd.reservoir_input(2)
        hidden = rd.ESNLayer(params["width"], feedback_size=2)(x)
        features = rd.Concatenate()(x, hidden)
        output = rd.AutoReadoutLayer(params["width"] + 2, 2, alpha=params["alpha"])(features)
        return rd.Model(inputs={"state": x}, outputs={"prediction": output}, washout=30)

def space(trial):
    return {
        "width": trial.suggest_int("width", 64, 512),
        "alpha": trial.suggest_float("alpha", 1e-8, 1e-3, log=True),
    }

def estimate(params, context):
    # This recipe retains the classic dense ESN structure.
    dense = replace(context, preset_options={"reservoir_size": params["width"]})
    return estimate_memory({"reservoir_size": params["width"]}, dense)

model = build({"width": 64, "alpha": 1e-6}, seed=7)
model.tune(
    development,
    search_space=space,
    factory=build,
    memory_estimator=estimate,
    memory_budget="8GiB",
    preset="quick",
    horizon=100,
)
```

For a different DAG, provide an estimate tailored to its layers. Include model
copies, temporary activations, readout workspaces, transfers and the final fit
on **all** development observations. A smaller screening run does not make that
final fit smaller. You can also override the estimate for a fixed DAG with
custom layers that allocate additional temporary storage.

The custom result cannot remove storage already known to exist. The tuner
floors each account at its resident dataset/model bytes, retained prediction
prefix and reserved replay history, without adding those bytes twice. Details record
the requested totals, minimum totals and whether this floor was applied.

The advanced raw runner uses the same budget rules. Because its model factory
is arbitrary, a budget always requires a custom estimator. Its context reserves
the complete prediction prefix needed by arbitrary raw losses; that part grows
with the forecast horizon. The common workflow's streaming built-in objective
does not retain that output prefix; custom objectives receiving full predictions
do. History replay can still retain growing input histories with the built-in
objective.
