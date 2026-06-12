---
description: GPU placement, float64 trade-offs, and why batch size — not sequence length — is your free parallelism.
---

<span class="rd-eyebrow">Cookbook</span>

# GPU & performance

By the end of this page you'll know the three rules that govern ResDAG
speed: everything on one device, batch is the free axis, and the time
loop is the bottleneck you can't parallelize away.

## One device for model and data

ResDAG follows PyTorch convention — `.to(device)` moves all weights, and
reservoir states are created lazily *on the device of the data you feed*,
so move the model before warming up:

<div class="rd-window" data-title="gpu_workflow.py" markdown>

```python
import torch
from resdag import ESNTrainer
from resdag.models import ott_esn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ott_esn(reservoir_size=2000, feedback_size=3, output_size=3).to(device)
warmup, train, target = warmup.to(device), train.to(device), target.to(device)

ESNTrainer(model).fit(
    warmup_inputs=(warmup,),
    train_inputs=(train,),
    targets={"output": target},
)
prediction = model.forecast(f_warmup.to(device), horizon=1000)
```

</div>

If an input arrives on a different device (or dtype, or batch size) than
the stored state, the layer re-initializes the state to zeros there — no
crash, but your warmup synchronization is gone. Pick a device and stay
on it for the whole warmup-fit-forecast arc.

## Batch is free parallelism

Every tensor is `(batch, time, features)`, and the reservoir update is a
batched matmul. Forecasting 64 trajectories costs nearly the same GPU
wall time as forecasting one — the hardware was idle at batch 1.

The flip side: the time loop is strictly sequential (state $t$ needs
state $t-1$), so a longer sequence is a longer wall clock, full stop.
When you have throughput to spend, spend it on **bigger batches, not
longer loops** — evaluate many initial conditions, ensemble members, or
hyperparameter probes side by side.

## The readout fit: float64 by default

`CGReadoutLayer` upcasts states and targets to `float64` for the ridge
solve, then casts the solution back. That doubles the solver's memory
footprint. For very large reservoirs where the fit becomes the memory
bottleneck, opt out:

```python
from resdag import CGReadoutLayer

readout = CGReadoutLayer(8000, 3, name="output", use_float64=False)
```

The trade-off is numerical headroom: stay in `float32` only when your
states are well-scaled (`tanh` reservoirs usually are). If forecasts
degrade after switching, the precision was earning its keep.

!!! warning "One model, one thread"
    Reservoir layers mutate `self.state` on every forward call. Sharing a
    model across threads races that state — clone the model per thread or
    serialize access. Across *processes* you're fine, which is exactly
    how [HPO](hpo.md) parallelizes: `run_hpo(n_workers=4)` forks real
    processes and throttles BLAS/OpenMP to one thread per worker so they
    don't oversubscribe your cores.

## Related

- [PyTorch pipelines](pipelines.md) — where these costs show up in SGD loops.
- [Hyperparameter optimization](hpo.md) — multi-process search without CPU thrash.
- [Reservoir equations](../under-the-hood/reservoir-equations.md) — the update the time loop runs.
