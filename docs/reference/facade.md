---
description: ESN is a PyTorch preset with the common Model training, forecasting and state workflow.
---

<span class="nb-kicker">Reference</span>

# ESN preset

`rd.ESN` is a standard `torch.nn.Module` for a classic reservoir and readout.
It uses the same training-only scaling, original units, state handling and
forecast timing as [`Model`](../start/model-workflow.md).

```python
import torch
import resdag as rd

series = torch.stack((torch.linspace(0, 8, 160).sin(),
                      torch.linspace(0, 8, 160).cos()), dim=-1)
esn = rd.ESN(reservoir_size=24, washout=10, seed=3).fit(series)
prediction = esn.forecast(horizon=20)  # Tensor, (20, 2), original units
raw_graph = esn.graph                # ESNModel, for already-normalized inputs
```

A 2-D input stays 2-D. A batched `(batch, time, features)` input keeps its batch
axis even when `batch=1`. Ordinary `esn(x)` carries state and retains gradients;
call `esn.reset_state()` between unrelated sequences. `esn.forward_stateless(x)`
returns predictions and explicit recurrent states without changing stored state.
`prediction_lead` sets the target's distance into the future. Its default
feedback delay follows that lead; explicit connections use the preset's
`"input"` and `"output"` ports. These controls and `forecast(execution="history")`
use the same rules as the [common model workflow](../start/model-workflow.md).

## Build before attaching an optimizer

Pass `input_size` to build eagerly, or call `build(width)` before collecting
parameters. Otherwise the first successful `fit` infers the feature width.

```python
pipeline_esn = rd.ESN(
    reservoir_size=16, input_size=2, washout=10, seed=4,
    readout_kwargs={"trainable": True},
).fit(series)
optimizer = torch.optim.Adam(pipeline_esn.parameters(), lr=1e-3)
pipeline_esn.reset_state()
batch = series[:8].detach().requires_grad_()
optimizer.zero_grad()
pipeline_esn(batch).square().mean().backward()
optimizer.step()
assert batch.grad is not None
```

The single registered `esn.workflow` child is a `Model`; `.graph` gives the raw
symbolic graph. Refitting preserves the existing parameter objects. Use
`Model(inputs=..., outputs=...)` for custom branches, drivers and multiple heads.

## Placement, solvers and checkpoints

An untouched lazy preset adopts the first floating tensor's device and dtype.
Explicit constructor settings and pre-build `.to(...)`/`.double()` conversions
fix the current placement. Later refits require matching tensors, as ordinary
PyTorch modules do. `build(width, device=..., dtype=...)` selects placement
explicitly; use `.to(...)` to move an already-built preset.

`readout="svd"` selects a different readout, while `readout_kwargs` passes solver
settings such as `solver`, `gram_dtype`, `use_float64` and `trainable`. Other
reservoir options and registered/custom initializers pass through the preset.
For native MPS fitting, select `readout_kwargs={"solver": "solve",
"use_float64": False, "gram_dtype": torch.float32}`; the default float64
Cholesky solve needs CPU/CUDA support. A CPU-fitted model can also be converted
with `.float().to("mps")` for predictions and upstream gradients.

A matching preset can restore a `state_dict` before its input width is built.
Checkpoint metadata validates preset architecture settings before weight copies;
scaling, observed history and trajectory sampling are restored by `Model`.
Custom Python callables and initializer implementations must remain available
with matching configuration. This is standard checkpoint loading; complete
self-describing model artifacts remain a separate roadmap item.

With the optional `dynamics` extra, `fit(trajectory)` and timestamped forecasts
work exactly as in the [trajectory guide](../start/trajectory-workflow.md).
Ordinary calls always return Torch tensors; `forecast(return_type="tensor")`
keeps trajectory-trained predictions on the model device.

See the [ESN migration guide](../project/esn-migration.md) for the breaking API
changes.

::: resdag.facade
    options:
      members:
        - ESN
