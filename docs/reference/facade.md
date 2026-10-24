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

`input_to_readout=True` includes the current input alongside reservoir states
in the readout features. Set it to `False` for a head using reservoir memory
alone. The setting is a strict boolean and can be supplied before lazy building
or with `input_size` for eager construction. Automatic vanilla tuning tries
both values. A custom `search_space` replaces that search; an omitted setting
keeps the current recipe's value. Saving preserves the selected wiring.

`input_transforms={"input": module}` and `output_transforms={"output": module}`
override the preset's preprocessing per port. Supplied modules remain
registered while the preset is lazy, and successful building/fitting keeps
the original module and Parameter objects. A failed first fit leaves them
registered without attaching an incomplete workflow. Build before collecting
all model parameters for an optimizer. Custom modules use the reversible
protocol described in [custom preprocessing](../advanced/custom-preprocessing.md).

## Placement, solvers and checkpoints

An untouched lazy preset adopts the first floating tensor's device and dtype.
Explicit constructor settings and pre-build `.to(...)`/`.double()` conversions
fix the current placement. Later refits require matching tensors, as ordinary
PyTorch modules do. `build(width, device=..., dtype=...)` selects placement
explicitly; use `.to(...)` to move an already-built preset.

The default `readout="auto"` checks numerical safety and chooses a solve while
preserving the requested `alpha`. `readout_kwargs` configures it, for example
`{"solver": "svd"}` or `{"solve_dtype": torch.float32, "cpu_fallback": False}`.
Its `last_solve` report records the actual solver, working device/precision,
conditioning and any fallback. Native MPS fits use supported float32 operations;
an unsafe or unsupported solve warns before using CPU float64 unless fallback
was disabled. Fitted parameters and predictions remain on the model device.
See [readout solvers](../guides/readout-solvers.md) for explicit alternatives and
their own keyword settings. Reservoir options and custom initializers continue
to pass through the preset.

A matching preset can restore a `state_dict` before its input width is built.
Checkpoint metadata validates preset architecture settings before weight copies;
scaling, observed history and trajectory sampling are restored by `Model`.
Custom Python callables and initializer implementations must remain available
with matching configuration. To save the full architecture and fitted state,
use `esn.save("model.pt")` and `rd.load("model.pt")`; see
[save, load and diagrams](../workflows/save-load.md).
The current `ESN` and `Model` metadata version is 2. Older metadata is rejected;
reconstruct and refit those models before saving a current checkpoint.

With the optional `dynamics` extra, `fit(trajectory)` and timestamped forecasts
work exactly as in the [trajectory guide](../start/trajectory-workflow.md).
The named single-port spelling `fit({"input": trajectory})` also works, with
explicit trajectory targets under `{"output": target}` when needed.
Ordinary calls always return Torch tensors; `forecast(return_type="tensor")`
keeps trajectory-trained predictions on the model device.

See the [ESN migration guide](../project/esn-migration.md) for the breaking API
changes.

## Method reference

`fit` returns this same module. `forecast` produces autonomous predictions;
`forward` follows observed inputs with carried state. For differentiable
chunking, `forward_stateless` returns a prediction and named states without
changing the stored state. `get_state`, `set_state` and `reset_state` manage
streaming explicitly. Standard inherited Torch methods and checkpoint protocol
hooks remain available and are omitted from this reference.

::: resdag.facade
    options:
      members:
        - ESN
