---
description: Move to the common PyTorch ESN workflow with explicit layouts, scaling and state.
---

# ESN workflow migration

`rd.ESN` now uses the same `nn.Module` workflow as a custom `rd.Model`.
The previous facade is replaced directly. There is no compatibility class.

| Previous behavior | Current workflow |
| --- | --- |
| NumPy input produced NumPy forecasts | Supply floating Torch tensors; forecasts are Torch tensors. Trajectories use the optional dynamics integration. |
| A batch of size one was squeezed | The input layout is preserved: `(1, T, D)` forecasts stay `(1, H, D)`. |
| `esn.forecast(100)` supplied a positional horizon | Use `esn.forecast(horizon=100)`. The optional first argument is observation history. |
| `.model` exposed the raw graph | Use `.graph`; `.workflow` is the registered common Model child. |
| `.feature_size` was inferred on fitting | `.input_size` is known after construction with a width, `build(width)` or successful fitting. |
| The facade had its own conversion and fitting logic | Training-only scaling, explicit targets and forecasts all use Model's implementation. |

Convert external arrays explicitly at the boundary:

```python
import numpy as np
import torch
import resdag as rd

external_array = np.sin(np.linspace(0, 6, 100))[:, None]
series = torch.as_tensor(external_array, dtype=torch.float64)
esn = rd.ESN(reservoir_size=16, washout=5, seed=5).fit(series)
forecast = esn.forecast(horizon=10)
array_for_plotting = forecast.detach().cpu().numpy()
assert forecast.shape == (10, 1)
```

Automatic standardization is now the default. Fit uses training samples after
washout and returns predictions in the original units. Set
`preprocessing="off"` when supplying your own transformed data.
Washout may be zero.

Build before attaching an optimizer so it sees every parameter:

```python
trainable = rd.ESN(
    reservoir_size=12, input_size=1, washout=5,
    dtype=torch.float64, readout_kwargs={"trainable": True}, seed=6,
)
optimizer = torch.optim.Adam(trainable.parameters(), lr=1e-3)
trainable.fit(series)
parameter_ids = [id(parameter) for parameter in trainable.parameters()]
trainable.build(1)  # Same-width calls preserve the existing parameters.
assert parameter_ids == [id(parameter) for parameter in trainable.parameters()]
```

Without a width, first fitting stages a candidate and attaches it after success.
A failed first fit leaves the preset unbuilt. Refitting preserves the existing
workflow, parameters and streaming state when fitting fails.

Ordinary `esn(x)` is differentiable and carries state between calls. Reset it
when switching independent sequences, or thread explicit states with
`forward_stateless`. Forecasting resynchronizes on supplied history or the
cached training tail and begins at the immediate unseen sample.

Use `.to(...)`, `.float()` and `.double()` for standard module conversion.
A conversion before lazy fitting fixes the current placement; subsequent data
must match. There are no facade-specific `.device` or `.dtype` shadow fields.
A fixed `seed` initializes the whole preset, including unfitted readout weights,
while restoring the surrounding Torch CPU/available accelerator RNG streams.
Unseeded construction follows ordinary Torch random-state behavior.

A `state_dict` includes the fitted workflow and width. Recreate the same preset
architecture/settings before loading; device and dtype may differ. Metadata
checks cannot verify the full behavior of arbitrary custom Python functions.
Complete artifact loading without reconstructing the architecture belongs to
the artifact roadmap phase.
