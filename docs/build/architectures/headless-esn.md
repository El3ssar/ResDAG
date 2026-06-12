---
description: Reservoir-only feature extractor with no readout, returning raw state sequences for use inside a larger PyTorch model.
---

<span class="nb-kicker">Build · Architecture</span>

# headless_esn

A reservoir with nothing after it: the model returns the raw state
sequence `(batch, time, reservoir_size)`. Use it as a frozen,
randomly-initialized feature extractor inside any PyTorch model you are
already training with gradients.

## Wiring

`Input → Reservoir`

The same nonlinear reservoir as [classic_esn](classic-esn.md) (tanh
activation, scaled spectrum, random bias) without the readout. The factory
accepts no readout arguments; any head applied to the states is defined in
the surrounding model.

<figure markdown>
![wiring](../../assets/figures/architectures/headless_esn.svg)
<figcaption>Model graph: the input drives a single reservoir whose states are the output.</figcaption>
</figure>

## Use

The reservoir is an `nn.Module` like any other, so it embeds directly. Its
weights are frozen by default (`trainable=False`); the optimizer only ever
sees the head's parameters:

```python
import torch
import torch.nn as nn
import resdag as rd

class Classifier(nn.Module):
    """Frozen reservoir features, gradient-trained head."""

    def __init__(self) -> None:
        super().__init__()
        self.features = rd.headless_esn(reservoir_size=300, feedback_size=3)
        self.head = nn.Linear(300, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        states = self.features(x)          # (batch, time, 300)
        return self.head(states[:, -1])    # classify on the last state

clf = Classifier()
opt = torch.optim.Adam(clf.head.parameters(), lr=1e-3)
```

The full SGD-head pattern — training loop, state resets between sequences,
GPU placement — is worked through in the
[deploy workflow](../../workflows/deploy.md).

!!! warning "The reservoir is stateful"
    States carry over between forward calls. Call
    `model.reset_reservoirs()` between independent sequences, or each
    batch starts where the last one left off.

## Parameters

| Parameter | Default | Notes |
| --- | --- | --- |
| `reservoir_size`, `feedback_size` | required | units, input dim — no `output_size` |
| `topology`, `feedback_initializer` | `None` | any [initialization spec](../initialization/index.md) |
| `spectral_radius` | `0.9` | the factory scales; the bare `ESNLayer` defaults to `None` |
| `leak_rate` | `1.0` | `1.0` = no leak |
| `activation` | `"tanh"` | also `"relu"`, `"sigmoid"`, `"identity"` |
| `bias`, `trainable` | `True`, `False` | random bias on; set `trainable=True` to backprop into the reservoir |
| `**reservoir_kwargs` | — | forwarded to `ESNLayer` (e.g. `bias_scaling`) |

## See also

- [Deploy](../../workflows/deploy.md) — the gradient-trained-head pattern in full.
- [linear_esn](linear-esn.md) — the same headless layout with identity activation, for analysis.
- [Architectures](index.md) — the other premade architectures.
