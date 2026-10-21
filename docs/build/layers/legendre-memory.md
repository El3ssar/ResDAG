---
description: A fixed Legendre delay network that represents recent signal history with a linear state-space memory.
---

# Legendre delay memory

`LegendreDelayLayer` compresses recent signal history into Legendre polynomial
coefficients. Its memory dynamics are fixed and linear; you can fit an ordinary
readout on top or combine it with a nonlinear reservoir.

```python
from resdag import LegendreDelayLayer

memory = LegendreDelayLayer(order=8, feedback_size=2, theta=50.0, dt=1.0)
# Input: (batch, time, 2). Output: (batch, time, 16).
```

`order` controls coefficients **per channel**, not total layer width. By
default, every feedback/driver feature gets its own memory channel. In the
example, two channels times eight coefficients gives 16 features.
`memory_channels=` instead chooses a projected channel count; use
`feedback_initializer` and `input_initializer` to customize the input encoders.

`theta` is the nominal history window in the same units as `dt`. With samples
one second apart, `theta=50, dt=1` targets roughly 50 seconds of history.
Increasing `order` improves the representation within that window at the cost
of a larger state. A finite-order delay network approximates a sliding history;
it does not store every past sample exactly.

The continuous Legendre matrices are discretized with an augmented matrix
exponential under a zero-order-held input. They are registered buffers, move
with `.to()`, and are saved with `state_dict()`. Their native spectrum is part
of the memory definition, so there is no arbitrary `spectral_radius` knob.
`trainable=True` allows input encoders to train while leaving the analytical
memory dynamics fixed.

The state and output share width `order * memory_channels`. Coefficients are
packed channel by channel. Both standard and explicit-state sequence paths
work, as do downstream readout fitting and checkpointing.

This is the fixed Legendre delay-network memory described in the
[Legendre Memory Unit paper](https://proceedings.neurips.cc/paper/2019/hash/952285b9b7e7a1be5aa7849f32ffff05-Abstract.html).
It implements the memory component; a learned nonlinear hidden unit is a
separate component you can compose around it.

See [layer composition](index.md), [training](../../workflows/train.md),
and [API reference](../../reference/layers.md).
