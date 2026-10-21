---
description: Coupled oscillator dynamics with separate position and velocity, using the standard reservoir training interface.
---

# Oscillator reservoir

`CoRNNLayer` carries position and velocity for each unit. The readout sees
position; velocity remains part of the recurrent memory. This gives you a
different temporal response from a leaky ESN while keeping the same model,
trainer, explicit-state and forecast interfaces.

```python
from resdag import CoRNNLayer

reservoir = CoRNNLayer(
    64, feedback_size=2,
    dt=0.05, gamma=1.0, epsilon=1.0,
    recurrent_initializer=("signed_permutation", {"gain": 0.5}),
    velocity_initializer=("signed_permutation", {"gain": 0.1}),
    seed=3,
)
```

One step uses the previous position `h`, velocity `v`, and projected input `u`:

```text
v_new = v + dt * (tanh(W_h h + W_v v + u + b) - gamma*h - epsilon*v)
h_new = h + dt * v_new
```

`gamma` supplies restoring force, `epsilon` damping, and `dt` the integration
step. `velocity_initializer` initializes the matrix multiplying velocity; it
is not a reset-state initializer. The feedback and optional driver have
independent initializer specs. `trainable=False` freezes all weights by default;
`True` enables gradients through the oscillator parameters.

The internal state packs `[position | velocity]`, shaped `(batch, 2*N)`.
Outputs have width `N`. Sequence processing preprojects the input stream once,
then advances the oscillator at each timestep. `forward_stateless` and model
`as_module()` work as for other reservoirs; explicitly returned states include
both position and velocity.

Small `dt` and nonnegative damping are useful starting points, but arbitrary
random weights do not inherit an unconditional stability guarantee. Evaluate
long rollouts and held-out prediction quality for your chosen parameters.
The implementation follows the explicit-damping discretization in the
[original coRNN paper](https://openreview.net/pdf?id=F3s69XzWOia).

See [composition and training](../../start/end-to-end.md),
[initializers](../initialization/recurrent.md), and the
[cell/layer API reference](../../reference/layers.md).
