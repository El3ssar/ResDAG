---
description: The echo-state reservoir layer — the leaky update equation, its constructor parameters, the state API, and the projected fast path.
---

<span class="nb-kicker">Build · Layers</span>

# ESNLayer

The echo-state family's layer: a recurrent reservoir whose weights are
drawn once and frozen. Per timestep it applies the leaky-ESN update

$$
h_t = (1-\alpha)\,h_{t-1} + \alpha\, f\!\left(W_{fb}\,x_t + W_{in}\,u_t + W\,h_{t-1} + b\right)
$$

and returns the full state trajectory `(batch, time, reservoir_size)`.
Every parameter is set in the constructor:

<div class="nb-specimen" data-label="esn_layer.py" markdown>

```python
from resdag import ESNLayer

reservoir = ESNLayer(
    reservoir_size=500,         # units; output is (batch, time, 500)
    feedback_size=3,            # dim of the first input (required)
    input_size=None,            # dim of exogenous drivers, if any
    spectral_radius=None,       # target rho(W); None leaves W unscaled
    bias=True,                  # random bias b ~ U(-beta, beta)
    bias_scaling=1.0,           # beta; 0.0 restores the pre-0.5 zero bias
    activation="tanh",          # "tanh" | "relu" | "identity" | "sigmoid"
    leak_rate=1.0,              # alpha in [0, 1]; < 1 slows the dynamics
    trainable=False,            # True unfreezes all weights for backprop
    feedback_initializer=None,  # builder for W_fb
    input_initializer=None,     # builder for W_in
    topology=None,              # structure of W — see Initialization
)
```

</div>

## Key parameters

`spectral_radius` rescales `W` to a target largest eigenvalue — the bare
layer defaults to `None` (unscaled), while every premade factory passes
`0.9`. `leak_rate` below 1 produces slow dynamics for slow signals.
`trainable=False` freezes every parameter, which is what makes an ESN an
ESN; flip it only when you intend to backpropagate through the dynamics.
The three structural arguments — `topology`, `feedback_initializer`,
`input_initializer` — accept the full spec grammar from
[Initialization](../initialization/index.md): registry names,
`(name, params)` tuples, bare callables, or configured objects.

!!! warning "Changed in 0.5"
    `bias=True` now draws a random bias from `uniform(-bias_scaling,
    bias_scaling)`. The bias breaks the odd symmetry of `tanh` dynamics —
    without it, negated inputs produce exactly negated states. Set
    `bias_scaling=0.0` to reproduce pre-0.5 runs, where the bias was
    zero-initialized and therefore a silent no-op.

## State

The state persists across `forward` calls. It silently re-initializes to
zeros when the incoming batch size, device, or dtype changes, and it is
detached between calls, so gradients never cross a `forward`-call
boundary. The full contract is described in
[the concepts page](../../start/concepts.md); the API in brief:

```python
reservoir.reset_state()              # forget; lazily re-initialized next forward
reservoir.reset_state(batch_size=4)  # explicit zero state
reservoir.get_state()                # clone of (batch, reservoir_size), or None
reservoir.set_state(saved)           # restore a checkpoint
reservoir.set_random_state()         # standard-normal state
```

## Cell, layer, and the fast path

Like every reservoir family, `ESNLayer` splits the work in two: an
`ESNCell` owns the weights and the single-step update, while the layer
owns the sequence loop and the state. The loop's fast path calls
`project_inputs` once to precompute $W_{fb}x + W_{in}u + b$ for every
timestep, then a fused `step` per timestep — three kernel launches per
step instead of six, which accounts for most of the GPU speedup over a
naive loop. Attribute access
delegates to the cell, so `reservoir.weight_hh` and
`reservoir.spectral_radius` work directly.

## See also

- [Initialization](../initialization/index.md) — topologies and weight builders
- [Reservoir dynamics](../../theory/dynamics.md) — analysis of the update equation
- [Layers reference](../../reference/layers.md) — full signatures
