---
description: Define, register and reuse custom recurrent, input, bias and reset-state initializers.
---

# Customize initialization

Start with a function. Register it when you want to select it by name in
configuration files, model factories or tuning studies. Registration is local
to the current Python process: import your registration module before using a
name in another process.

## Recurrent matrices

A builder receives the reservoir size and returns an `(n, n)` tensor, NumPy
array, or graph. An in-place `torch.nn.init` function works too.

```python
import torch
from resdag import ESNLayer
from resdag.init import register_recurrent_initializer

@register_recurrent_initializer("my_cycle", prescaled=True, gain=0.9)
def my_cycle(n, gain=0.9, seed=None, *, device=None, dtype=None):
    # This matrix has an exact radius of abs(gain), so it owns its scale.
    return gain * torch.eye(n, device=device, dtype=dtype).roll(1, dims=0)

layer = ESNLayer(64, feedback_size=2, recurrent_initializer="my_cycle")
layer = ESNLayer(
    64, feedback_size=2,
    recurrent_initializer=("my_cycle", {"gain": 0.7}),
)
```

The same function can be passed directly without registration:
`recurrent_initializer=my_cycle` or `(my_cycle, {"gain": 0.7})`.
`register_graph_topology` and `register_matrix_topology` continue to work;
they share the registry used by the new recurrent API. Unknown names and
conflicting `topology` / `recurrent_initializer` specs raise errors.

For random builders, accept `seed` and use a local generator on the requested
device. A layer seed is forwarded when your function accepts it; an explicitly
bound seed wins. Your builder must return the requested shape with finite
values. A function that ignores its seed cannot be made reproducible by the
registry. Returning an already normalized matrix is optional: choose
`prescaled=True` when its native scale is meaningful, or let the layer apply
`spectral_radius` afterward.

## Input, feedback, bias and reset state

Rectangular builders receive `(rows, cols)`. They can also be registered:

```python
from resdag.init.input_feedback import register_input_feedback

@register_input_feedback("my_constant", value=0.1)
def my_constant(rows, cols, value=0.1, *, device=None, dtype=None):
    return torch.full((rows, cols), value, device=device, dtype=dtype)

layer = ESNLayer(
    64, feedback_size=2, input_size=1,
    feedback_initializer=("normal", {"scale": 0.2}),
    input_initializer="my_constant",
    bias_initializer=("my_constant", {"value": 0.0}),
    initial_state_initializer=("my_constant", {"value": 0.05}),
    seed=4,
)
```

`bias_initializer` sees a `(1, reservoir_size)` view of the bias.
`initial_state_initializer` sees `(1, state_size)` once at construction.
The resulting reset template is saved as a buffer and repeated independently
for each batch item whenever a fresh state is needed. The default is zero.
A sampled template stays identical across resets and changing batch sizes. Explicit supplied states
still take precedence. `bias=False` rejects a bias initializer instead of
silently ignoring it. For packed multi-component states, initialize the full
packed width and respect the cell's documented component order.

In-place callables such as `torch.nn.init.zeros_` and
`(torch.nn.init.normal_, {"std": 0.02})` use the same grammar. A configured
`InputFeedbackInitializer` object is accepted as-is. Its own seed and state
remain its responsibility.

## Separate gate parameters

The light-gated layer shares its input projection by default. Supply a gate
initializer to make that projection independent, and control the gate's
recurrence and bias separately:

```python
from resdag import GatedReservoirLayer

gated = GatedReservoirLayer(
    64, feedback_size=2,
    recurrent_initializer=("signed_permutation", {"gain": 0.8}),
    gate_recurrent_initializer=("signed_permutation", {"gain": 0.1}),
    gate_feedback_initializer=("normal", {"scale": 0.05}),
    gate_bias_initializer=("my_constant", {"value": 2.0}),
    bias_initializer=torch.nn.init.zeros_,
    seed=9,
)
```

`gate_input_initializer` controls the optional driver's gate projection.
An omitted gate projection reuses its candidate counterpart.
`gate_spectral_radius` controls only the gate recurrence and obeys that
initializer's scaling policy. These controls are independent of the
candidate's `spectral_radius`. New parameters appear in `state_dict` only
when configured, so rebuild the same configuration before loading weights.

## Save and share custom configurations

Weight checkpoints need the same architecture when loaded. Full-model
checkpoints additionally pickle the initializer configuration; keep custom
classes/functions at importable module scope. Import registration modules on
workers before reconstructing architectures from names. Avoid anonymous or
nested functions if you need full-model serialization.

See the [input/feedback catalog](initializers/index.md) for existing recipes,
[recurrent choices](recurrent.md) for spectral control, and
[API reference](../../reference/init.md) for subclass contracts.

## Reproducibility and numerical precision

Direct callbacks, `(callback, kwargs)` tuples and registered names use the
same seed rules. A callback accepting `seed=` receives the layer seed unless
you explicitly bind one. In-place Torch functions accepting `generator=` get
a local generator on the destination device. A bound generator takes precedence
and advances on each call; reuse a fresh generator with the same seed when you
want to reproduce its first draw. Configured initializer objects keep their own
seed policy and are used as supplied.

```python
from resdag.init import get_recurrent_initializer
from resdag.init.input_feedback import get_input_feedback

# Explicit initializer seeds take precedence over the layer seed.
recurrent = get_recurrent_initializer(
    "orthogonal", seed=11, rescale=False,
)
feedback = get_input_feedback("normal", scale=0.2, seed=12)
layer = ESNLayer(64, feedback_size=2, recurrent_initializer=recurrent,
                 feedback_initializer=feedback, seed=99)
```

For built-in NumPy-backed initializers, omitting a seed draws from a generator
derived from Torch's global RNG, so resetting `torch.manual_seed(...)` reproduces
the draw. Custom callbacks that ignore both seed and generator remain responsible
for their own randomness. Graph or NumPy float64 outputs retain their precision
when copied into float64 reservoir weights; constructing float32 values first
and converting them later cannot recover the lost digits.

Some structured initializers need options or dimensions that cannot be inferred
from the layer. For example, supply `("dendrocycle_input", {"c": 0.5})` or
`("ring_window", {"c": 0.5, "window": 5})`. These tuple forms work the same
way as configuring an initializer object explicitly. Each catalog page includes
a complete example with suitable dimensions and required parameters, and the
examples are executed in the test suite.
