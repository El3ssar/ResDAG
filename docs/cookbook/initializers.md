---
description: Input and feedback weight initializers — 11 built-ins, plain functions, torch.nn.init, and how to register your own with one decorator.
---

<span class="rd-eyebrow">Cookbook</span>

# Input & feedback initializers

The input and feedback matrices decide how signals enter the reservoir, and
their scale is the single most common source of bad forecasts: too large
saturates `tanh`, too small makes the input invisible. Structure matters
too — anchoring channels to specific units changes which neurons specialize.

## Five ways to specify an initializer

<div class="rd-window" data-title="initializer_specs.py" markdown>

```python
import torch
from resdag import ESNLayer
from resdag.init.input_feedback import get_input_feedback

# 1. Registry name — uses the registered defaults
ESNLayer(500, feedback_size=3, feedback_initializer="chebyshev")

# 2. Name + parameter overrides
ESNLayer(500, feedback_size=3, feedback_initializer=("random", {"input_scaling": 0.5}))

# 3. Pre-configured object
ESNLayer(500, feedback_size=3,
         feedback_initializer=get_input_feedback("binary_balanced", input_scaling=0.5))

# 4. Any callable fn(rows, cols, **kw) -> (rows, cols) matrix
def first_neuron_only(rows, cols, scale=1.0):
    w = torch.zeros(rows, cols)
    w[0, :] = scale
    return w

ESNLayer(500, feedback_size=3, feedback_initializer=first_neuron_only)
ESNLayer(500, feedback_size=3, feedback_initializer=(first_neuron_only, {"scale": 0.1}))

# 5. torch.nn.init in-place functions work as-is
ESNLayer(500, feedback_size=3, feedback_initializer=torch.nn.init.xavier_uniform_)
```

</div>

Formats 4 and 5 are wrapped in
[`FunctionInitializer`](../reference/init/input-feedback.md) automatically —
build-style `fn(rows, cols, **kw)` is tried first, then in-place
`fn(tensor, **kw)`. The same formats work for `input_initializer`, used for
driving inputs when `input_size` is set. A third entry path is the bias:
random since v0.5, scaled by `bias_scaling` (default 1.0; set 0.0 to
restore the pre-0.5 zero bias).

## The 11 built-ins

<figure markdown>
![Weight matrices for all 11 input/feedback initializers](../assets/figures/initializers_grid.png)
<figcaption>Every registered initializer at reservoir = 96, feedback = 12. Red positive, blue negative.</figcaption>
</figure>

| Name | Character |
|---|---|
| `random` | i.i.d. uniform in [-1, 1]; the robust default. |
| `random_binary` | i.i.d. {-1, +1} entries. |
| `binary_balanced` | Walsh–Hadamard {-1, +1}: balanced columns, low correlation. |
| `chebyshev` | Deterministic weights from Chebyshev-map dynamics — reproducible without seeding. |
| `chessboard` | Alternating {-1, +1} pattern. |
| `pseudo_diagonal` | Block-diagonal structure; each channel owns a band of units. |
| `opposite_anchors` | Each channel anchored at opposite points of the reservoir. |
| `ring_window` | Windowed inputs onto a ring-structured reservoir. |
| `chain_of_neurons_input` | One unit per channel — extreme sparsity. |
| `dendrocycle_input` | Companion to the `dendrocycle` topology; feeds the core ring. |
| `zeros` | No connections — proves the model actually needs the signal. |

`show_input_initializers()` lists these at runtime; `show_input_initializers("chebyshev")` prints one initializer's parameters.

## Registering your own

`@register_input_feedback` accepts plain functions or classes. The function
form is the easy path — write the matrix logic, decorate, done:

```python
import torch
from resdag.init.input_feedback import register_input_feedback

@register_input_feedback("sparse_columns", density=0.1, scale=1.0)
def sparse_columns(rows, cols, density=0.1, scale=1.0):
    mask = torch.rand(rows, cols) < density   # each channel touches ~density
    return mask * torch.empty(rows, cols).uniform_(-scale, scale)

reservoir = ESNLayer(500, feedback_size=3,
                     feedback_initializer=("sparse_columns", {"density": 0.05}))
```

Reach for a class when the initializer carries state or configuration:

```python
from resdag.init.input_feedback import InputFeedbackInitializer

@register_input_feedback("orthogonal_input", gain=1.0)
class OrthogonalInputInitializer(InputFeedbackInitializer):
    def __init__(self, gain: float = 1.0) -> None:
        self.gain = gain

    def initialize(self, weight: torch.Tensor, **kwargs) -> torch.Tensor:
        torch.nn.init.orthogonal_(weight, gain=self.gain)
        return weight
```

!!! tip "Debugging with `zeros`"
    Set `feedback_initializer="zeros"` and confirm the forecast collapses.
    If it doesn't, the readout is solving the task without the reservoir.

## Related

- [Topologies](topologies.md) — the same spec formats, for the square recurrent matrix.
- [Custom components](custom-components.md) — registration rules across the whole library.
- [Tuning](../learn/tuning.md) — where input scaling sits among the knobs that matter.
