# Custom input/feedback initializer

A complete worked example: implement a brand-new
`InputFeedbackInitializer`, register it with the library, and use it
inside a model the same way you'd use any built-in initializer.

We'll build an **orthogonal** initializer — its weight matrix has
orthonormal columns scaled by a `gain`. Orthogonal feedback is a
well-known regularizer for recurrent networks: it preserves the norm of
the input vector as it enters the reservoir, avoiding the
amplitude-blow-up that random Gaussian weights can produce.

## 1. The class

A registered initializer is an `nn.Module`-style class with two
methods: `__init__` for the parameters and `initialize(weight,
**kwargs)` for the actual matrix construction. Decorate the class with
`@register_input_feedback("<name>")` to make it selectable everywhere.

```python
import torch
from resdag.init.input_feedback import (
    InputFeedbackInitializer,
    register_input_feedback,
)


@register_input_feedback("orthogonal", gain=1.0)
class OrthogonalInputInitializer(InputFeedbackInitializer):
    """Random orthonormal columns scaled by ``gain``."""

    def __init__(self, gain: float = 1.0) -> None:
        if gain <= 0:
            raise ValueError(f"gain must be > 0, got {gain}")
        self.gain = gain

    def initialize(self, weight: torch.Tensor, **kwargs) -> torch.Tensor:
        torch.nn.init.orthogonal_(weight, gain=self.gain)
        return weight
```

## 2. Verify it's registered

```python
from resdag.init.input_feedback import show_input_initializers, get_input_feedback

print("orthogonal" in show_input_initializers())   # True
show_input_initializers("orthogonal")              # shows the parameter table
```

You can also use `get_input_feedback("orthogonal", gain=0.8)` to get a
configured instance directly.

## 3. Use it in a model

The three specification forms — bare name, name + params, instance —
all work, just like for the built-ins:

```python
from resdag import classic_esn
from resdag.init.input_feedback import get_input_feedback

# string
model_a = classic_esn(400, feedback_size=3, output_size=3,
                      feedback_initializer="orthogonal")

# tuple
model_b = classic_esn(400, feedback_size=3, output_size=3,
                      feedback_initializer=("orthogonal", {"gain": 0.7}))

# instance
init = get_input_feedback("orthogonal", gain=0.7)
model_c = classic_esn(400, feedback_size=3, output_size=3,
                      feedback_initializer=init)
```

## 4. Head-to-head against `random`

```python
import torch
from resdag import classic_esn
from resdag.training import ESNTrainer
from resdag.utils.data import prepare_esn_data

torch.manual_seed(0)

# Synthetic two-frequency series: a hard test for input encoding.
t = torch.linspace(0, 80 * torch.pi, 4_000)
data = (torch.sin(t) + 0.6 * torch.sin(3.2 * t)).view(1, -1, 1)

warmup, train, target, f_warmup, val = prepare_esn_data(
    data, warmup_steps=500, train_steps=3_000, val_steps=500, normalize=False,
)


def evaluate(feedback_initializer):
    torch.manual_seed(1)
    model = classic_esn(
        reservoir_size=300, feedback_size=1, output_size=1,
        spectral_radius=0.99, leak_rate=0.3, readout_alpha=1e-8,
        feedback_initializer=feedback_initializer,
    )
    ESNTrainer(model).fit(
        warmup_inputs=(warmup,),
        train_inputs=(train,),
        targets={"output": target},
    )
    model.reset_reservoirs()
    pred = model.forecast(f_warmup, horizon=val.shape[1])
    return float(((pred - val) ** 2).mean())


print("random      val MSE:", evaluate("random"))
print("orthogonal  val MSE:", evaluate(("orthogonal", {"gain": 1.0})))
```

The orthogonal initializer typically matches or beats random on
multi-frequency targets because the input map can't accidentally align
all channels with a single reservoir direction.

## What you just learned

1. A registered initializer is any class that subclasses
   `InputFeedbackInitializer` and implements `initialize(weight,
   **kwargs)`.
2. The `@register_input_feedback("name", **defaults)` decorator wires it
   into the resolver used by every factory and `ESNLayer`.
3. Once registered, your initializer is interchangeable with the 11
   built-ins — same string spec, same tuple spec, same instance spec.

The same recipe applies to custom **topologies** (see the
[topologies guide](../guides/topologies.md) and the
[custom topology extending page](../extending/custom-topology.md)).
