# Add a premade model

Factory functions return `ESNModel` built with `pytorch_symbolic`.

```python
from typing import Any
from resdag.core import ESNModel, reservoir_input
from resdag.layers import CGReadoutLayer, Concatenate, ESNLayer, Power


def power_esn(
    reservoir_size: int,
    feedback_size: int,
    output_size: int,
    exponent: float = 2.0,
    spectral_radius: float = 0.9,
    **reservoir_kwargs: Any,
) -> ESNModel:
    inp = reservoir_input(feedback_size)
    res = ESNLayer(
        reservoir_size,
        feedback_size=feedback_size,
        spectral_radius=spectral_radius,
        **reservoir_kwargs,
    )(inp)
    aug = Power(exponent)(res)
    features = Concatenate()(inp, aug)
    out = CGReadoutLayer(features.shape[-1], output_size, name="output")(features)
    return ESNModel(inp, out)
```

Export from `resdag.models.__init__` and add to `resdag.__all__` if it is public API.

Mirror [`classic_esn.py`](https://github.com/El3ssar/resdag/blob/main/src/resdag/models/classic_esn.py).

## See also

- [Premade models reference](../reference/models.md)
