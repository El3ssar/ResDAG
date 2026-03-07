# Custom Layers

resdag provides several utility layers in `resdag.layers.custom` that are used internally in premade models and available for your own architectures.

---

## Concatenate

Concatenates multiple tensors along the **feature dimension** (last axis). Parameterless.

```python
from resdag.layers import Concatenate

cat = Concatenate()

# In a symbolic model
inp      = ps.Input((100, 3))
res      = ESNLayer(500, feedback_size=3)(inp)
combined = Concatenate()(inp, res)   # (batch, 100, 3+500)
```

**Shape**: Takes `N` inputs of `(batch, time, feat_i)` and returns `(batch, time, sum(feat_i))`.

Used in `ott_esn` and `power_augmented` to concatenate the original input with augmented reservoir states.

---

## SelectiveExponentiation

Exponentiates either **even** or **odd** indexed features to a given power, leaving the other indices unchanged.

```python
from resdag.layers import SelectiveExponentiation

layer = SelectiveExponentiation(
    index=0,       # 0 = even indices, 1 = odd indices
    exponent=2.0,  # power to raise selected features to
)

# Example
x = torch.tensor([[[1., 2., 3., 4., 5.]]])
out = layer(x)
# Even indices (0,2,4): squared → [1, -, 9, -, 25]
# Odd indices (1,3):   unchanged → [-, 2, -, 4, -]
# out: [[[1., 2., 9., 4., 25.]]]
```

**Used in**: `ott_esn` — squares even-indexed reservoir neurons for state augmentation.

---

## Power

Raises **all** features to a given power.

```python
from resdag.layers.custom import Power

layer = Power(exponent=2.0)

x = torch.tensor([[[1., -2., 3.]]])
out = layer(x)
# out: [[[1., 4., 9.]]]
```

**Used in**: `power_augmented` — raises all reservoir states to the specified exponent.

---

## SelectiveDropout

Per-feature dropout with selectivity control. Applies dropout only to a subset of features (based on index or mask).

```python
from resdag.layers.custom import SelectiveDropout

layer = SelectiveDropout(p=0.5, selectivity=0.3)
# Randomly drops features during training; eval mode disables dropout
```

---

## FeaturePartitioner

Splits the feature dimension into overlapping groups and processes them separately.

```python
from resdag.layers.custom import FeaturePartitioner

layer = FeaturePartitioner(n_groups=4, overlap=0.1)
# Partitions input features into 4 overlapping groups
```

**Use case**: Hierarchical processing; connecting different feature groups to separate reservoir sub-populations.

---

## OutliersFilteredMean

Computes a mean across the batch or time dimension, filtering out outlier values beyond a threshold.

```python
from resdag.layers import OutliersFilteredMean

layer = OutliersFilteredMean(threshold=2.0, dim=0)
# Computes mean along dim=0, excluding values > threshold standard deviations from mean
```

**Use case**: Robust aggregation in noisy settings.

---

## Building Custom Architectures with These Layers

Here is the `ott_esn` architecture built manually with these layers:

```python
import pytorch_symbolic as ps
from resdag import ESNModel, ESNLayer, CGReadoutLayer
from resdag.layers import Concatenate, SelectiveExponentiation

feedback_size = 3
reservoir_size = 500
output_size = 3

inp = ps.Input((100, feedback_size))

# Reservoir
reservoir = ESNLayer(
    reservoir_size=reservoir_size,
    feedback_size=feedback_size,
    input_size=0,              # no separate driving input
    spectral_radius=0.9,
)(inp)

# Augment: square even-indexed neurons
augmented = SelectiveExponentiation(index=0, exponent=2.0)(reservoir)

# Concatenate original input with augmented reservoir states
concat = Concatenate()(inp, augmented)  # (batch, time, 3+500)

# Readout
readout = CGReadoutLayer(
    in_features=feedback_size + reservoir_size,
    out_features=output_size,
    alpha=1e-6,
    name="output",
)(concat)

model = ESNModel(inp, readout)
```

---

## Creating Your Own Layers

Any `torch.nn.Module` can be used in a symbolic model:

```python
import torch.nn as nn
import pytorch_symbolic as ps

class MyTransform(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, features)
        return torch.abs(x) + x ** 3

# Use in symbolic model
inp       = ps.Input((100, 3))
res       = ESNLayer(500, feedback_size=3)(inp)
custom    = MyTransform()(res)
readout   = CGReadoutLayer(500, 3, name="output")(custom)
model     = ESNModel(inp, readout)
```

The `pytorch_symbolic` framework handles the computational graph automatically as long as your module:

1. Extends `torch.nn.Module`
2. Implements `forward()` with tensor inputs and output
3. Does not change the batch or time dimensions unexpectedly

---

## All Custom Layer Imports

```python
# Public API
from resdag.layers import (
    Concatenate,
    SelectiveExponentiation,
    OutliersFilteredMean,
)

# Additional custom layers (direct import)
from resdag.layers.custom import (
    Power,
    SelectiveDropout,
    FeaturePartitioner,
)
```
