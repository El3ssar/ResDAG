---
description: Parameterless glue for any DAG — concatenation, augmentation powers, deterministic dropout, and feature partitioning, plus the augmentation pattern composed end to end.
---

<span class="nb-kicker">Build · Layers</span>

# Transforms

Transforms are the parameterless feature operations that wire reservoirs,
readouts, and everything between them into a DAG. They act only on the
feature axis of `(batch, time, features)`, so they compose with any
reservoir family's outputs — and with each other — without knowing what
produced the tensor.

| Layer | Signature | What it does |
| --- | --- | --- |
| `Concatenate` | `Concatenate()` | Joins inputs along the feature dimension |
| `Power` | `(exponent)` | Raises every feature to `exponent`; the `power_augmented` augmentation |
| `SelectiveExponentiation` | `(index, exponent)` | Raises even- or odd-indexed features (parity of `index`) to `exponent`; the Ott augmentation |
| `SelectiveDropout` | `(mask)` | Zeros a fixed boolean mask of features — deterministic, for ablations |
| `FeaturePartitioner` | `(partitions, overlap)` | Splits features into overlapping circular slices, returns a list — feeds parallel reservoirs |

All five import from `resdag` directly. Two fine points: `Concatenate`
takes any number of inputs that agree on every dimension but the last,
and `FeaturePartitioner` requires the feature count to divide evenly by
`partitions`, returning slices of width `features // partitions +
2 * overlap` with circular wrapping at the boundaries.

---

## The augmentation pattern

The most common composition: enrich a reservoir's states with a nonlinear
copy, then let the readout see both the raw input and the augmented
features. This is exactly what the `ott_esn` factory wires — squaring the
even-indexed states — but the pattern accepts any reservoir in the middle
slot:

<div class="nb-specimen" data-label="augmentation.py" markdown>

```python
from resdag import (
    CGReadoutLayer, Concatenate, ESNLayer, ESNModel,
    SelectiveExponentiation, reservoir_input,
)

inp = reservoir_input(3)
states = ESNLayer(500, feedback_size=3)(inp)    # any reservoir slots in here
augmented = SelectiveExponentiation(index=0, exponent=2.0)(states)
features = Concatenate()(inp, augmented)        # readout sees input + states
out = CGReadoutLayer(3 + 500, 3, name="output")(features)
model = ESNModel(inp, out)
```

</div>

Swap `SelectiveExponentiation` for `Power(2.0)` and you have
`power_augmented`; drop the `Concatenate` and the readout sees only the
augmented states. The arithmetic to keep straight is the readout's
`in_features` — concatenation adds feature dimensions, and the readout
must be told the sum.

Nothing here is a closed set. Any `nn.Module` that maps
`(batch, time, features)` to the same layout composes identically; these
five are simply the ones the premade factories needed first.

## See also

- [Architectures](../architectures/index.md) — full pattern gallery, factories included
- [The Build hub](../index.md) — the composition thesis in one page
- [Layers reference](../../reference/layers.md) — full signatures
