---
description: Premade architectures in resdag.models — each is a factory function that returns a ready-to-train model, documented with its wiring diagram.
---

<span class="nb-kicker">Build</span>

# Architectures

The factories in `resdag.models` build common architectures. Each one is
a short function that wires [layers](../layers/index.md) into a model the
same way you would by hand, then returns it ready to train. The result is
an ordinary `ESNModel`: you can retrain it on your data, swap its
topology, embed it in a larger network, or read its source and use it as
a template. The current factories are ESN-family, but the composition
system is not limited to them; new architectures are written as ordinary
factory functions.

When no premade fits, write your own. A custom factory is a function that
wires a graph and returns a model, and it works in every workflow
(training, forecasting, tuning) the same way the premades do:

<div class="nb-specimen" data-label="my_factory.py" markdown>

```python
import resdag as rd

def my_esn(n: int, dim: int) -> rd.ESNModel:
    inp = rd.reservoir_input(dim)
    states = rd.ESNLayer(n, feedback_size=dim)(inp)
    return rd.ESNModel(inp, rd.CGReadoutLayer(n, dim, name="output")(states))
```

</div>

<!-- nb-cards: build/architectures -->

Each architecture page shows the wiring diagram of the model its factory
builds, along with the factory's parameters and a usage example.

## See also

- [Layers](../layers/index.md) — the components the factories are wired from.
- [Train](../../workflows/train.md) — fitting these models with `ESNTrainer.fit`.
- [Models reference](../../reference/models.md) — full factory signatures.
