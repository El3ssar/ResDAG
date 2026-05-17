# Visualizing architectures

Requires [Graphviz](https://graphviz.org/) installed on your system for `plot_model`.

```python
from resdag import ott_esn

model = ott_esn(300, feedback_size=3, output_size=3)

model.summary()       # text table in the terminal
model.plot_model()    # writes a graph image (format depends on pytorch_symbolic)
```

For custom graphs built with `pytorch_symbolic`:

```python
import pytorch_symbolic as ps
from resdag.core import ESNModel
from resdag.layers import ESNLayer, Concatenate
from resdag.layers.readouts import CGReadoutLayer
from resdag.layers.transforms import SelectiveExponentiation

inp = ps.Input((50, 3))
res = ESNLayer(200, feedback_size=3)(inp)
aug = SelectiveExponentiation(index=2, exponent=2.0)(res)
cat = Concatenate()(inp, aug)
out = CGReadoutLayer(cat.shape[-1], 3)(cat)
model = ESNModel(inp, out)

model.summary()
model.plot_model()
```

## See also

- [Example 04](https://github.com/El3ssar/resdag/blob/main/examples/04_model_visualization.py)
- [Premade `ott_esn`](../reference/models.md)
