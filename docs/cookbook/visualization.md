---
description: See your model — text summary via pytorch_symbolic, graphviz DAG via plot_model, with trainability padlocks.
---

<span class="rd-eyebrow">Cookbook</span>

# Visualizing architectures

By the end of this page you'll have your model's DAG rendered as an SVG,
with tensor shapes on the edges and a padlock on every layer telling you
what backprop can touch.

<div class="rd-window" data-title="plot.py" markdown>

```python
from resdag.models import ott_esn

model = ott_esn(reservoir_size=300, feedback_size=3, output_size=3)

model.summary()                      # text table from pytorch_symbolic:
                                     # every node, shapes, parameter counts

model.plot_model(
    show_shapes=True,                # tensor shapes on nodes and edges
    show_trainable=True,             # 🔒 frozen / 🔓 trainable per layer
    rankdir="LR",                    # "TB" (default), "LR", "BT", "RL"
    save_path="ott_esn.svg",         # write to disk instead of displaying
    format="svg",                    # "svg", "png", "pdf"
)
```

</div>

<figure markdown>
![Ott ESN architecture](../assets/figures/arch_ott_esn.svg)
<figcaption>Literal <code>plot_model()</code> output for
<code>ott_esn</code>: reservoir, selective squaring, skip connection from
the input, readout.</figcaption>
</figure>

Without `save_path`, the behavior adapts: Jupyter renders the SVG inline;
a script or REPL opens your system viewer and returns a
`graphviz.Source`.

**The padlocks** (with `show_trainable=True`): 🔒 means every parameter
in that layer has `requires_grad=False` — the default for reservoirs and
readouts, since readouts are fitted algebraically, not by gradient
descent. 🔓 means the layer participates in backprop. Check this before
wiring a model into an [SGD pipeline](pipelines.md); a surprise 🔒 on the
layer you meant to train explains an empty optimizer faster than any
stack trace.

!!! note "Requires graphviz twice"
    `plot_model` needs both the Python package (`pip install graphviz`)
    and the system binary (`apt install graphviz` or equivalent). Missing
    the package — or missing the binary when saving or in a notebook — it
    falls back to printing the raw DOT source; paste it into an online
    Graphviz renderer and you still get your diagram.

## Related

- [Building models](../learn/building-models.md) — the symbolic API these diagrams depict.
- [PyTorch pipelines](pipelines.md) — where the trainability padlocks earn their keep.
- [Multi-readout models](multi-readout.md) — architectures worth plotting before training.
