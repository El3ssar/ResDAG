---
description: Next-Generation Reservoir Computing — delay embedding plus polynomial features, zero random weights, trained and forecast like any other ResDAG model.
---

<span class="rd-eyebrow">Cookbook</span>

# Next-Generation Reservoir Computing

By the end of this page you'll have a forecasting model with no random
matrices, no recurrence, and no spectral radius to tune — and you'll know
exactly how wide its feature vector is before you build it.

NG-RC (Gauthier et al. 2021, arXiv:2106.07688) replaces the random
reservoir with a deterministic feature map: a time-delay embedding of the
input plus every polynomial monomial you can form from it. With $d$ input
features, $k$ delay taps spaced $s$ steps apart, and degree $p$:

$$
D = d \cdot k,
\qquad
\text{feature\_dim} = \underbrace{1}_{\text{constant}}
 + \underbrace{D}_{\text{linear}}
 + \underbrace{\binom{D+p-1}{p}}_{\text{monomials}}
$$

## NG-RC + readout in a DAG

`NGReservoir` slots into the symbolic API exactly where an `ESNLayer`
would, so training and forecasting are unchanged.

<div class="rd-window" data-title="ngrc_forecast.py" markdown>

```python
import torch
from resdag import ESNModel, ESNTrainer, CGReadoutLayer, NGReservoir, reservoir_input
from resdag.utils.data import prepare_esn_data

t = torch.linspace(0, 60, 3000)
data = torch.stack([torch.sin(t), torch.cos(t), torch.sin(2 * t)], dim=-1).unsqueeze(0)
warmup, train, target, f_warmup, val = prepare_esn_data(
    data, warmup_steps=100, train_steps=2000, val_steps=300
)

inp = reservoir_input(3)
ngrc = NGReservoir(input_dim=3, k=2, s=1, p=2)
feats = ngrc(inp)                                  # (batch, time, 28)

# include_constant=True already gives the readout an intercept column,
# so bias=False is the principled choice (solves uncentered ridge).
readout = CGReadoutLayer(ngrc.feature_dim, 3, name="output", bias=False)(feats)
model = ESNModel(inp, readout)

# Same workflow as an ESN: one teacher-forced pass, one algebraic solve.
ESNTrainer(model).fit(
    warmup_inputs=(warmup,),
    train_inputs=(train,),
    targets={"output": target},
)
prediction = model.forecast(f_warmup, horizon=300)   # compare against val
```

</div>

For `input_dim=3, k=2, p=2`: $D = 6$, monomials $\binom{7}{2} = 21$, so
`ngrc.feature_dim == 1 + 6 + 21 == 28`. Do this arithmetic before wiring
the readout — `in_features` must match.

<figure markdown>
![NG-RC architecture](../assets/figures/arch_ngrc.svg)
<figcaption>The whole model: one feature-construction layer straight into
a readout. No reservoir matrix, no Echo State Property.</figcaption>
</figure>

## The delay buffer needs to fill

The only state is a FIFO buffer holding the last `(k-1)*s` inputs. Until
it fills, the delay taps read zeros, so the first `(k-1)*s` outputs are
built from padding — discard them:

```python
layer = NGReservoir(input_dim=3, k=4, s=5, p=2)
x = torch.randn(8, 200, 3)                  # (batch, time, features)
features = layer(x)                         # (8, 200, 91)
clean = features[:, layer.warmup_length:, :]  # drop the first (4-1)*5 = 15 steps
```

In the full-model workflow above this is a non-issue: the trainer's
warmup pass fills the buffer before any fitting happens.

!!! warning "Combinatorial explosion"
    The monomial count $\binom{D+p-1}{p}$ grows fast. `input_dim=10, k=5,
    p=3` gives $D=50$ and 22,100 monomials. `NGReservoir` warns when
    `feature_dim` exceeds 10,000 — treat that warning as a design review,
    not noise. Reduce `k`, `p`, or the input dimension.

## Knobs

| Parameter | Default | Effect |
|---|---|---|
| `k` | 2 | Delay taps, including the current input |
| `s` | 1 | Spacing between taps, in timesteps |
| `p` | 2 | Polynomial degree of the monomials |
| `include_constant` | `True` | Prepend a 1.0 feature (the intercept) |
| `include_linear` | `True` | Include the raw delay embedding in the output |

The layer has zero learnable parameters — all the capacity lives in the
readout, which is why NG-RC trains in milliseconds. Gradients still flow
through it, so it composes with the [pipeline patterns](pipelines.md) too.

## Related

- [Custom components](custom-components.md) — `NGCell` is a `ReservoirCell`; build your own the same way.
- [Forecasting](../learn/forecasting.md) — the two-phase forecast loop NG-RC plugs into.
- [Readout fitting](../under-the-hood/readout-fitting.md) — why `bias=False` changes the ridge problem.
