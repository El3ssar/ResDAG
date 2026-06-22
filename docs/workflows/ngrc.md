---
description: An end-to-end next-generation reservoir computing workflow — delay embedding, the degree convention, fitting the readout, forecasting, and when to reach for NG-RC over an echo-state reservoir.
---

<span class="nb-kicker">Work · NG-RC</span>

# NG-RC

Next-generation reservoir computing (NG-RC, Gauthier et al. 2021) replaces
the random recurrent reservoir with a **deterministic** feature map: time
delays and the polynomial products of those delays. There are no recurrent
weights, no spectral radius, no leak rate, and no seed — two runs with the
same `k`, `s`, `p` produce identical features. It slots into the same DAG
position as any other reservoir, so the rest of the pipeline — algebraic
readout, autoregressive forecast — is unchanged from an echo-state model.

This page is the end-to-end workflow. For the layer itself — every
parameter, the feature-count formula, the cumulative-degree convention —
see [Build · NGReservoir](../build/layers/ng-reservoir.md).

## The feature map

`NGReservoir` builds three blocks and concatenates them:

1. a constant `1.0` feature (`include_constant`),
2. the **delay embedding** — the current input stacked with `k-1` earlier
   taps spaced `s` steps apart (`include_linear`), giving
   $D = \text{input\_dim} \cdot k$ linear features,
3. the **monomials** — by default every product of *exactly* degree `p`
   formed from those $D$ linear features.

The output dimension is

$$
\text{feature\_dim} = \mathbb{1}[\text{constant}] + \mathbb{1}[\text{linear}]\,D + \binom{D+p-1}{p}.
$$

<div class="nb-specimen" data-label="ngrc_features.py" markdown>

```python
import torch
from resdag import NGReservoir

layer = NGReservoir(input_dim=3, k=2, s=1, p=2)  # delay taps k, spacing s, degree p
x = torch.randn(1, 500, 3)                       # (batch, time, input_dim)
features = layer(x)                              # (1, 500, feature_dim)

print(layer.feature_dim)    # 28 = 1 (constant) + 6 (linear, D=6) + 21 (degree-2 monomials)
print(layer.warmup_length)  # (k-1)*s = 1 — steps before this contain buffer zeros
```

</div>

!!! warning "Exact degree, not degree-up-to-`p` (default)"
    By default the nonlinear block holds monomials of **exactly** degree `p`
    — the lower-order cross terms (degrees `2, …, p-1`) are *excluded*, which
    matches the single-degree bases in Gauthier et al. (2021). If you are
    porting a config that expects the full "degree up to `p`" basis common in
    other NVAR implementations, pass `cumulative=True`. The
    [layer page](../build/layers/ng-reservoir.md#cumulative-degree-cumulativetrue)
    has the cumulative feature-count formula.

## End to end

NG-RC is wired exactly like an echo-state model: a `reservoir_input`, the
feature map, a `CGReadoutLayer`, then `ESNModel`. The only NG-RC-specific
step is reading `feature_dim` off the layer to size the readout, since the
feature count depends on `k`, `s`, `p` rather than a `reservoir_size` you
chose directly. Build the layer once so you can both query its dimension and
wire it into the graph.

<div class="nb-specimen" data-label="ngrc_forecast.py" markdown>

```python
import torch
import resdag as rd
from resdag import ESNModel, NGReservoir, reservoir_input, lorenz
from resdag.layers import CGReadoutLayer

data = lorenz(5000, seed=0)                      # (1, 5000, 3) — chaotic benchmark
warmup, train, target, f_warmup, val = rd.utils.prepare_esn_data(
    data, warmup_steps=20, train_steps=3000, val_steps=1000, normalize=True
)

ngrc = NGReservoir(input_dim=3, k=2, s=1, p=2)   # build once: query + wire the same layer

inp = reservoir_input(3)
features = ngrc(inp)
out = CGReadoutLayer(ngrc.feature_dim, 3, name="output", alpha=1e-6)(features)
model = ESNModel(inp, out)

rd.ESNTrainer(model).fit(
    warmup_inputs=(warmup,),                     # short: warmup only fills the (k-1)*s buffer
    train_inputs=(train,),
    targets={"output": target},                  # keyed by the readout name
)

forecast = model.forecast(f_warmup, horizon=200)  # (1, 200, 3) autoregressive rollout
```

</div>

Everything after the feature map is the standard workflow:
[`ESNTrainer.fit`](train.md#path-1-the-algebraic-solve) solves the readout by
ridge regression in one pass, and [`model.forecast`](forecast.md) runs the
two-phase autoregressive rollout. `alpha` on the readout is the only fitting
hyperparameter; sweep it on a log scale exactly as for an echo-state model.

!!! note "Warmup is short on purpose"
    NG-RC's only state is the FIFO delay buffer, which fills in `(k-1)*s`
    steps — `1` step for `k=2, s=1`. The `warmup_steps=20` above is generous;
    NG-RC needs nothing like the hundreds of steps an echo-state reservoir
    spends washing out its random initial condition. The first
    `layer.warmup_length` outputs still contain buffer zeros, so keep the
    warmup at least that long and discard those steps from a loss if you score
    one directly.

## When to prefer NG-RC over an ESN

NG-RC and the echo-state reservoir occupy the same slot, so the choice is a
modeling decision rather than an API one:

| Reach for NG-RC when… | Reach for an ESN when… |
| --- | --- |
| The dataset is **short** — the `(k-1)*s`-step warmup wastes almost no data. | You have plenty of data and warmup length is not a constraint. |
| You need **exact reproducibility** — the map is deterministic, no seed. | A random projection into a large state is acceptable or desirable. |
| The system is **low-dimensional** with smooth polynomial structure. | The signal is **high-dimensional** — monomials explode combinatorially. |
| You want **few, interpretable knobs** — three small integers, grid-searchable. | You are comfortable tuning spectral radius, leak rate, and topology. |

The trade-off is the binomial term in `feature_dim`: it grows fast in `k`,
`p`, and `input_dim`, and `NGReservoir` warns once `feature_dim` exceeds
10,000. For high-dimensional signals, an echo-state reservoir that compresses
into a fixed-size state is usually the better fit. Because NG-RC is a plain
feature map with no learnable parameters, it also drops into the
[frozen-features training path](train.md#path-2-frozen-reservoir-gradient-head)
unchanged — gradients flow through it to any upstream module.

## Next

- [**Build · NGReservoir**](../build/layers/ng-reservoir.md) — every parameter and the cumulative-degree basis
- [Train](train.md) — the algebraic solve and the gradient-head path
- [Forecast](forecast.md) — the two-phase autoregressive rollout
