---
description: The next-generation family's layer — designed dynamics from delay taps and polynomial monomials; k, s, p semantics, the feature-count formula, and the warmup the buffer needs.
---

<span class="nb-kicker">Build · Layers</span>

# NGReservoir

The next-generation family's layer — a reservoir with designed, not
random, dynamics (Gauthier et al. 2021). There are no recurrent weights
and no randomness: features are built from time-delayed inputs and their
polynomial products, so two constructions with the same `k`, `s`, `p` are
bit-identical.

<figure markdown>
![NG-RC architecture](../../assets/figures/arch_ngrc.svg)
<figcaption>NG-RC occupies the same DAG position as any other
reservoir.</figcaption>
</figure>

<div class="nb-specimen" data-label="ng_reservoir.py" markdown>

```python
from resdag import NGReservoir

layer = NGReservoir(
    input_dim=3,            # dimensionality of each input vector
    k=2,                    # delay taps, including the current input
    s=1,                    # spacing between taps, in timesteps
    p=2,                    # polynomial degree of the monomials
    include_constant=True,  # prepend a constant 1.0 feature
    include_linear=True,    # keep the delay-embedded inputs themselves
)

features = layer(x)         # x: (batch, time, 3) -> (batch, time, 28)
layer.feature_dim           # 28
layer.warmup_length         # (k-1)*s = 1
```

</div>

## What k, s, p buy

`k` taps spaced `s` steps apart give the layer a memory window of
`(k-1)*s + 1` timesteps; `p` sets the degree of the monomials formed from
everything in that window. With $D = \text{input\_dim} \cdot k$ linear
features, the output dimension is

$$
\text{feature\_dim} = \mathbb{1}[\text{constant}] + \mathbb{1}[\text{linear}]\,D + \binom{D+p-1}{p}
$$

— a constant 1, the delay-embedded inputs, and every degree-`p` monomial
(`include_constant` and `include_linear` toggle the first two terms).

!!! warning "Combinatorial budget"
    The binomial term grows fast in `k`, `p`, and `input_dim`; the layer
    warns when `feature_dim` exceeds 10,000. Treat that warning as a
    design review, not a formality.

## State and warmup

The state is not a hidden vector but a FIFO delay buffer of shape
`(batch, (k-1)*s, input_dim)` — the standard state API applies, with
`set_state` validating that 3-D layout. The buffer needs `(k-1)*s` steps
to fill, and outputs before that contain zeros from empty slots: discard
the first `warmup_length` steps when accuracy matters. That is the whole
warmup — a handful of steps, versus the hundreds an echo-state reservoir
needs to wash out its initial condition.

## When to prefer it

Reach for this family when the data is short (the tiny warmup is a real
budget win), when the underlying system is low-dimensional with smooth
polynomial structure, or when you need exact reproducibility with no seed
in sight. The hyperparameters are three small integers you can grid
exhaustively — no spectral radius, no leak rate, no topology. The cost is
the formula above: `feature_dim` explodes with `input_dim`, so
high-dimensional signals favor reservoirs that compress into a fixed
state instead. The layer has no learnable parameters, and gradients flow
through it to upstream modules when you need them to.

## See also

- [Architectures](../architectures/index.md) — slotting a reservoir into a DAG
- [Reservoir design](../../theory/design.md) — choosing between families
- [Layers reference](../../reference/layers.md) — full signatures
