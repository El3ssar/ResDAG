---
description: NG-RC reservoir layer with deterministic dynamics built from delay taps and polynomial monomials. Covers the k, s, p parameters, the feature-count formula, and warmup requirements.
---

<span class="nb-kicker">Build · Layers</span>

# NGReservoir

`NGReservoir` implements the next-generation reservoir computing family
(Gauthier et al. 2021): a reservoir with designed rather than random
dynamics. It has no recurrent weights and no randomness. Features are
built from time-delayed inputs and their polynomial products, so two
constructions with the same `k`, `s`, `p` produce identical features.

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

## What k, s, p control

`k` taps spaced `s` steps apart give the layer a memory window of
`(k-1)*s + 1` timesteps; `p` sets the degree of the monomials formed from
everything in that window. With $D = \text{input\_dim} \cdot k$ linear
features, the output dimension is

$$
\text{feature\_dim} = \mathbb{1}[\text{constant}] + \mathbb{1}[\text{linear}]\,D + \binom{D+p-1}{p}
$$

The three terms are the constant 1, the delay-embedded inputs, and every
monomial of *exactly* degree `p`; `include_constant` and `include_linear`
toggle the first two.

!!! warning "Exact degree, not degree-up-to-`p` (default)"
    By default the nonlinear block holds monomials of **exactly** degree `p` —
    lower-order cross terms (degrees `2, …, p-1`) are *excluded*. So
    `NGReservoir(p=3, include_linear=True)` emits `constant + linear + cubic`
    with **no quadratic terms**. This matches Gauthier et al. (2021), whose
    Lorenz63 and double-scroll bases each use a single polynomial degree. Many
    NVAR / NG-RC implementations instead use the *cumulative* "degree up to `p`"
    basis — see below.

## Cumulative degree (`cumulative=True`)

Pass `cumulative=True` to emit **every** monomial of degree up to `p` (the full
polynomial basis), rather than only the top degree:

```python
layer = NGReservoir(input_dim=3, k=2, p=3, cumulative=True)
```

With `D = input_dim · k` linear features, the nonlinear block then spans degrees
`2 … p` when `include_linear=True` (degree-1 monomials are dropped — they would
duplicate the linear block), or `1 … p` when `include_linear=False`:

$$
\text{feature\_dim} = \mathbb{1}[\text{constant}] + \mathbb{1}[\text{linear}]\,D + \sum_{g=d_{\text{lo}}}^{p} \binom{D+g-1}{g},
\quad d_{\text{lo}} = \begin{cases} 2 & \text{include\_linear} \\ 1 & \text{otherwise} \end{cases}
$$

The default (`cumulative=False`) is unchanged and stays bit-for-bit identical to
previous releases; reach for the cumulative basis when porting configs that
expect "degree up to `p`".

!!! note "Degree 1 (`p=1`)"
    With `p=1` the degree-1 monomials are the linear features themselves.
    To avoid two identical blocks (which would make the readout design
    matrix rank-deficient), the monomial block is dropped when `p=1` and
    `include_linear=True`. With `p=1` and `include_linear=False` the
    degree-1 monomials are kept, since they are then the only
    delay-embedded features.

!!! warning "Combinatorial growth"
    The binomial term grows fast in `k`, `p`, and `input_dim`; the layer
    warns when `feature_dim` exceeds 10,000. If the warning fires,
    reduce `k`, `p`, or the input dimension before training.

## State and warmup

The state is not a hidden vector but a FIFO delay buffer of shape
`(batch, (k-1)*s, input_dim)`. The standard state API applies, with
`set_state` validating that 3-D layout. The buffer needs `(k-1)*s` steps
to fill, and outputs before that contain zeros from empty slots: discard
the first `warmup_length` steps when accuracy matters. The full warmup is
those `(k-1)*s` steps, compared to the hundreds an echo-state reservoir
typically needs to wash out its initial condition.

## When to prefer it

This family suits short datasets, where the `(k-1)*s`-step warmup
consumes little data; systems that are low-dimensional with smooth
polynomial structure; and settings that require exact reproducibility,
since the construction is deterministic and needs no seed. The
hyperparameters are three small integers that can be grid-searched
exhaustively, with no spectral radius, leak rate, or topology to tune.
The trade-off is the formula above: `feature_dim` grows combinatorially
with `input_dim`, so high-dimensional signals favor reservoirs that
compress into a fixed-size state. The layer has no learnable parameters,
and gradients flow through it to upstream modules.

## See also

- [Architectures](../architectures/index.md) — slotting a reservoir into a DAG
- [Reservoir design](../../theory/design.md) — choosing between families
- [Layers reference](../../reference/layers.md) — full signatures
