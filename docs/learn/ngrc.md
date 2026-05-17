# Next-Generation Reservoir Computing

!!! info "Why this exists"
    Classical ESNs rely on **random recurrent weights**. NG-RC (Gauthier et al.,
    arXiv:2106.07688) shows that strong performance is possible with **no recurrence**:
    features are built from **time-delayed inputs** and **polynomial monomials**, then
    a linear readout is fit — same training story as an ESN, different feature map.

## Feature construction

Let $x_t \in \mathbb{R}^d$ be the input at time $t$.

1. **Linear delay embedding** $O_{\mathrm{lin}}$: stack $k$ taps spaced by $s$ steps  
   $[x_t, x_{t-s}, \ldots, x_{t-(k-1)s}]$ → dimension $D = d \cdot k$.

2. **Nonlinear monomials** $O_{\mathrm{nonlin}}$: all degree-$p$ monomials over the $D$
   linear features — count $\binom{D+p-1}{p}$.

3. **Total** $O_{\mathrm{total}} = [\text{optional } 1] \;||\; O_{\mathrm{lin}} \;||\; O_{\mathrm{nonlin}}$.

There is **no** $W_{\mathrm{res}}$. State is a FIFO **delay buffer** of past inputs.

## ResDAG mapping

| Concept | Class |
|---------|-------|
| Single step | `NGCell` |
| Sequence API | `NGReservoir` |

```python
from resdag.layers import NGReservoir

layer = NGReservoir(input_dim=3, k=2, s=1, p=2)
x = torch.randn(4, 100, 3)
features = layer(x)  # (4, 100, feature_dim)
```

### Warmup / transient

The buffer needs $(k-1)\cdot s$ steps to fill. Early timesteps still run but embed
zeros for missing history — discard the first `warmup_length` outputs when evaluating
accuracy.

### Combinatorial caution

`feature_dim` grows as $\binom{D+p-1}{p}$. `NGCell` warns when `feature_dim > 10_000`.
Reduce `k`, `p`, or `input_dim` before pushing GPU memory.

## When to prefer NG-RC vs ESN

| Choose NG-RC | Choose ESN |
|--------------|------------|
| Moderate $d$, short memory depth | Long fading memory, large reservoirs |
| Want deterministic, weight-free features | Want graph topologies + spectral radius tuning |
| Lorenz / low-dimensional chaos benchmarks | Input-driven chaos with rich recurrence |

See [NG-RC vs ESN guide](../guides/ngrc-vs-esn.md) for a side-by-side workflow.

## See also

- [`NGCell` / `NGReservoir`](../reference/layers/reservoirs.md)
- [Reservoir layers](reservoir-layers.md)
- [Readouts](readouts.md) — same ridge training
