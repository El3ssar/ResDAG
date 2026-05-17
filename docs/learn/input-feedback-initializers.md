# Input & feedback initializers

!!! info "Why this exists"
    $W_{\mathrm{res}}$ sets recurrent dynamics; **input and feedback matrices**
    ($W_{\mathrm{in}}$, $W_{\mathrm{fb}}$) control how observations enter the reservoir.
    ResDAG treats them like topologies: named initializers, registry, and
    `get_input_feedback()`.

## Shapes

For `ESNLayer(reservoir_size=N, feedback_size=F, input_size=I)`:

| Matrix | Shape |
|--------|-------|
| Feedback weights | `(N, F)` |
| Input weights | `(N, I)` |

Specify independently:

```python
ESNLayer(
    500, feedback_size=3, input_size=5,
    feedback_initializer="chebyshev",
    input_initializer=("random", {"input_scaling": 0.5}),
)
```

## Registered initializers (11)

| Name | Idea |
|------|------|
| `random` | Uniform in $[-1, 1]$ (scaled) |
| `random_binary` | Random $\pm 1$ |
| `zeros` | All zeros |
| `pseudo_diagonal` | Block-structured sparse pattern; good baseline structure |
| `chebyshev` | Deterministic chaotic map (Chebyshev) on weights — reproducible "structured chaos" |
| `chessboard` | Alternating $\pm 1$ checkerboard |
| `binary_balanced` | Hadamard-style balanced columns |
| `opposite_anchors` | Opposite anchor points on a ring |
| `ring_window` | Localized window on ring topology |
| `chain_of_neurons_input` | Chain-specific sparse pattern |
| `dendrocycle_input` | Matches dendrocycle graph layouts |

List and inspect defaults:

```python
from resdag.init.input_feedback import show_input_initializers
print(show_input_initializers("chebyshev"))
```

## Highlights

### Chebyshev

Uses a Chebyshev map to fill the weight matrix deterministically (parameters `p`, `q`, `k`).
Useful when you want **repeatable** nonlinear mixing without storing a huge random seed
file — common in chaotic-system experiments.

### Pseudo-diagonal

Builds a near-block-diagonal pattern with optional binarization. Often pairs well with
structured topologies when you want each input channel to hit a subset of neurons.

## See also

- [Graph topologies](topologies.md)
- [Reference: input/feedback](../reference/init/input-feedback.md)
- [Extend: custom initializer](../extending/custom-initializer.md)
