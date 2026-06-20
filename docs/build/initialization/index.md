---
description: Every weight matrix in a reservoir is built by a swappable function — the recurrent matrix by a topology, the input and feedback matrices by initializers.
---

<span class="nb-kicker">Build</span>

# Initialization

Every weight matrix in a reservoir is built by a function you can swap: the
recurrent matrix by a *topology*, the input and feedback matrices by
*initializers*. One spec grammar covers both:

| Spec | Example | Resolves to |
| --- | --- | --- |
| `None` | — | Library default: dense `uniform(-1, 1)` |
| `"name"` | `"erdos_renyi"` | Registry entry with its registered defaults |
| `(name, params)` | `("watts_strogatz", {"k": 6})` | Registry entry with overrides |
| callable | `my_fn`, `torch.nn.init.orthogonal_` | Wrapped automatically |
| `(callable, params)` | `(my_fn, {"blocks": 2})` | Wrapper with bound kwargs |
| object | `get_topology("ring_chord")` | Used as-is |

A topology callable is `fn(n, **kw)` returning an `(n, n)` tensor, ndarray,
or NetworkX graph; an initializer callable is `fn(rows, cols, **kw)` returning the
rectangular matrix. In-place mutators — the `torch.nn.init.*_` convention — are detected and handled.

## From the registry

A name uses the registered defaults — `show_topologies()` and
`show_input_initializers()` print what is available — a `(name, params)` tuple
overrides them, and a configured object from `get_topology()` is built once
and reused across layers. The same grammar drives the rectangular matrices:

```python
from resdag import ESNLayer

layer = ESNLayer(200, feedback_size=3, topology="erdos_renyi", spectral_radius=0.9)
layer = ESNLayer(200, feedback_size=3, spectral_radius=0.9,
                 topology=("watts_strogatz", {"k": 6, "p": 0.3, "seed": 42}))
layer = ESNLayer(200, feedback_size=3, input_size=5,
                 feedback_initializer="chebyshev",
                 input_initializer=("random", {"input_scaling": 0.5}))
```

## Reproducibility

A single `seed` on `ESNLayer` fixes the **entire** reservoir — the recurrent
(topology) matrix, the feedback and input weights, and the random bias — so two
layers built with the same seed are identical down to the last entry. It covers
the registry/callable specs *and* the default `uniform(-1, 1)` draws used when
no spec is given, and it is independent of the global RNG state:

```python
a = ESNLayer(200, feedback_size=3, topology="erdos_renyi", seed=42)
b = ESNLayer(200, feedback_size=3, topology="erdos_renyi", seed=42)
assert torch.equal(a.weight_hh, b.weight_hh)
assert torch.equal(a.weight_feedback, b.weight_feedback)
assert torch.equal(a.bias_h, b.bias_h)
```

`seed` accepts a plain `int` or a `torch.Generator` — the latter is convenient
for threading a per-trial generator (e.g. `seed + trial.number`) through an HPO
`model_creator` so every trial draws an identical reservoir run-to-run. An
explicit `seed` inside a tuple/object spec always wins over the layer-level one.
Pass `seed=None` (the default) and the reservoir still tracks
`torch.manual_seed`, because every generator — NumPy for graph topologies, torch
for the rectangular/bias draws — is derived from torch's global RNG.

## Any function is a topology

A bare callable is a valid spec; no registration is required.

<div class="nb-specimen" data-label="block_diagonal.py" markdown>

```python
import torch
from resdag import ESNLayer

def block_diagonal(n: int, blocks: int = 4) -> torch.Tensor:
    """Independent sub-reservoirs along the diagonal."""
    w = torch.zeros(n, n)
    size = n // blocks
    for b in range(blocks):
        s = b * size
        w[s : s + size, s : s + size] = torch.randn(size, size)
    return w

layer = ESNLayer(200, feedback_size=3, topology=block_diagonal, spectral_radius=0.9)
layer = ESNLayer(200, feedback_size=3, spectral_radius=0.9,
                 topology=(block_diagonal, {"blocks": 2}))
layer = ESNLayer(200, feedback_size=3, spectral_radius=1.0,
                 topology=torch.nn.init.orthogonal_)
```

</div>

!!! note "Scale is separate from structure"
    However the matrix is built — graph, registry, bare callable, `torch.nn.init` —
    `spectral_radius` rescales the result *afterwards*. A topology function does not
    need to normalize its own spectrum.

## The `input_scaling` contract

`input_scaling` is the one knob that controls input-injection magnitude — among
the most performance-critical ESN hyperparameters. Every input/feedback
initializer honors the **same** contract, defined once on
`InputFeedbackInitializer`:

- `input_scaling=None` (the default for most initializers) applies **no
  scaling** — the matrix keeps its natural range.
- `input_scaling=s` applies a single, uniform `W <- s * W` as the documented
  *final* transform, so the matrix's magnitude statistic scales **linearly**
  with `s`. Concretely, `input_scaling=0.5` halves it and `input_scaling=2.0`
  doubles it.

The "magnitude statistic" is `max|W|` for the elementwise initializers and the
**per-channel L2 norm** for the two structured ring initializers (whose value
*is* the scaling target). What `input_scaling=0.5` does, per initializer:

| Initializer | Natural range / statistic | Effect of `input_scaling=0.5` |
| --- | --- | --- |
| `random` | entries in `[-1, 1]` | entries in `[-0.5, 0.5]`; `max\|W\|` → `0.5` |
| `random_binary` | entries in `{-1, +1}` | entries in `{-0.5, +0.5}` |
| `chessboard` | entries in `{-1, +1}` | entries in `{-0.5, +0.5}` |
| `chebyshev` | Chebyshev map values | every entry × `0.5` |
| `pseudo_diagonal` | structured `[-1, 1]` | every entry × `0.5` |
| `binary_balanced` | balanced `{-1, +1}` | every entry × `0.5` |
| `dendrocycle_input` | `U[-draw_width, draw_width]` on core | every drawn entry × `0.5` (see below) |
| `opposite_anchors` | per-channel L2 norm = `input_scaling` | per-channel L2 norm = `0.5` |
| `ring_window` | per-channel L2 norm = `input_scaling` | per-channel L2 norm = `0.5` |

!!! warning "`gain` is now `input_scaling`"
    `opposite_anchors` and `ring_window` previously named this knob `gain`.
    `gain` is now a **deprecated alias** for `input_scaling` (identical meaning —
    the per-channel L2 norm); passing it emits a `DeprecationWarning`, and
    passing both raises.

!!! note "`dendrocycle_input`: draw width vs. scaling"
    `dendrocycle_input` historically overloaded `input_scaling` to mean the draw
    half-width of `U[-s, s]`. That role is now the separate **`draw_width`**
    parameter (default `1.0`); `input_scaling` is the uniform final multiply
    shared with every other initializer (default `None`). To reproduce the old
    `dendrocycle_input(input_scaling=s)` draw, pass `draw_width=s`.

```python
# Same magnitude regime regardless of which initializer you swap in:
layer = ESNLayer(200, feedback_size=3, feedback_initializer=("random", {"input_scaling": 0.5}))
layer = ESNLayer(200, feedback_size=3, feedback_initializer=("opposite_anchors", {"input_scaling": 0.5}))
```

An optional `connectivity` knob (a fraction in `(0, 1]`) lives on the same base
class: it keeps that fraction of nonzero entries per input channel. Structured
initializers that already define their own connectivity pattern (only the core
ring receives input, a fixed window per channel) document whether they honor it.

---

## The catalogs

<div class="grid cards" markdown>

- **[Topology catalog](topologies/index.md)**

    ---

    Every registered recurrent-matrix builder, with its connectivity
    portrait and parameters.

- **[Initializer catalog](initializers/index.md)**

    ---

    Every registered input/feedback initializer, with a portrait of the
    matrix it draws and its parameters.

</div>

## Registering your own

Registration makes a builder usable by name in every layer and factory, and
therefore sweepable by name in an HPO study. Matrix builders return the matrix
directly (the built-in `"orthogonal"` Haar-random matrix is one); graph builders
return a NetworkX graph whose edge weights become matrix entries (unweighted edges count as 1):

```python
import networkx as nx
from resdag.init.topology import register_graph_topology, register_matrix_topology

@register_matrix_topology("two_blocks", blocks=2)
def two_blocks(n: int, blocks: int = 2) -> torch.Tensor:
    return block_diagonal(n, blocks=blocks)

@register_graph_topology("double_ring", offset=2)
def double_ring(n: int, offset: int = 2) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_edges_from((i, (i + 1) % n, {"weight": 1.0}) for i in range(n))
    g.add_edges_from((i, (i + offset) % n, {"weight": 0.5}) for i in range(n))
    return g
```

Input/feedback initializers register the same way — a plain
`fn(rows, cols, **kw)` is enough; `register_input_feedback` also accepts
an `InputFeedbackInitializer` subclass when the initializer needs state or validation:

```python
from resdag.init.input_feedback import register_input_feedback

@register_input_feedback("first_neuron", scale=1.0)
def first_neuron(rows: int, cols: int, scale: float = 1.0) -> torch.Tensor:
    w = torch.zeros(rows, cols)
    w[0, :] = scale
    return w
```

## See also

- [**Reservoir dynamics**](../../theory/dynamics.md) — why structure and spectral radius shape what a reservoir remembers.
- [**Tune**](../../workflows/tune.md) — sweeping topologies and initializers by name inside an HPO study.
