---
description: Structure is a function — every weight matrix in a reservoir is built by a swappable function, named in a registry or passed as a bare callable.
---

<span class="nb-kicker">Build</span>

# Structure is a function

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

## Any function is a topology

The escape hatch that keeps the system open: a bare callable is a valid spec, no registration required.

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
    `spectral_radius` rescales the result *afterwards*. A topology never needs to worry about its own eigenvalues.

---

## The catalogs

<div class="grid cards" markdown>

- **[Topology catalog](topologies/index.md)**

    ---

    Every registered recurrent-matrix builder with its connectivity
    portrait — generated from the live registry at build time, so
    registering a topology adds its page on the next build.

- **[Initializer catalog](initializers/index.md)**

    ---

    Every registered input/feedback initializer with a portrait of the
    matrix it draws — generated from the live registry, so new
    initializers document themselves.

</div>

## Naming your own

Registration makes a builder usable by string in every layer and factory —
and therefore sweepable by name in HPO. Matrix builders return the matrix
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
an `InputFeedbackInitializer` subclass when state or validation earns a class:

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
