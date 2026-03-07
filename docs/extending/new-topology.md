# Adding a New Topology

Topology functions generate a NetworkX graph that defines the **structure** of the reservoir's recurrent weight matrix.

---

## Step 1: Create the graph file

Create `src/resdag/init/graphs/my_topology.py`:

```python
"""
My Custom Topology
==================
Description of what this topology does and when to use it.
"""

import numpy as np
import networkx as nx

from resdag.init.topology import register_graph_topology


@register_graph_topology(
    "my_topology",
    density=0.05,     # default parameter values
    directed=True,
    allow_self_loops=False,
)
def my_topology(
    n: int,
    density: float = 0.05,
    directed: bool = True,
    allow_self_loops: bool = False,
    seed: int | None = None,
) -> nx.DiGraph:
    """
    My custom reservoir topology.

    Parameters
    ----------
    n : int
        Number of nodes (reservoir size).
    density : float
        Edge probability (fraction of all possible edges).
    directed : bool
        Whether to create a directed graph.
    allow_self_loops : bool
        Whether to allow self-connections.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    nx.DiGraph or nx.Graph
        Graph with weighted edges (float 'weight' attribute).
    """
    rng = np.random.default_rng(seed)
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(n):
            if i == j and not allow_self_loops:
                continue
            if rng.random() < density:
                weight = rng.normal()
                G.add_edge(i, j, weight=weight)

    return G
```

!!! important "Required signature"
    - First argument must be `n: int` (reservoir size)
    - Must accept `seed` as a keyword argument
    - Must return a `nx.Graph` or `nx.DiGraph`
    - Edges must have a `weight` float attribute

---

## Step 2: Import in `__init__.py`

Add your import to `src/resdag/init/graphs/__init__.py`:

```python
# ... existing imports ...
from . import my_topology  # noqa: F401 — import triggers registration
```

---

## Step 3: Use it

After importing, the topology is available by name everywhere:

```python
from resdag.init.graphs import my_topology   # triggers registration
from resdag.init.topology import get_topology
from resdag.layers import ESNLayer

# By name string
reservoir = ESNLayer(500, feedback_size=3, topology="my_topology")

# With custom params
reservoir = ESNLayer(
    500, feedback_size=3,
    topology=("my_topology", {"density": 0.1, "directed": False}),
)

# As object
topo = get_topology("my_topology", density=0.08, seed=42)
reservoir = ESNLayer(500, feedback_size=3, topology=topo)
```

---

## Registry API

```python
from resdag.init.topology import register_graph_topology, get_topology, show_topologies

# List all registered topologies (including yours)
show_topologies()

# Get topology object
topo = get_topology("my_topology", density=0.05)

# Get topology info
show_topologies("my_topology")
```

---

## Tips for Good Topologies

- **Weight distribution**: Use `rng.normal()` or `rng.uniform(-1, 1)` for weights; they'll be rescaled by `spectral_radius` anyway
- **Connectivity**: Ensure the graph is (weakly) connected, or warn if it's not
- **Sparsity**: Sparse graphs (5–15% density) typically outperform dense ones for large reservoirs
- **Seed**: Always pass `seed` to `np.random.default_rng()` for reproducibility
- **Directed vs undirected**: Directed graphs are standard for ESNs (asymmetric connections)

---

## Example: Ring + Random Shortcuts

```python
@register_graph_topology("ring_shortcuts", n_shortcuts=10)
def ring_shortcuts(
    n: int,
    n_shortcuts: int = 10,
    seed: int | None = None,
) -> nx.DiGraph:
    """Ring topology with random long-range shortcuts."""
    rng = np.random.default_rng(seed)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    # Ring connections
    for i in range(n):
        G.add_edge(i, (i + 1) % n, weight=rng.normal())

    # Random shortcuts
    for _ in range(n_shortcuts):
        i = rng.integers(n)
        j = rng.integers(n)
        if i != j:
            G.add_edge(i, j, weight=rng.normal())

    return G
```
