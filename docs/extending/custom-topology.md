# Add a graph topology

Implement a NetworkX generator and register it.

```python
import networkx as nx
from resdag.init.topology import register_graph_topology


@register_graph_topology("my_ring", weight=1.0, directed=True)
def my_ring_graph(n: int, weight: float = 1.0, directed: bool = True, seed=None) -> nx.DiGraph:
    g = nx.DiGraph() if directed else nx.Graph()
    g.add_nodes_from(range(n))
    for i in range(n):
        g.add_edge(i, (i + 1) % n, weight=weight)
    return g
```

Import the module from `resdag.init.graphs` (or rely on your package importing it) so
the decorator runs at startup.

Use it:

```python
from resdag.layers import ESNLayer

ESNLayer(200, feedback_size=3, topology="my_ring")
ESNLayer(200, feedback_size=3, topology=("my_ring", {"weight": 0.5}))
```

Requirements:

- First argument is always `n` (reservoir size).
- Return `nx.Graph` or `nx.DiGraph` with numeric `weight` on edges.
- See [`erdos_renyi_graph`](https://github.com/El3ssar/resdag/blob/main/src/resdag/init/graphs/erdos_renyi.py) for the canonical pattern.

## See also

- [Topologies (Learn)](../learn/topologies.md)
- [Graph reference](../reference/init/graphs.md)
