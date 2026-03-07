# Topologies

The topology system controls the **structure** of the reservoir's recurrent weight matrix \(W\). Rather than a dense random matrix, you can impose graph structures that affect the reservoir's connectivity, modularity, and dynamics.

---

## Specifying a Topology

Three equivalent ways to set the topology when constructing `ESNLayer`:

=== "String (registry defaults)"
    ```python
    reservoir = ESNLayer(500, feedback_size=3, topology="watts_strogatz")
    ```

=== "Tuple (custom params)"
    ```python
    reservoir = ESNLayer(
        500,
        feedback_size=3,
        topology=("watts_strogatz", {"k": 6, "p": 0.3}),
    )
    ```

=== "Object (pre-configured)"
    ```python
    from resdag.init.topology import get_topology

    topo = get_topology("watts_strogatz", k=6, p=0.3, seed=42)
    reservoir = ESNLayer(500, feedback_size=3, topology=topo)
    ```

When `topology=None` (the default), a dense random weight matrix is used.

---

## Registry API

```python
from resdag.init.topology import get_topology, show_topologies

# List all available topologies
show_topologies()

# Get details for a specific one
show_topologies("erdos_renyi")

# Create a topology object
topo = get_topology("erdos_renyi", p=0.1, directed=True, seed=0)
```

---

## Available Topologies

<div class="rd-topology-grid">
<div class="rd-topology-item">erdos_renyi</div>
<div class="rd-topology-item">connected_erdos_renyi</div>
<div class="rd-topology-item">watts_strogatz</div>
<div class="rd-topology-item">connected_watts_strogatz</div>
<div class="rd-topology-item">newman_watts_strogatz</div>
<div class="rd-topology-item">barabasi_albert</div>
<div class="rd-topology-item">kleinberg_small_world</div>
<div class="rd-topology-item">complete</div>
<div class="rd-topology-item">regular</div>
<div class="rd-topology-item">ring_chord</div>
<div class="rd-topology-item">multi_cycle</div>
<div class="rd-topology-item">simple_cycle_jumps</div>
<div class="rd-topology-item">dendrocycle</div>
<div class="rd-topology-item">chord_dendrocycle</div>
<div class="rd-topology-item">spectral_cascade</div>
<div class="rd-topology-item">random</div>
<div class="rd-topology-item">zero</div>
</div>

---

## Topology Reference

### `erdos_renyi`
Random graph where each directed edge exists independently with probability `p`.

```python
topology = ("erdos_renyi", {"p": 0.1, "directed": True})
```

| Param | Default | Description |
|---|---|---|
| `p` | `0.1` | Edge probability |
| `directed` | `True` | Directed or undirected |
| `seed` | `None` | Random seed |

**Best for**: General-purpose, baseline experiments. Tends to produce sparse, random connectivity.

---

### `connected_erdos_renyi`
Like `erdos_renyi` but guaranteed to produce a connected (or weakly connected) graph.

```python
topology = ("connected_erdos_renyi", {"p": 0.05})
```

---

### `watts_strogatz`
**Small-world** network — a ring lattice with random rewiring. Combines high clustering with short path lengths.

```python
topology = ("watts_strogatz", {"k": 6, "p": 0.3})
```

| Param | Default | Description |
|---|---|---|
| `k` | `4` | Each node connected to k nearest neighbors in ring |
| `p` | `0.3` | Rewiring probability |
| `seed` | `None` | Random seed |

**Best for**: Systems requiring a balance between local structure and global information flow.

---

### `connected_watts_strogatz`
Like `watts_strogatz` but ensures connectivity. More robust for small `p`.

---

### `newman_watts_strogatz`
Newman–Watts–Strogatz variant: adds shortcuts without removing existing ring edges.

```python
topology = ("newman_watts_strogatz", {"k": 4, "p": 0.1})
```

---

### `barabasi_albert`
**Scale-free** network — preferential attachment produces hubs. Few high-degree nodes, many low-degree nodes.

```python
topology = ("barabasi_albert", {"m": 3})
```

| Param | Default | Description |
|---|---|---|
| `m` | `2` | Number of edges to attach per new node |
| `seed` | `None` | Random seed |

**Best for**: Modeling systems with hub-and-spoke structure; often more robust reservoirs.

---

### `kleinberg_small_world`
Kleinberg's small-world model on a 2D lattice with long-range connections following a power law.

```python
topology = ("kleinberg_small_world", {"p": 2, "q": 1, "r": 2})
```

---

### `complete`
Fully connected graph — every node connects to every other node. Maximally expressive but dense.

```python
topology = "complete"
```

**Best for**: Small reservoirs or ablation studies. Scales quadratically — avoid for large `reservoir_size`.

---

### `regular`
Each node has exactly `d` connections. Uniform connectivity.

```python
topology = ("regular", {"d": 4})
```

| Param | Default | Description |
|---|---|---|
| `d` | `3` | Degree (connections per node) |

---

### `ring_chord`
A ring topology with additional long-range chord connections.

```python
topology = ("ring_chord", {"chord_length": 10})
```

---

### `multi_cycle`
Multiple overlapping cycles, creating structured redundancy.

```python
topology = ("multi_cycle", {"n_cycles": 3})
```

---

### `simple_cycle_jumps`
A simple cycle with random jump connections for long-range communication.

```python
topology = "simple_cycle_jumps"
```

---

### `dendrocycle`
Dendritic cycle structure — combines tree-like branches with cycle connections.

```python
topology = "dendrocycle"
```

**Best for**: Hierarchical dynamics; suited for multi-scale temporal patterns.

---

### `chord_dendrocycle`
Dendrocycle with additional chord connections.

```python
topology = "chord_dendrocycle"
```

---

### `spectral_cascade`
Cascade structure designed around spectral properties for controlled memory decay.

```python
topology = "spectral_cascade"
```

---

### `random`
Alias for random sparse connections, similar to `erdos_renyi` with different default density.

```python
topology = "random"
```

---

### `zero`
No recurrent connections. The reservoir degenerates to a purely input-driven system.

```python
topology = "zero"
```

**Best for**: Ablation — understanding contribution of recurrent connections.

---

## Adding Custom Topologies

See [Extending resdag → New Topology](../extending/new-topology.md) for a step-by-step guide.

```python
from resdag.init.topology import register_graph_topology
import networkx as nx

@register_graph_topology("my_graph", density=0.05, directed=True)
def my_graph(n: int, density: float = 0.05, directed: bool = True, seed=None) -> nx.DiGraph:
    """My custom reservoir topology."""
    rng = np.random.default_rng(seed)
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if i != j and rng.random() < density:
                G.add_edge(i, j, weight=rng.normal())
    return G
```

---

## Topology and Spectral Radius

Regardless of topology, the `spectral_radius` parameter of `ESNLayer` rescales the weight matrix so that \(\rho(W) = \text{spectral\_radius}\). This decouples the structural choice (which edges exist) from the spectral scaling.

```python
# Same topology, different spectral radii
for sr in [0.7, 0.9, 0.95, 1.1]:
    reservoir = ESNLayer(
        500, feedback_size=3,
        topology="watts_strogatz",
        spectral_radius=sr,
    )
```
