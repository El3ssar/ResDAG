# Graph Topology System

## Overview

The topology system in `resdag` provides a powerful way to initialize reservoir recurrent weight matrices using graph structures. Instead of random initialization, you can use well-studied graph topologies from network science to control the connectivity patterns and dynamics of your reservoirs.

## Key Concepts

### Why Graph Topologies?

Different graph structures lead to different reservoir dynamics:

- **Erdős-Rényi (Random)**: Uniform connectivity, good baseline
- **Watts-Strogatz (Small-World)**: Local clustering + long-range connections
- **Barabási-Albert (Scale-Free)**: Hub nodes, power-law degree distribution
- **Ring with Chords**: Delay lines with shortcuts, good for temporal tasks
- **Regular Lattice**: Local connectivity, structured dynamics

### Architecture

The topology system has three layers:

1. **Graph Functions** (`resdag.graphs`): Pure NetworkX graph generators
2. **Topology Interface** (`resdag.topology`): Converts graphs → PyTorch tensors
3. **Reservoir Integration**: Automatic initialization in `ReservoirLayer`

## Usage

### Basic Usage with Named Topologies

```python
from resdag.layers import ReservoirLayer

# Use a pre-registered topology by name
reservoir = ReservoirLayer(
    reservoir_size=100,
    feedback_size=10,
    topology="erdos_renyi",
    spectral_radius=0.9,
)
```

### Available Named Topologies

```python
from resdag.init.topology import show_topologies

# List all available topologies
print(show_topologies())
# ['erdos_renyi', 'watts_strogatz', 'barabasi_albert', 'regular',
#  'complete', 'ring_chord', 'cycle_jumps', 'multi_cycle',
#  'dendrocycle', 'spectral_cascade']

# Show details for a specific topology
print(show_topologies("erdos_renyi"))
# {'name': 'erdos_renyi', 'defaults': {...}, 'parameters': {...}}
```

### Custom Topology Parameters

```python
from resdag.topology import get_topology

# Customize parameters for a named topology
topology = get_topology(
    "watts_strogatz",
    k=6,           # Each node connects to 6 neighbors
    p=0.1,         # 10% rewiring probability
    directed=True,
    seed=42        # For reproducibility
)

reservoir = ReservoirLayer(
    reservoir_size=200,
    feedback_size=5,
    topology=topology,
    spectral_radius=0.95,
)
```

### Creating Custom Topologies

```python
from resdag.topology import GraphTopology
from resdag.graphs import erdos_renyi_graph

# Direct control over graph generation
custom_topology = GraphTopology(
    graph_func=erdos_renyi_graph,
    graph_kwargs={
        "p": 0.15,
        "directed": True,
        "self_loops": True,
        "seed": 123
    }
)

reservoir = ReservoirLayer(
    reservoir_size=150,
    feedback_size=8,
    topology=custom_topology,
)
```

### Registering New Topologies

```python
from resdag.topology import register_topology
from my_custom_graphs import my_special_graph

# Register your own topology
register_topology(
    name="my_topology",
    graph_func=my_special_graph,
    default_kwargs={"param1": 1.0, "param2": True}
)

# Now use it by name
reservoir = ReservoirLayer(
    reservoir_size=100,
    feedback_size=10,
    topology="my_topology",
)
```

## Available Graph Topologies

### Random Graphs

#### Erdős-Rényi (`erdos_renyi`)

Random graph where each edge exists with probability `p`.

- **Parameters**: `p` (edge probability), `directed`, `self_loops`, `seed`
- **Use case**: Baseline, uniform connectivity

#### Connected Erdős-Rényi (`connected_erdos_renyi`)

Guaranteed connected Erdős-Rényi graph.

- **Parameters**: Same as `erdos_renyi`, plus `tries` (connection attempts)

### Small-World Graphs

#### Watts-Strogatz (`watts_strogatz`)

Ring lattice with random rewiring.

- **Parameters**: `k` (neighbors), `p` (rewiring prob), `directed`, `self_loops`, `seed`
- **Use case**: Balance between local and global information flow

#### Newman-Watts-Strogatz (`newman_watts_strogatz`)

Like Watts-Strogatz but adds shortcuts without removing edges.

- **Parameters**: `k`, `p`, `directed`, `self_loops`, `seed`

#### Kleinberg Small-World (`kleinberg_small_world`)

2D grid with distance-dependent long-range connections.

- **Parameters**: `n` (grid size), `q`, `k`, `directed`, `weighted`, `beta`, `seed`

#### Ring with Chords (`ring_chord`)

Directed ring with backward shortcuts.

- **Parameters**: `L` (delay), `w` (chord weight), `alpha` (decay)
- **Use case**: Delay-line reservoirs, temporal patterns

### Scale-Free Graphs

#### Barabási-Albert (`barabasi_albert`)

Preferential attachment → power-law degree distribution.

- **Parameters**: `m` (edges per new node), `directed`, `seed`
- **Use case**: Hub structures, hierarchical processing

### Structured Graphs

#### Regular Lattice (`regular`)

Ring where each node connects to `k` neighbors.

- **Parameters**: `k`, `directed`, `self_loops`, `random_weights`, `seed`
- **Use case**: Local, structured connectivity

#### Complete Graph (`complete`)

Every node connected to every other node.

- **Parameters**: `self_loops`, `random_weights`, `seed`
- **Use case**: Fully connected, dense reservoirs

#### Multi-Cycle (`multi_cycle`)

Multiple disjoint cycles.

- **Parameters**: `k` (number of cycles), `weight`
- **Use case**: Parallel oscillators

### Specialized Reservoirs

#### Dendrocycle (`dendrocycle`)

Core cycle with dendritic chains and quiescent nodes.

- **Parameters**: `c` (core fraction), `d` (dendrite fraction), weights, `seed`
- **Use case**: Hierarchical dynamics

#### Dendrocycle with Chords (`chord_dendrocycle`)

Dendrocycle + small-world shortcuts on core.

- **Parameters**: Dendrocycle params + `L`, `w`, `alpha`

#### Simple Cycle with Jumps (`cycle_jumps`)

Cycle with bidirectional jump edges.

- **Parameters**: `jump_length`, `r_c` (cycle weight), `r_l` (jump weight)

#### Spectral Cascade (`spectral_cascade`)

Disconnected cliques with controlled spectral properties.

- **Parameters**: `spectral_radius`, `self_loops`
- **Use case**: Precise spectral control

## Implementation Details

### How It Works

1. **Graph Generation**: NetworkX graph created with `n = reservoir_size`
2. **Adjacency Extraction**: Convert graph to weighted adjacency matrix
3. **Spectral Scaling**: Scale matrix to target spectral radius
4. **Tensor Initialization**: Copy to PyTorch parameter

### Spectral Radius

The spectral radius (largest absolute eigenvalue) controls reservoir dynamics:

- **< 1**: Echo state property, stable dynamics
- **≈ 0.9**: Common choice, good memory-nonlinearity balance
- **> 1**: Unstable, amplifies signals (rarely used)

The topology system automatically scales the adjacency matrix to achieve the target spectral radius.

### Reproducibility

Use `seed` parameters for reproducible graphs:

```python
topology1 = get_topology("erdos_renyi", p=0.1, seed=42)
topology2 = get_topology("erdos_renyi", p=0.1, seed=42)

# Same seed → identical graphs → identical reservoirs
```

## Best Practices

### Choosing a Topology

1. **Start with `erdos_renyi`**: Good baseline, well-understood
2. **Try `watts_strogatz`** for tasks with mixed local/global structure
3. **Use `ring_chord`** for temporal/sequential tasks
4. **Experiment with `dendrocycle`** for hierarchical patterns

### Hyperparameter Tuning

Key parameters to tune:

- **Topology type**: Different structures for different tasks
- **Spectral radius** (0.7-0.99): Higher = more nonlinearity, less stability
- **Graph parameters**: `p` (density), `k` (connectivity), etc.

### Performance Tips

- Graph generation happens once during initialization (not forward pass)
- GPU tensors work automatically
- Topology doesn't affect inference speed (only initialization)

## Advanced Topics

### Custom Graph Functions

Your graph function must follow this signature:

```python
def my_graph(n: int, *args, **kwargs) -> nx.Graph | nx.DiGraph:
    """Generate a graph with n nodes."""
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    # ... add edges with weights ...
    return G
```

Then wrap it in `GraphTopology`:

```python
from resdag.topology import GraphTopology

topology = GraphTopology(my_graph, {"param1": value1})
```

### Non-Square Topologies (Future)

Currently, topologies only apply to recurrent weights (square matrices).
Future versions may support:

- Input weight topologies (non-square)
- Feedback weight topologies (non-square)
- Bipartite graphs for layered structures

## Examples

See `examples/01_reservoir_with_topology.py` for complete working examples.

## References

- Rodan, A., & Tiňo, P. (2011). "Minimum complexity echo state network." _IEEE TNNLS_
- Gallicchio, C., & Micheli, A. (2011). "Architectural and Markovian factors of echo state networks." _Neural Networks_
- Watts, D. J., & Strogatz, S. H. (1998). "Collective dynamics of 'small-world' networks." _Nature_
- Barabási, A. L., & Albert, R. (1999). "Emergence of scaling in random networks." _Science_
