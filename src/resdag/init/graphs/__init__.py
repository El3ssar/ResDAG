"""Graph topology implementations for reservoir weight initialization.

This module contains pure graph generation functions that produce adjacency
matrices using networkx. These are wrapped by the topology system to initialize
reservoir weight tensors.

All graph functions follow the signature:
    graph_func(n: int, *args, **kwargs) -> nx.Graph | nx.DiGraph

Where:
    - n: Number of nodes (will be reservoir_size)
    - *args, **kwargs: Graph-specific parameters
    - Returns: NetworkX Graph or DiGraph with weighted edges
"""

from .barabasi_albert import barabasi_albert_graph
from .chord_dendrocycle import dendrocycle_with_chords_graph
from .complete import complete_graph
from .connected_erdos_renyi import connected_erdos_renyi_graph
from .connected_watts_strogatz import connected_watts_strogatz_graph
from .dendrocycle import dendrocycle_graph
from .erdos_renyi import erdos_renyi_graph
from .kleinberg_small_world import kleinberg_small_world_graph
from .multi_cycle import multi_cycle_graph
from .newman_watts_strogatz import newman_watts_strogatz_graph
from .random import random_graph
from .regular import regular_graph
from .ring_chord import ring_chord_graph
from .simple_cycle_jumps import simple_cycle_jumps_graph
from .spectral_cascade import spectral_cascade_graph
from .watts_strogatz import watts_strogatz_graph

__all__ = [
    "barabasi_albert_graph",
    "chord_dendrocycle_graph",
    "complete_graph",
    "connected_erdos_renyi_graph",
    "connected_watts_strogatz_graph",
    "dendrocycle_graph",
    "dendrocycle_with_chords_graph",
    "erdos_renyi_graph",
    "kleinberg_small_world_graph",
    "multi_cycle_graph",
    "newman_watts_strogatz_graph",
    "random_graph",
    "regular_graph",
    "ring_chord_graph",
    "simple_cycle_jumps_graph",
    "spectral_cascade_graph",
    "watts_strogatz_graph",
]
