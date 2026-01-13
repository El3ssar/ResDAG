from math import isqrt

import numpy as np
from networkx import Graph

from resdag.init.topology.registry import register_graph_topology


@register_graph_topology(
    "spectral_cascade",
    spectral_radius=1.0,
    self_loops=False,
)
def spectral_cascade_graph(
    n: int,
    spectral_radius: float,
    self_loops: bool = False,
) -> Graph:
    """
    Generates a graph of N disconnected components (cliques) of sizes 1..N, where n = N(N+1)/2.
    Each clique is fully connected (or complete in the directed sense),
    with deterministic weights in {+1, -1}:
      - off-diagonal: (-1)^(i + j)
      - diagonal (if self_loops=True): (-1)^i

    Then each clique's adjacency block is spectrally scaled so that:
      - The 1-node clique has spectral radius 0 (trivial).
      - The 2-node clique has spectral radius = `spectral_radius`.
      - Each larger clique k=3..N has spectral radius (N - k + 1)*spectral_radius / N.

    Parameters
    ----------
    n : int
        Total number of nodes (must be triangular, i.e. n = N(N+1)/2).
    spectral_radius : float
        The "starting" radius for the 2-node clique, which is then decreased
        across bigger cliques down to sr/N for the largest clique.
    self_loops : bool, optional
        If True, each node has a self-loop of weight = (-1)^i. Default: False.

    Returns
    -------
    networkx.Graph
        A NetworkX graph with block-diagonal clique structure,
        spectrally scaled as specified.

    Raises
    ------
    ValueError
        If `n` is not triangular.
    """

    # 1) Check that n is triangular:  n = N(N+1)/2.
    D = 1 + 8 * n
    sqrt_D = isqrt(D)
    if sqrt_D * sqrt_D != D or (sqrt_D - 1) % 2 != 0:
        raise ValueError(f"{n} is not a triangular number (N(N+1)/2 for some integer N).")

    N = (sqrt_D - 1) // 2  # number of cliques

    # 2) Prepare an n x n adjacency matrix
    A = np.zeros((n, n), dtype=float)

    offset = 0
    for k in range(1, N + 1):
        size_k = k
        # adjacency block for this clique is A_sub, shape (k, k).
        # We'll fill it, compute its largest eigenvalue, then scale.

        # If k=1 => trivial, spectral radius = 0 => skip edges
        if k == 1:
            offset += 1
            continue

        # Build the local adjacency block
        A_sub = np.zeros((size_k, size_k), dtype=float)

        for i in range(size_k):
            global_i = offset + i
            for j in range(size_k):
                global_j = offset + j
                if i == j:
                    # Self-loop if requested
                    if self_loops:
                        # weight = (-1)^global_i
                        A_sub[i, j] = (-1) ** (global_i)
                else:
                    # Normal edge in a "complete" sense
                    w = (-1) ** (global_i + global_j)
                    A_sub[i, j] = w

        # 3) Desired spectral radius for this clique
        if k == 2:
            desired_r = spectral_radius
        else:
            desired_r = (N - k + 1) * spectral_radius / N

        # 4) Scale block by ratio (desired_r / actual_r)
        #    First compute largest eigenvalue magnitude:
        vals = np.linalg.eigvals(A_sub)
        current_r = max(abs(vals)) if len(vals) > 0 else 0.0

        if current_r > 1e-14 and desired_r != 0:
            scale_factor = desired_r / current_r
            A_sub *= scale_factor

        # 5) Insert A_sub back into the global adjacency matrix A
        A[offset : offset + k, offset : offset + k] = A_sub

        offset += k

    # 6) Build the final networkx Graph/DiGraph from adjacency
    G = Graph()
    G.add_nodes_from(range(n))

    # Mirror the upper/lower triangle
    A = np.triu(A) + np.triu(A, 1).T

    # Now add edges with 'weight'
    for i in range(n):
        for j in range(n):
            if i == j and not self_loops:
                # Possibly remove diagonal if self_loops=False
                continue
            w = A[i, j]
            if abs(w) > 1e-14:  # avoid floating garbage
                G.add_edge(i, j, weight=w)

    return G
