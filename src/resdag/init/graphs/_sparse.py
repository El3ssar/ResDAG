"""Memory-bounded sparse adjacency construction for graph initializers."""

from typing import Any, Callable

import networkx as nx
import numpy as np
import scipy.sparse as sp

from resdag.utils.general import create_rng


def _erdos_renyi_csr(
    n: int,
    p: float,
    directed: bool = True,
    self_loops: bool = True,
    seed: Any = None,
) -> sp.csr_matrix:
    """Match dense ER draws exactly, retaining only selected coordinates.

    The two random passes have exactly the dense builder's draw order. Each
    temporary block has at most 32 rows rather than allocating an n-by-n mask
    and sign array. Work remains quadratic; stored weights scale with edges.
    """
    rng = create_rng(seed)
    coordinates = []
    for start in range(0, n, 32):
        stop = min(n, start + 32)
        mask = rng.random((stop - start, n)) < p
        row, col = np.nonzero(mask)
        row = row + start
        keep = np.ones(len(row), dtype=bool)
        if not directed:
            keep &= row <= col
        if not self_loops:
            keep &= row != col
        coordinates.append((row[keep], col[keep]))
    rows, columns, values = [], [], []
    for block, start in enumerate(range(0, n, 32)):
        stop = min(n, start + 32)
        signs = rng.random((stop - start, n))
        row, col = coordinates[block]
        data = np.where(signs[row - start, col] < 0.5, -1.0, 1.0).astype(np.float32)
        rows.append(row)
        columns.append(col)
        values.append(data)
        if not directed:
            off_diagonal = row != col
            rows.append(col[off_diagonal])
            columns.append(row[off_diagonal])
            values.append(data[off_diagonal])
    return sp.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))), shape=(n, n)
    ).tocsr()


def sparse_adjacency(graph_func: Callable, n: int, kwargs: dict[str, Any]) -> sp.csr_matrix:
    from .erdos_renyi import erdos_renyi_graph

    if graph_func is erdos_renyi_graph:
        return _erdos_renyi_csr(n, **kwargs)
    graph = graph_func(n, **kwargs)
    if len(graph) != n:
        raise ValueError(f"Graph produced {len(graph)} nodes, expected {n}")
    return sp.csr_matrix(
        nx.to_scipy_sparse_array(
            graph, nodelist=sorted(graph.nodes()), weight="weight", dtype=np.float64
        )
    )
