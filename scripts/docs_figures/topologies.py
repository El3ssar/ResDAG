"""Connectivity heatmaps for every registered graph topology.

Each panel shows the adjacency matrix of a 100-node reservoir built from
the topology with its default parameters. Black/indigo pixels are non-zero
weights; white is empty.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from resdag.init.topology import get_topology

from ._common import AMBER, INDIGO, RULE, SOFT, apply_style, save


N_NODES = 100
# Some topologies need a special N (e.g. spectral_cascade requires a
# triangular number; kleinberg_small_world treats N as grid side length
# producing N² nodes).  Map name → override.
N_NODES_OVERRIDE: dict[str, int] = {
    "spectral_cascade": 105,
    "kleinberg_small_world": 10,    # 10×10 = 100 nodes
}
DEFAULT_SR = 0.9

# Topologies to include, with optional parameter overrides per topology so
# every panel produces an interesting matrix at n=100.
TOPOLOGIES: list[tuple[str, dict]] = [
    ("erdos_renyi",              {"p": 0.05}),
    ("connected_erdos_renyi",    {"p": 0.05}),
    ("watts_strogatz",           {"k": 6, "p": 0.2}),
    ("newman_watts_strogatz",    {"k": 6, "p": 0.2}),
    ("connected_watts_strogatz", {"k": 6, "p": 0.2}),
    ("barabasi_albert",          {"m": 3}),
    ("regular",                  {"k": 4}),
    ("complete",                 {}),
    ("ring_chord",               {"L": 4, "w": 0.5, "alpha": 1.0}),
    ("simple_cycle_jumps",       {"jump_length": 3}),
    ("multi_cycle",              {"k": 4}),
    ("dendrocycle",              {"c": 0.5, "d": 0.2}),
    ("chord_dendrocycle",        {"c": 0.5, "d": 0.2}),
    ("spectral_cascade",         {"spectral_radius": 0.9}),
    ("kleinberg_small_world",    {"q": 2, "k": 1}),
    ("random",                   {"density": 0.05}),
    ("zeros",                    {}),
]


_CMAP = LinearSegmentedColormap.from_list(
    "indigo_white", [(0, "#ffffff"), (0.5, INDIGO), (1.0, AMBER)],
)


def _adjacency(name: str, params: dict) -> np.ndarray:
    """Sample one adjacency matrix from a topology and return its |·|."""
    import networkx as nx
    import torch

    # ``kleinberg_small_world`` interprets ``n`` as the grid side, returning
    # an n²-node graph. The TopologyInitializer wrapper doesn't reconcile
    # that, so we go around it and call the graph function directly.
    if name == "kleinberg_small_world":
        from resdag.init.graphs.kleinberg_small_world import (
            kleinberg_small_world_graph,
        )
        G = kleinberg_small_world_graph(10, **params)
        adj = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
        return np.abs(adj).astype(np.float32)

    n = N_NODES_OVERRIDE.get(name, N_NODES)
    topo = get_topology(name, **params)
    w = torch.zeros(n, n)
    try:
        topo.initialize(w, spectral_radius=DEFAULT_SR)
    except Exception:  # pragma: no cover
        topo.initialize(w)
    return w.detach().abs().cpu().numpy()


def grid_all_topologies() -> None:
    """A single grid figure with every topology's adjacency."""
    n = len(TOPOLOGIES)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 2.4, rows * 2.4), constrained_layout=True,
    )
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax, (name, params) in zip(axes, TOPOLOGIES):
        mat = _adjacency(name, params)
        ax.imshow(mat, cmap=_CMAP, interpolation="nearest",
                  vmin=0, vmax=max(mat.max(), 1e-12))
        ax.set_title(name, fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(RULE)
    # Hide leftover axes
    for ax in axes[len(TOPOLOGIES):]:
        ax.set_visible(False)
    fig.suptitle(
        f"Adjacency matrices for every registered topology (n = {N_NODES}, "
        f"spectral radius = {DEFAULT_SR})",
        fontsize=11, y=1.02,
    )
    save(fig, "topologies_grid.png")


def individual_topology(name: str, params: dict) -> None:
    """One large heatmap of a single topology."""
    mat = _adjacency(name, params)
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    ax.imshow(mat, cmap=_CMAP, interpolation="nearest",
              vmin=0, vmax=max(mat.max(), 1e-12))
    ax.set_title(f"{name} — n = {N_NODES}")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(RULE)
    save(fig, f"topology_{name}.png")


def main() -> None:
    apply_style()
    print("Topologies …")
    grid_all_topologies()
    # Pick four widely-used topologies for full-size panels.
    for name in ("erdos_renyi", "watts_strogatz", "barabasi_albert",
                 "ring_chord"):
        for n, p in TOPOLOGIES:
            if n == name:
                individual_topology(name, p)
                break


if __name__ == "__main__":
    main()
