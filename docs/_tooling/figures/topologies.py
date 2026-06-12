"""Per-topology portraits: circular connectivity + weight matrix."""

import networkx as nx
import numpy as np
import torch

from _style import ACCENT, DIVERGING, INK_SOFT, RULE, plt, save


def main() -> None:
    from resdag.init.topology import get_topology
    from resdag.init.topology.registry import _TOPOLOGY_REGISTRY

    SIZE = {"kleinberg_small_world": 49, "spectral_cascade": 45}
    for name in sorted(_TOPOLOGY_REGISTRY):
        n = SIZE.get(name, 48)
        topo = get_topology(name)
        w = torch.empty(n, n)
        try:
            topo.initialize(w, spectral_radius=None)
        except Exception as e:
            print(f"  skip {name}: {e}")
            continue
        A = w.numpy()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 3.0),
                                       gridspec_kw={"width_ratios": [1.15, 1]})
        G = nx.from_numpy_array((np.abs(A) > 1e-9).astype(int), create_using=nx.DiGraph)
        pos = nx.circular_layout(G)
        nx.draw_networkx_edges(G, pos, ax=ax1, edge_color=ACCENT, alpha=0.25,
                               width=0.7, arrows=False)
        nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=22, node_color=INK_SOFT,
                               linewidths=0)
        ax1.set_title(f"connectivity (n={n})")
        ax1.set_axis_off()
        vmax = np.abs(A).max() or 1
        ax2.imshow(A, cmap=DIVERGING, vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax2.set_title("weight matrix")
        ax2.set_xticks([]); ax2.set_yticks([])
        for s in ax2.spines.values():
            s.set_visible(True); s.set_color(RULE)
        save(fig, f"topologies/{name}.png")


if __name__ == "__main__":
    main()
