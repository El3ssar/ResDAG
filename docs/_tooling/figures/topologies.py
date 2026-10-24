"""Per-topology portraits: circular connectivity + weight matrix."""

import hashlib
import inspect
import json

import networkx as nx
import numpy as np
import torch
from _style import ACCENT, DIVERGING, INK_SOFT, RULE, plt, save


def main() -> None:
    from resdag.init.topology import get_topology
    from resdag.init.topology.registry import _TOPOLOGY_REGISTRY

    sizes = {"kleinberg_small_world": 49, "spectral_cascade": 45}
    for name in sorted(_TOPOLOGY_REGISTRY):
        n = sizes.get(name, 48)
        builder = _TOPOLOGY_REGISTRY[name][0]
        options = {"seed": 3} if "seed" in inspect.signature(builder).parameters else {}
        topo = get_topology(name, **options)
        w = torch.empty(n, n, dtype=torch.float64)
        topo.initialize(w, spectral_radius=None)
        matrix = w.numpy()
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(6.6, 3.0), gridspec_kw={"width_ratios": [1.15, 1]}
        )
        graph = nx.from_numpy_array((np.abs(matrix) > 1e-9).astype(int), create_using=nx.DiGraph)
        pos = nx.circular_layout(graph)
        nx.draw_networkx_edges(
            graph, pos, ax=ax1, edge_color=ACCENT, alpha=0.25, width=0.7, arrows=False
        )
        nx.draw_networkx_nodes(graph, pos, ax=ax1, node_size=22, node_color=INK_SOFT, linewidths=0)
        ax1.set_title(f"connectivity (n={n})")
        ax1.set_axis_off()
        vmax = np.abs(matrix).max() or 1
        ax2.imshow(matrix, cmap=DIVERGING, vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax2.set_title("weight matrix")
        ax2.set_xticks([])
        ax2.set_yticks([])
        for s in ax2.spines.values():
            s.set_visible(True)
            s.set_color(RULE)
        save(
            fig,
            f"topologies/{name}.png",
            metadata={
                "Description": json.dumps(
                    {
                        "component": name,
                        "shape": list(matrix.shape),
                        "parameters": options,
                        "matrix_sha256": hashlib.sha256(matrix.astype("<f8").tobytes()).hexdigest(),
                    },
                    sort_keys=True,
                )
            },
        )


if __name__ == "__main__":
    main()
