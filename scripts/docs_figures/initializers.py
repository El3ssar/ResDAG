"""Weight-matrix heatmaps for every registered input/feedback initializer."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm

from resdag.init.input_feedback import get_input_feedback

from ._common import RULE, apply_style, save


# (reservoir_size, feedback_size) for a rectangular weight matrix.
# 96 = 8 × 12 so ``chain_of_neurons_input`` (which needs reservoir to
# be a multiple of ``features``) fits cleanly.
ROWS = 96
COLS = 12

# Every registered initializer. Override params only when needed for a
# visually informative panel.
INITIALIZERS: list[tuple[str, dict]] = [
    ("random",                  {"input_scaling": 1.0}),
    ("random_binary",           {}),
    ("binary_balanced",         {}),
    ("chebyshev",               {}),
    ("chessboard",              {}),
    ("pseudo_diagonal",         {}),
    ("opposite_anchors",        {}),
    ("ring_window",             {"c": 0.4, "window": 5}),
    ("chain_of_neurons_input",  {"features": 12}),
    ("dendrocycle_input",       {"c": 0.5}),
    ("zeros",                   {}),
]


def _weight(name: str, params: dict) -> np.ndarray:
    init = get_input_feedback(name, **params)
    w = torch.zeros(ROWS, COLS)
    init.initialize(w)
    return w.detach().cpu().numpy()


def grid_all_initializers() -> None:
    n = len(INITIALIZERS)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 2.0, rows * 2.6), constrained_layout=True,
    )
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax, (name, params) in zip(axes, INITIALIZERS):
        w = _weight(name, params)
        vmax = max(abs(w).max(), 1e-12)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        ax.imshow(w, cmap="RdBu_r", aspect="auto", norm=norm,
                  interpolation="nearest")
        ax.set_title(name, fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(RULE)
    for ax in axes[len(INITIALIZERS):]:
        ax.set_visible(False)
    fig.suptitle(
        f"Input/feedback initializer weight matrices "
        f"(reservoir = {ROWS}, feedback = {COLS})",
        fontsize=11, y=1.02,
    )
    save(fig, "initializers_grid.png")


def main() -> None:
    apply_style()
    print("Initializers …")
    grid_all_initializers()


if __name__ == "__main__":
    main()
