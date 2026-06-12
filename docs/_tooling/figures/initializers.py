"""Per-initializer portraits: the matrix each initializer draws."""

import numpy as np
import torch

from _style import DIVERGING, MUTED, RULE, plt, save


def main() -> None:
    from resdag.init.input_feedback import get_input_feedback
    from resdag.init.input_feedback.registry import _INPUT_FEEDBACK_REGISTRY

    rows, cols = 80, 8
    EXTRA = {"chain_of_neurons_input": {"features": cols},
             "dendrocycle_input": {"C": 8},
             "ring_window": {"c": 0.25, "window": 10}}
    for name in sorted(_INPUT_FEEDBACK_REGISTRY):
        w = torch.empty(rows, cols)
        try:
            init = get_input_feedback(name, **EXTRA.get(name, {}))
            torch.manual_seed(3)
            init.initialize(w)
        except Exception as e:
            print(f"  skip {name}: {e}")
            continue
        A = w.numpy()
        fig, ax = plt.subplots(figsize=(3.4, 3.2))
        vmax = np.abs(A).max() or 1
        im = ax.imshow(A, cmap=DIVERGING, vmin=-vmax, vmax=vmax,
                       interpolation="nearest", aspect="auto")
        ax.set_xlabel("input dim"); ax.set_ylabel("reservoir unit")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(RULE)
        cb = fig.colorbar(im, ax=ax, fraction=0.08, pad=0.04)
        cb.outline.set_edgecolor(RULE)
        cb.ax.tick_params(labelsize=7, color=MUTED)
        save(fig, f"initializers/{name}.png")


if __name__ == "__main__":
    main()
