"""Per-initializer portraits: the matrix each initializer draws."""

import hashlib
import inspect
import json

import numpy as np
import torch
from _style import DIVERGING, MUTED, RULE, plt, save


def main() -> None:
    from resdag.init.input_feedback import get_input_feedback
    from resdag.init.input_feedback.registry import _INPUT_FEEDBACK_REGISTRY

    rows, cols = 80, 8
    extras = {
        "chain_of_neurons_input": {"features": cols},
        "dendrocycle_input": {"C": 8},
        "ring_window": {"c": 0.25, "window": 10},
        "informed_init": {"model_in_size": 2},
    }
    for name in sorted(_INPUT_FEEDBACK_REGISTRY):
        builder = _INPUT_FEEDBACK_REGISTRY[name][0]
        target = builder.__init__ if inspect.isclass(builder) else builder
        options = dict(extras.get(name, {}))
        if "seed" in inspect.signature(target).parameters:
            options["seed"] = 3
        w = torch.empty(rows, cols, dtype=torch.float64)
        init = get_input_feedback(name, **options)
        init.initialize(w)
        matrix = w.numpy()
        fig, ax = plt.subplots(figsize=(3.4, 3.2))
        vmax = np.abs(matrix).max() or 1
        im = ax.imshow(
            matrix, cmap=DIVERGING, vmin=-vmax, vmax=vmax, interpolation="nearest", aspect="auto"
        )
        ax.set_xlabel("input dim")
        ax.set_ylabel("reservoir unit")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(RULE)
        cb = fig.colorbar(im, ax=ax, fraction=0.08, pad=0.04)
        cb.outline.set_edgecolor(RULE)
        cb.ax.tick_params(labelsize=7, color=MUTED)
        save(
            fig,
            f"initializers/{name}.png",
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
