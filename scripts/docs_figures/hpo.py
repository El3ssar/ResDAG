"""Synthetic HPO illustration (no actual Optuna run)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ._common import AMBER, apply_style, save


def hpo_scatter() -> None:
    rng = np.random.default_rng(42)
    n = 200
    spectral = rng.uniform(0.5, 1.4, size=n)
    leak = rng.uniform(0.05, 1.0, size=n)
    optimum_s, optimum_l = 0.95, 0.4
    score = (
        -3.0 * (spectral - optimum_s) ** 2
        - 4.5 * (leak - optimum_l) ** 2
        + rng.normal(0, 0.18, size=n)
    )

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    sc = ax.scatter(spectral, leak, c=score, cmap="viridis", s=30,
                    edgecolor="white", linewidths=0.5)
    best = int(np.argmax(score))
    ax.scatter([spectral[best]], [leak[best]], s=110, marker="*",
               color=AMBER, edgecolor="white", linewidths=0.9,
               zorder=5, label="best trial")
    ax.set_xlabel("spectral_radius")
    ax.set_ylabel("leak_rate")
    ax.set_title("HPO trials (synthetic) — colour = −loss", loc="left")
    ax.grid(True, alpha=0.6)
    ax.legend(loc="upper right")
    fig.colorbar(sc, ax=ax, label="score").outline.set_linewidth(0)
    save(fig, "hpo_scatter.png")


def main() -> None:
    apply_style()
    print("HPO scatter …")
    hpo_scatter()


if __name__ == "__main__":
    main()
