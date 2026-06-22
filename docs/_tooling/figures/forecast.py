"""First-forecast figure: three Lorenz components, system vs forecast."""

import numpy as np
from _style import ACCENT, MUTED, TRUE, plt, save, train_best_lorenz


def main() -> None:
    vh, pred, val = train_best_lorenz()
    lyap = 0.906  # largest Lyapunov exponent of Lorenz-63
    t = np.arange(val.shape[1]) * 0.01 * lyap
    show = slice(0, 1800)

    fig, axes = plt.subplots(3, 1, figsize=(8.6, 4.6), sharex=True)
    for i, (ax, comp) in enumerate(zip(axes, "xyz")):
        ax.plot(t[show], val[0, show, i], color=TRUE, lw=1.3, label="system")
        ax.plot(t[show], pred[0, show, i], color=ACCENT, lw=1.3, label="forecast")
        ax.axvline(vh * 0.01 * lyap, color=MUTED, lw=0.8, ls=(0, (4, 3)))
        ax.set_ylabel(comp, rotation=0, va="center", style="italic")
        ax.margins(x=0)
    axes[0].legend(ncols=2, loc="upper right", fontsize=8)
    axes[0].text(
        0.012,
        0.06,
        f"valid horizon ≈ {vh * 0.01 * lyap:.1f} Lyapunov times",
        transform=axes[0].transAxes,
        fontsize=8,
        color=MUTED,
    )
    axes[-1].set_xlabel("Lyapunov times  (λ₁ t)")
    fig.align_ylabels(axes)
    fig.subplots_adjust(hspace=0.18)
    save(fig, "first_forecast.png")


if __name__ == "__main__":
    main()
