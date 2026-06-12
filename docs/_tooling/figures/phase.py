"""Lorenz phase portraits: the system and the autonomous forecast."""

from _style import ACCENT, PAPER, TRUE, plt, save, train_best_lorenz


def main() -> None:
    vh, pred, val = train_best_lorenz(horizon=4000)
    fig = plt.figure(figsize=(8.6, 4.0))
    for k, (title, dat, color) in enumerate([
            ("system", val[0].numpy(), TRUE),
            ("forecast (autonomous)", pred[0].numpy(), ACCENT)]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.plot(dat[:, 0], dat[:, 1], dat[:, 2], color=color, lw=0.45, alpha=0.85)
        ax.set_title(title, pad=0)
        ax.set_axis_off()
        ax.view_init(elev=12, azim=-60)
        ax.set_facecolor(PAPER)
    fig.subplots_adjust(wspace=0.02, left=0.0, right=1.0, top=0.92, bottom=0.02)
    save(fig, "lorenz_phase.png")


if __name__ == "__main__":
    main()
