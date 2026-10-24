"""Plot every recorded sample of the fixed Lorenz documentation forecast."""

import numpy as np
from _style import ACCENT, MUTED, TRUE, plt, save
from forecast_data import load_showcase, valid_steps


def main() -> None:
    case = load_showcase()["cases"]["lorenz"]
    truth, prediction = np.asarray(case["truth"]), np.asarray(case["prediction"])
    time = np.arange(1, len(truth) + 1) * case["dt"] * case["lambda_max"]
    steps = valid_steps(case)
    horizon = steps * case["dt"] * case["lambda_max"]
    censored = steps == len(truth)
    fig, axes = plt.subplots(3, 1, figsize=(8.6, 4.6), sharex=True)
    for component, (ax, label) in enumerate(zip(axes, case["variables"], strict=True)):
        ax.plot(time, truth[:, component], color=TRUE, lw=1.3, label="observed continuation")
        ax.plot(time, prediction[:, component], color=ACCENT, lw=1.1, label="autonomous forecast")
        if not censored:
            ax.axvline(horizon, color=MUTED, lw=0.8, ls=(0, (4, 3)))
        ax.set_ylabel(label, rotation=0, va="center")
        ax.margins(x=0)
    axes[0].legend(ncols=2, loc="lower left", bbox_to_anchor=(0, 1.02), fontsize=8)
    sign = "≥" if censored else ""
    fig.text(
        0.13,
        0.98,
        f"0.3-error valid prefix: {sign}{horizon:.2f} Lyapunov times",
        fontsize=8,
        color=MUTED,
    )
    axes[-1].set_xlabel("Forecast time in Lyapunov units (λ₁ Δt)")
    fig.align_ylabels(axes)
    fig.subplots_adjust(hspace=0.18, top=0.88)
    save(fig, "first_forecast.png")


if __name__ == "__main__":
    main()
