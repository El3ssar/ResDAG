"""Render fixed forecast recordings through TSDynamics' public plotting API.

No fitting, ODE integration, seed selection, axis clipping or path truncation is
performed. The JSON source records the fixed display cases and their study.
MP4 requires FFmpeg; both posters and movies use TSDynamics' Matplotlib backend.

    python docs/_tooling/figures/hero_forecasts.py
"""

from pathlib import Path

from _style import ACCENT, FIG, PAPER, TRUE
from forecast_data import load_showcase, trajectories


def forecast_plot(case: dict):
    import tsdynamics as ts

    truth, forecast = trajectories(case)
    plot = ts.plot(
        ts.plot(truth, color=TRUE),
        ts.plot(forecast, color=ACCENT),
        labels=["Observed continuation", "Autonomous forecast"],
    )
    plot.background(PAPER).camera(elev=18, azim=-60)
    return plot


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for name, case in load_showcase()["cases"].items():
        plot = forecast_plot(case)
        poster = FIG / f"{name}_forecast_poster.png"
        plot.save(poster, size=(960, 640), dpi=120)
        movie = FIG / f"{name}_forecast.mp4"
        plot.animate(duration=12, fps=24).trail(None)
        plot.save(movie, size=(960, 640), dpi=120)
        print(f"wrote {Path(poster).name} and {Path(movie).name}")


if __name__ == "__main__":
    main()
