"""Draw the full fixed Lorenz continuation and forecast, in original units."""

from _style import ACCENT, FIG, PAPER, TRUE
from forecast_data import load_showcase, trajectories


def main() -> None:
    import tsdynamics as ts

    truth, forecast = trajectories(load_showcase()["cases"]["lorenz"])
    plot = ts.plot(
        ts.plot(truth, color=TRUE, title="Observed continuation"),
        ts.plot(forecast, color=ACCENT, title="Autonomous forecast"),
        layout="row",
    )
    plot.background(PAPER).camera(elev=18, azim=-60)
    destination = FIG / "lorenz_phase.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    plot.save(destination, size=(1032, 480), dpi=120)
    print(f"wrote {destination.name}")


if __name__ == "__main__":
    main()
