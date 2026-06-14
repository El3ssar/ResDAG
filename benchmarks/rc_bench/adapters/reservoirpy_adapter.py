"""reservoirpy adapter (CPU).

``Reservoir(units, sr, lr, input_scaling, rc_connectivity) >> Ridge(ridge)``.
reservoirpy uses NumPy/SciPy (sparse reservoir) and a direct ridge solve.
"""

from __future__ import annotations

import os

# Silence reservoirpy's tqdm progress bars before it is imported.
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np

from ..config import HParams, Point
from ..timing import RunResult, repeat_timed
from .base import Adapter


class ReservoirPyAdapter(Adapter):
    key = "reservoirpy"
    label = "reservoirpy"
    family = "reservoirpy"
    device = "cpu"

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        try:
            import reservoirpy  # noqa: F401
        except Exception as e:  # pragma: no cover
            return False, f"import failed: {e}"
        return True, ""

    def _build(self, hp: HParams, point: Point):
        import reservoirpy
        from reservoirpy.nodes import Reservoir, Ridge

        reservoirpy.set_seed(hp.seed)
        reservoir = Reservoir(
            units=point.reservoir_size,
            sr=hp.spectral_radius,
            lr=hp.leak_rate,
            input_scaling=hp.input_scaling,
            rc_connectivity=hp.connectivity,
            seed=hp.seed,
        )
        readout = Ridge(ridge=hp.ridge)
        esn = reservoir >> readout
        return esn, reservoir

    def time_train(self, series, hp, point, repeats, warmups) -> RunResult:
        x, y = self._train_window(series, point)

        def setup():
            return self._build(hp, point)

        def op(ctx):
            esn, _ = ctx
            esn.fit(x, y, warmup=point.warmup)
            return None

        times, _ = repeat_timed(setup, op, repeats, warmups)
        return RunResult(times=times, info={"device": "cpu", "dtype": "float64"})

    def time_forecast(self, series, hp, point, repeats, warmups) -> RunResult:
        x, y = self._train_window(series, point)
        esn, reservoir = self._build(hp, point)
        esn.fit(x, y, warmup=point.warmup)
        warm = np.ascontiguousarray(series[: point.warmup])
        horizon = point.horizon
        dim = hp.dim

        def setup():
            reservoir.reset()
            return esn

        def op(model):
            warming = model.run(warm)
            yv = warming[-1]
            gen = np.empty((horizon, dim), dtype=np.float64)
            for t in range(horizon):
                yv = model(yv)
                gen[t] = yv
            return gen

        times, last = repeat_timed(setup, op, repeats, warmups)

        rmse = None
        if last is not None:
            truth = self._forecast_truth(series, point)
            n = min(len(truth), len(last), 200)
            if n > 0:
                rmse = float(np.sqrt(np.mean((last[:n] - truth[:n]) ** 2)))
        return RunResult(times=times, rmse=rmse, info={"device": "cpu", "dtype": "float64"})

    def predict_trajectory(self, series, hp, point) -> np.ndarray:
        x, y = self._train_window(series, point)
        esn, reservoir = self._build(hp, point)
        esn.fit(x, y, warmup=point.warmup)
        reservoir.reset()
        warming = esn.run(np.ascontiguousarray(series[: point.warmup]))
        yv = warming[-1]
        gen = np.empty((point.horizon, hp.dim), dtype=np.float64)
        for t in range(point.horizon):
            yv = esn(yv)
            gen[t] = yv
        return gen
