"""Adapter interface.

Every benchmarked library implements an :class:`Adapter`. The harness hands it
the *same* NumPy series and the same :class:`~rc_bench.config.HParams` /
:class:`~rc_bench.config.Point`; the adapter maps those onto its library's API
and returns timed :class:`~rc_bench.timing.RunResult` objects.

Data convention (shared by all adapters)
----------------------------------------
One-step-ahead forecasting on a ``(T, dim)`` series:

* training uses ``X = series[0:train_len]``, ``Y = series[1:train_len+1]`` with
  the first ``warmup`` steps washed out of the readout fit;
* the final observed training target is ``series[train_len]``;
* forecasting resets recurrent state and synchronizes on the ``warmup`` known
  samples ending at that observation, then generates ``horizon`` closed-loop
  predictions; the first prediction is for ``series[train_len + 1]``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..config import HParams, Point
from ..timing import RunResult


class Adapter(ABC):
    key: str = "base"  # stable id used in result JSON, e.g. "resdag-gpu"
    label: str = "base"  # human label for tables
    family: str = "base"  # library family, e.g. "resdag", "reservoirpy", "rcjl"
    device: str = "cpu"

    @classmethod
    @abstractmethod
    def is_available(cls) -> tuple[bool, str]:
        """Return ``(available, reason)``. ``reason`` explains a ``False``."""

    @abstractmethod
    def time_train(
        self, series: np.ndarray, hp: HParams, point: Point, repeats: int, warmups: int
    ) -> RunResult:
        """Time fitting the readout (build the model fresh each repeat)."""

    @abstractmethod
    def time_forecast(
        self, series: np.ndarray, hp: HParams, point: Point, repeats: int, warmups: int
    ) -> RunResult:
        """Time autoregressive generation (train once, untimed; time generation)."""

    @abstractmethod
    def predict_trajectory(self, series: np.ndarray, hp: HParams, point: Point) -> np.ndarray:
        """Train, then return the closed-loop forecast as a ``(horizon, dim)`` array
        (untimed — used to score predictive accuracy / valid prediction time)."""

    # -- shared helpers -----------------------------------------------------
    @staticmethod
    def _train_window(series: np.ndarray, point: Point) -> tuple[np.ndarray, np.ndarray]:
        Adapter._validate_series(series, point.train_len + 1)
        x = series[: point.train_len]
        y = series[1 : point.train_len + 1]
        return np.ascontiguousarray(x), np.ascontiguousarray(y)

    @staticmethod
    def _validate_series(series: np.ndarray, length: int) -> None:
        if series.ndim != 2 or series.shape[1] < 1 or len(series) < length:
            raise ValueError(f"series must have shape (time, features) with at least {length} rows")
        if not np.isfinite(series).all():
            raise ValueError("series must contain only finite observations")

    @staticmethod
    def _forecast_history(series: np.ndarray, point: Point) -> np.ndarray:
        Adapter._validate_series(series, point.forecast_start)
        return np.ascontiguousarray(series[point.history_start : point.forecast_start])

    @staticmethod
    def _forecast_truth(series: np.ndarray, point: Point) -> np.ndarray:
        Adapter._validate_series(series, point.series_len)
        return np.ascontiguousarray(series[point.forecast_start : point.series_len])
