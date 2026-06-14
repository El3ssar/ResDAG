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
* forecasting teacher-forces ``series[0:warmup]`` then generates ``horizon``
  steps closed-loop; ground truth for the RMSE sanity metric is
  ``series[warmup:warmup+horizon]``.
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
        x = series[: point.train_len]
        y = series[1 : point.train_len + 1]
        return np.ascontiguousarray(x), np.ascontiguousarray(y)

    @staticmethod
    def _forecast_truth(series: np.ndarray, point: Point) -> np.ndarray:
        return series[point.warmup : point.warmup + point.horizon]
