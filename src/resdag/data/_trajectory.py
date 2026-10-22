"""Lazy, physical-time adapters for the optional TSDynamics trajectory type.

Only preparing or rendering an actual trajectory imports TSDynamics. Contexts
contain plain JSON-compatible metadata; a live source system is carried
separately and must not be included in checkpoint extra state.
"""

from __future__ import annotations

import copy
import importlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

_METHODS = ("linear", "pchip", "cubic")
_CONTEXT_KEYS = {
    "version",
    "start_time",
    "last_time",
    "dt",
    "n_samples",
    "variables",
    "source",
    "sampling",
}


@dataclass(frozen=True)
class PreparedTrajectory:
    """Validated observed arrays, portable clock metadata and a transient system."""

    times: np.ndarray
    values: np.ndarray
    context: dict[str, Any]
    system: Any


def trajectory_type() -> type:
    """Resolve the optional public result type before starting model execution."""
    try:
        result = importlib.import_module("tsdynamics").Trajectory
    except ImportError as error:
        raise ImportError(
            "Trajectory input/output requires the optional resdag[dynamics] extra "
            "on Python >= 3.12"
        ) from error
    return cast(type, result)


def is_trajectory(value: Any) -> bool:
    """Recognize Trajectory subclasses without importing optional code for tensors."""
    if not any(
        cls.__name__ == "Trajectory" and cls.__module__.startswith("tsdynamics.")
        for cls in type(value).__mro__
    ):
        return False
    return isinstance(value, trajectory_type())


def validate_resampling_options(dt: Any, method: Any) -> tuple[float | None, str]:
    """Validate plain options independently of the optional dependency."""
    if not isinstance(method, str) or method not in _METHODS:
        raise ValueError(f"resample_method must be one of {_METHODS}")
    if dt is not None:
        if isinstance(dt, (bool, np.bool_)) or not isinstance(dt, (int, float, np.number)):
            raise ValueError("resample_dt must be a positive finite real number or None")
        if np.iscomplexobj(dt):
            raise ValueError("resample_dt must be a positive finite real number or None")
        try:
            dt = float(dt)
        except (ValueError, TypeError, OverflowError):
            raise ValueError("resample_dt must be a positive finite real number or None") from None
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("resample_dt must be a positive finite real number or None")
    return dt, str(method)


def validate_time_grid(times: Any, dt: float) -> None:
    """Require a representable uniform grid compatible with public Trajectory.dt.

    A large absolute clock can make mathematically uniform small steps uneven
    in float64. Validate the actual stored differences rather than only the
    endpoint formula. This needs no optional dependency.
    """
    interval, _ = validate_resampling_options(dt, "linear")
    if interval is None:
        raise ValueError("A uniform time grid requires a positive sampling interval")
    times = np.asarray(times)
    if times.ndim != 1 or not len(times) or times.dtype.kind not in "iuf":
        raise ValueError("Time grid must be a non-empty finite real vector")
    if not np.isfinite(times).all():
        raise ValueError("Time grid must be a non-empty finite real vector")
    steps = np.diff(np.asarray(times, dtype=np.float64))
    if not len(steps):
        return
    first = float(steps[0])
    if (
        first <= 0
        or not np.isfinite(steps).all()
        or np.max(np.abs(steps - first)) > 1e-9 * abs(first)
        or np.max(np.abs(steps - interval)) > 1e-9 * abs(interval)
    ):
        raise ValueError(
            "The time grid cannot represent a uniform sampling interval at this time origin; "
            "shift the time origin closer to zero or use a coarser sampling interval"
        )


def _safe_source(value: Any, active: frozenset[int] = frozenset()) -> Any:
    """Snapshot source provenance without making arbitrary objects checkpointable."""
    if isinstance(value, str):
        return str(value)
    if value is None or type(value) in (bool, int):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else str(number)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    identity = id(value)
    if identity in active or len(active) >= 20:
        return {"omitted_type": type(value).__name__}
    active = active | {identity}
    if isinstance(value, np.ndarray):
        return _safe_source(value.tolist(), active)
    if isinstance(value, Mapping):
        return {str(key): _safe_source(item, active) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe_source(item, active) for item in value]
    return {"omitted_type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _json_value(value: Any) -> bool:
    if value is None or type(value) in (str, bool, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(_json_value(item) for item in value)
    if type(value) is dict:
        return all(type(key) is str and _json_value(item) for key, item in value.items())
    return False


def _signal_names(value: Any, width: int) -> list[str]:
    if not isinstance(value, (tuple, list)) or len(value) != width:
        raise ValueError("Trajectory variables must name every signal")
    if any(not isinstance(name, str) or not name.strip() for name in value):
        raise ValueError("Trajectory variables must be non-empty strings")
    if len(set(value)) != len(value):
        raise ValueError("Trajectory variables must be distinct")
    return [str(name) for name in value]


def validate_trajectory_context(context: Any) -> dict[str, Any]:
    """Validate a restored portable context before it can define forecast times."""
    if type(context) is not dict or set(context) != _CONTEXT_KEYS:
        raise ValueError("Invalid trajectory context fields")
    try:
        safe = _json_value(context)
    except RecursionError:
        safe = False
    if not safe or type(context["version"]) is not int or context["version"] != 1:
        raise ValueError("Trajectory context must contain finite JSON-safe metadata, version 1")
    count = context["n_samples"]
    if type(count) is not int or count < 2:
        raise ValueError("Trajectory context n_samples must be an integer >= 2")
    for key in ("start_time", "last_time", "dt"):
        if type(context[key]) not in (int, float):
            raise ValueError(f"Trajectory context {key} must be a finite real number")
    start, last, dt = context["start_time"], context["last_time"], context["dt"]
    if dt <= 0 or last <= start:
        raise ValueError("Trajectory context must describe a positive sampling interval")
    span = last - start
    try:
        expected_span = dt * (count - 1)
    except OverflowError:
        raise ValueError("Trajectory context sampling count is not representable") from None
    if not math.isfinite(span) or not math.isclose(span, expected_span, rel_tol=1e-9):
        raise ValueError("Trajectory context times/count do not match its sampling interval")
    variables = context["variables"]
    _signal_names(variables, len(variables) if isinstance(variables, list) else 0)
    if (
        not variables
        or type(context["source"]) is not dict
        or type(context["sampling"]) is not dict
    ):
        raise ValueError("Trajectory context requires signals, source and sampling metadata")
    sampling = context["sampling"]
    required = {
        "requested_dt",
        "effective_dt",
        "method",
        "resampled",
        "original_n_samples",
        "original_start_time",
        "original_last_time",
    }
    if set(sampling) != required:
        raise ValueError("Invalid trajectory sampling metadata fields")
    validate_resampling_options(sampling["requested_dt"], sampling["method"])
    if (
        type(sampling["resampled"]) is not bool
        or type(sampling["original_n_samples"]) is not int
        or sampling["original_n_samples"] < 2
        or type(sampling["effective_dt"]) not in (int, float)
        or sampling["effective_dt"] != dt
        or any(
            type(sampling[key]) not in (int, float)
            for key in ("original_start_time", "original_last_time")
        )
        or sampling["original_start_time"] > start
        or sampling["original_last_time"] != last
    ):
        raise ValueError("Inconsistent trajectory sampling metadata")
    return copy.deepcopy(context)


def _observations(value: Any) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not is_trajectory(value):
        raise TypeError("Expected a tsdynamics.Trajectory")
    times, values = np.asarray(value.t), np.asarray(value.y)
    if times.ndim != 1 or len(times) < 2 or times.dtype.kind not in "iuf":
        raise ValueError("Trajectory needs at least two real sample times")
    if values.ndim != 2 or values.shape[0] != len(times) or values.shape[1] == 0:
        raise ValueError("Trajectory values must have shape (time, signals)")
    if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
        raise ValueError("Trajectory values must be finite real numbers")
    times = np.asarray(times, dtype=np.float64)
    if not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError("Trajectory sample times must be finite and strictly increasing")
    return times, values, _signal_names(value.variables, values.shape[1])


def prepare_trajectory(value: Any, *, dt: Any = None, method: str = "linear") -> PreparedTrajectory:
    """Read one run and automatically regularize its observed physical-time grid.

    Unspecified dt preserves sample count and endpoints. An explicit dt anchors
    at the last observation and keeps only samples within the observed span.
    Interpolation never extrapolates and is unnecessary for an existing grid.
    """
    requested_dt, method = validate_resampling_options(dt, method)
    times, values, variables = _observations(value)
    start, last = float(times[0]), float(times[-1])
    span = last - start
    if not math.isfinite(span):
        raise ValueError("Trajectory time span is not representable")
    dt = span / (len(times) - 1) if requested_dt is None else requested_dt
    if dt <= 0 or not math.isfinite(dt):
        raise ValueError("Trajectory sampling interval is not representable")
    if requested_dt is None:
        grid = np.linspace(start, last, len(times), dtype=np.float64)
    else:
        intervals = span / dt
        if not math.isfinite(intervals) or intervals >= np.iinfo(np.intp).max:
            raise ValueError("resample_dt requests too many samples")
        count = math.floor(float(np.nextafter(intervals, math.inf))) + 1
        grid = last - dt * np.arange(count - 1, -1, -1, dtype=np.float64)
        if len(grid) and grid[0] < start:
            tolerance = 8 * np.finfo(np.float64).eps * max(abs(start), abs(last), span)
            if start - grid[0] <= tolerance:
                grid[0] = start
            else:
                grid = grid[1:]
    if len(grid) < 2 or not np.all(np.diff(grid) > 0):
        raise ValueError("resample_dt must leave at least two distinct observed samples")
    regular = len(grid) == len(times) and np.allclose(np.diff(times), dt, rtol=1e-9, atol=0)
    if regular:
        grid = times
        prepared_values = values
    elif method == "linear":
        prepared_values = np.stack(
            [np.interp(grid, times, values[:, column]) for column in range(values.shape[1])], axis=1
        )
    else:
        from scipy.interpolate import CubicSpline, PchipInterpolator

        interpolator = PchipInterpolator if method == "pchip" else CubicSpline
        prepared_values = interpolator(times, values, axis=0, extrapolate=False)(grid)
    validate_time_grid(grid, dt)
    if not np.isfinite(prepared_values).all():
        raise ValueError("Trajectory interpolation produced nonfinite values")
    system = value.system
    context = {
        "version": 1,
        "start_time": float(grid[0]),
        "last_time": last,
        "dt": float(dt),
        "n_samples": len(grid),
        "variables": variables,
        "source": {
            "metadata": _safe_source(value.meta),
            "system": (
                None
                if system is None
                else {
                    "name": type(system).__name__,
                    "module": type(system).__module__,
                }
            ),
        },
        "sampling": {
            "requested_dt": requested_dt,
            "effective_dt": float(dt),
            "method": method,
            "resampled": not regular,
            "original_n_samples": len(times),
            "original_start_time": start,
            "original_last_time": last,
        },
    }
    return PreparedTrajectory(grid, prepared_values, validate_trajectory_context(context), system)


def forecast_trajectory(values: Any, context: Any, *, system: Any = None, times: Any = None) -> Any:
    """Render original-unit predictions using the public Trajectory constructor."""
    context = validate_trajectory_context(context)
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != len(context["variables"]) or len(values) == 0:
        raise ValueError("Forecast values must have shape (time, trajectory signals)")
    if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
        raise ValueError("Forecast values must be finite real numbers")
    if times is None:
        times = context["last_time"] + context["dt"] * np.arange(1, len(values) + 1)
        checked_times = np.concatenate(([context["last_time"]], times))
    else:
        times = np.asarray(times)
        checked_times = times
    if (
        times.ndim != 1
        or len(times) != len(values)
        or times.dtype.kind not in "iuf"
        or not np.isfinite(times).all()
        or not np.all(np.diff(times) > 0)
    ):
        raise ValueError("Forecast times must be finite, increasing and match forecast rows")
    validate_time_grid(checked_times, context["dt"])
    metadata = {
        "dt": context["dt"],
        "t0": float(times[0]),
        "ic": values[0].copy(),
        "variables": list(context["variables"]),
        "source": context["source"],
        "resdag": {
            "kind": "forecast",
            "history_last_time": context["last_time"],
            "sampling": context["sampling"],
        },
    }
    source_metadata = context["source"].get("metadata", {})
    if isinstance(source_metadata, dict) and "field_shape" in source_metadata:
        metadata["field_shape"] = copy.deepcopy(source_metadata["field_shape"])
    return trajectory_type()(
        times, values, system=system, meta=metadata, variables=context["variables"]
    )
