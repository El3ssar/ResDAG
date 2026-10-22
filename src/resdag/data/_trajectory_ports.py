"""Shared physical clocks for named optional trajectory ports.

All interpolation happens at the explicit trajectory boundary. Tensor-only
execution never needs this module to import TSDynamics or move data to NumPy.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._trajectory import (
    _json_value,
    _safe_source,
    _signal_names,
    is_trajectory,
    trajectory_type,
    validate_resampling_options,
    validate_time_grid,
)


@dataclass(frozen=True)
class PreparedTrajectoryPorts:
    """Full observed inputs and a separate contiguous supervision window.

    ``targets`` contains rows for origins ``supervised_start:supervised_stop``.
    Configured washout is not applied here. Every input retains the entire
    shared observed grid, including rows after the last supervised origin.
    """

    times: np.ndarray
    inputs: tuple[np.ndarray, ...]
    targets: tuple[np.ndarray, ...]
    supervised_start: int
    supervised_stop: int
    context: dict[str, Any]
    input_systems: tuple[Any, ...]
    output_systems: tuple[Any, ...]


@dataclass(frozen=True)
class _Port:
    times: np.ndarray
    values: np.ndarray
    variables: list[str]
    source: dict[str, Any]
    system: Any
    held_out: int


def has_trajectory_port(value: Any) -> bool:
    """Recognize clocked mappings without importing optional code for tensors."""
    return isinstance(value, Mapping) and any(is_trajectory(item) for item in value.values())


def _named_values(value: Any, names: tuple[str, ...], label: str) -> tuple[Any, ...]:
    if is_trajectory(value) and len(names) == 1:
        return (value,)
    if not isinstance(value, Mapping) or set(value) != set(names):
        raise ValueError(f"{label} requires named trajectory keys {list(names)}")
    values = tuple(value[name] for name in names)
    if not all(is_trajectory(item) for item in values):
        raise TypeError(
            f"{label} must contain only Trajectories; mixed Tensor/Trajectory ports "
            "have no inferred common clock"
        )
    return values


def _read_port(value: Any, width: int, label: str, *, cutoff: float | None = None) -> _Port:
    if not is_trajectory(value):
        raise TypeError(f"{label} must be a TSDynamics Trajectory")
    times, data = np.asarray(value.t), np.asarray(value.y)
    if times.ndim != 1 or times.dtype.kind not in "iuf":
        raise ValueError(f"{label} times must be a finite increasing real vector")
    if data.ndim != 2 or data.shape != (len(times), width) or data.dtype.kind not in "iuf":
        raise ValueError(f"{label} values must have shape (time, {width}) and real numeric dtype")
    variables = _signal_names(value.variables, width)
    times = np.asarray(times, dtype=np.float64)
    if not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError(f"{label} times must be finite and strictly increasing")
    # Held-out target values must not enter PCHIP/cubic coefficients, nor even
    # invalidate training values that precede the known input endpoint.
    count = len(times) if cutoff is None else int(np.searchsorted(times, cutoff, side="right"))
    held_out = len(times) - count
    times, data = times[:count], data[:count]
    if not np.isfinite(data).all():
        raise ValueError(f"{label} observed values must be finite")
    system = value.system
    source = {
        "metadata": _safe_source(value.meta),
        "system": (
            None
            if system is None
            else {"name": type(system).__name__, "module": type(system).__module__}
        ),
    }
    return _Port(times, data, variables, source, system, held_out)


def _shared_grid(ports: tuple[_Port, ...], dt: float | None) -> tuple[np.ndarray, float]:
    if not ports or any(not len(port.times) for port in ports):
        raise ValueError("Every input trajectory needs at least one observed sample")
    start = max(float(port.times[0]) for port in ports)
    last = min(float(port.times[-1]) for port in ports)
    if start > last:
        raise ValueError("Input trajectories have no shared observed time range")
    if dt is None:
        first = ports[0].times
        if len(first) < 2:
            raise ValueError("A singleton input history requires a fitted or explicit resample_dt")
        dt = float((first[-1] - first[0]) / (len(first) - 1))
    interval, _ = validate_resampling_options(dt, "linear")
    assert interval is not None
    span = last - start
    intervals = span / interval
    if not math.isfinite(span) or not math.isfinite(intervals):
        raise ValueError("The shared trajectory time range is not representable")
    if intervals >= np.iinfo(np.intp).max:
        raise ValueError("resample_dt requests too many shared samples")
    count = math.floor(float(np.nextafter(intervals, math.inf))) + 1
    grid = last - interval * np.arange(count - 1, -1, -1, dtype=np.float64)
    if len(grid) and grid[0] < start:
        tolerance = 8 * np.finfo(np.float64).eps * max(abs(start), abs(last), span)
        if start - grid[0] <= tolerance:
            grid[0] = start
        else:
            grid = grid[1:]
    validate_time_grid(grid, interval)
    return grid, interval


def _sample(port: _Port, queries: np.ndarray, dt: float, method: str, label: str) -> np.ndarray:
    if not len(queries):
        return np.empty((0, port.values.shape[1]), dtype=port.values.dtype)
    if not len(port.times):
        raise ValueError(f"{label} does not cover the required clock samples")
    tolerance = 1e-9 * dt
    if np.any(queries < port.times[0] - tolerance) or np.any(queries > port.times[-1] + tolerance):
        raise ValueError(
            f"{label} does not cover the required clock samples; extrapolation is disabled"
        )
    # Clamp only floating-point endpoint roundoff, never a missing graph tick.
    queries = np.clip(queries, port.times[0], port.times[-1])
    if len(port.times) == len(queries) and np.array_equal(port.times, queries):
        values = port.values.copy()
    elif len(port.times) == 1:
        values = np.repeat(port.values, len(queries), axis=0)
    elif method == "linear":
        values = np.stack(
            [
                np.interp(queries, port.times, port.values[:, i])
                for i in range(port.values.shape[1])
            ],
            axis=1,
        )
    else:
        from scipy.interpolate import CubicSpline, PchipInterpolator

        interpolator = PchipInterpolator if method == "pchip" else CubicSpline
        values = interpolator(port.times, port.values, axis=0, extrapolate=False)(queries)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} interpolation produced nonfinite values")
    return np.asarray(values)


def _record(port: _Port, queries: np.ndarray) -> dict[str, Any]:
    return {
        "variables": list(port.variables),
        "source": copy.deepcopy(port.source),
        "observation": {
            "start_time": float(port.times[0]),
            "last_time": float(port.times[-1]),
            "n_samples": len(port.times),
        },
        "resampled": not np.array_equal(port.times, queries),
        "held_out_samples": port.held_out,
    }


def _context(
    grid: np.ndarray,
    interval: float,
    requested_dt: float | None,
    method: str,
    input_names: tuple[str, ...],
    inputs: tuple[_Port, ...],
    outputs: dict[str, Any],
    supervision: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "version": 2,
        "clock": {
            "start_time": float(grid[0]),
            "last_time": float(grid[-1]),
            "dt": interval,
            "n_samples": len(grid),
            "reference_input": input_names[0],
            "requested_dt": requested_dt,
            "method": method,
        },
        "inputs": {name: _record(port, grid) for name, port in zip(input_names, inputs)},
        "outputs": outputs,
        "supervision": supervision,
    }


def prepare_trajectory_ports(
    training: Any,
    *,
    input_names: tuple[str, ...],
    input_widths: tuple[int, ...],
    output_names: tuple[str, ...],
    output_widths: tuple[int, ...],
    prediction_leads: tuple[int, ...],
    targets: Any = None,
    dt: float | None = None,
    method: str = "linear",
) -> PreparedTrajectoryPorts:
    """Align named inputs once and query physical target clocks without leakage."""
    requested_dt, method = validate_resampling_options(dt, method)
    items = _named_values(training, input_names, "training")
    inputs = tuple(
        _read_port(item, width, f"training[{name!r}]")
        for item, name, width in zip(items, input_names, input_widths)
    )
    grid, interval = _shared_grid(inputs, requested_dt)
    values = tuple(
        _sample(port, grid, interval, method, f"training[{name!r}]")
        for name, port in zip(input_names, inputs)
    )
    if targets is None:
        if len(input_names) != 1 or len(output_names) != 1 or input_widths != output_widths:
            raise ValueError(
                "Named multiport trajectory fitting requires explicit per-output targets"
            )
        target_items = items
    else:
        target_items = _named_values(targets, output_names, "targets")
    target_ports = tuple(
        _read_port(item, width, f"targets[{name!r}]", cutoff=float(grid[-1]))
        for item, name, width in zip(target_items, output_names, output_widths)
    )
    if len(prediction_leads) != len(output_names) or any(
        type(lead) is not int or lead < 1 for lead in prediction_leads
    ):
        raise ValueError("prediction_leads must contain one positive integer per output")
    support = np.ones(len(grid), dtype=bool)
    tolerance = 1e-9 * interval
    for port, lead in zip(target_ports, prediction_leads):
        valid = np.zeros(len(grid), dtype=bool)
        if len(port.times) and lead < len(grid):
            label_times = grid[lead:]
            valid[:-lead] = (label_times >= port.times[0] - tolerance) & (
                label_times <= port.times[-1] + tolerance
            )
        support &= valid
    indices = np.flatnonzero(support)
    if not len(indices):
        raise ValueError("Trajectory targets leave no jointly supported supervised origins")
    start, stop = int(indices[0]), int(indices[-1] + 1)
    target_values = tuple(
        _sample(port, grid[start + lead : stop + lead], interval, method, f"targets[{name!r}]")
        for port, lead, name in zip(target_ports, prediction_leads, output_names)
    )
    output_records = {
        name: _record(port, grid[start + lead : stop + lead])
        for port, lead, name in zip(target_ports, prediction_leads, output_names)
    }
    context = _context(
        grid,
        interval,
        requested_dt,
        method,
        input_names,
        inputs,
        output_records,
        {
            "start": start,
            "stop": stop,
            "prediction_leads": dict(zip(output_names, prediction_leads)),
        },
    )
    context = validate_trajectory_ports_context(
        context, input_names, input_widths, output_names, output_widths, prediction_leads
    )
    return PreparedTrajectoryPorts(
        grid,
        values,
        target_values,
        start,
        stop,
        context,
        tuple(port.system for port in inputs),
        tuple(port.system for port in target_ports),
    )


def validate_trajectory_ports_context(
    context: Any,
    input_names: tuple[str, ...],
    input_widths: tuple[int, ...],
    output_names: tuple[str, ...],
    output_widths: tuple[int, ...],
    prediction_leads: tuple[int, ...] | None = None,
    *,
    require_outputs: bool = True,
) -> dict[str, Any]:
    """Validate the current named context before it defines any physical clock."""
    if type(context) is not dict or set(context) != {
        "version",
        "clock",
        "inputs",
        "outputs",
        "supervision",
    }:
        raise ValueError("Invalid named trajectory context fields")
    try:
        safe = _json_value(context)
    except RecursionError:
        safe = False
    if not safe or type(context["version"]) is not int or context["version"] != 2:
        raise ValueError("Named trajectory context requires finite JSON-safe metadata, version 2")
    clock = context["clock"]
    if type(clock) is not dict or set(clock) != {
        "start_time",
        "last_time",
        "dt",
        "n_samples",
        "reference_input",
        "requested_dt",
        "method",
    }:
        raise ValueError("Invalid named trajectory clock fields")
    if any(type(clock[key]) not in (int, float) for key in ("start_time", "last_time", "dt")):
        raise ValueError("Invalid named trajectory clock times")
    dt, _ = validate_resampling_options(clock["dt"], clock["method"])
    validate_resampling_options(clock["requested_dt"], clock["method"])
    count, first, last = clock["n_samples"], clock["start_time"], clock["last_time"]
    if (
        type(count) is not int
        or count < 1
        or dt is None
        or clock["reference_input"] != input_names[0]
    ):
        raise ValueError("Invalid named trajectory clock count/reference")
    try:
        span = dt * (count - 1)
    except OverflowError:
        raise ValueError("Named trajectory clock count is not representable") from None
    if (
        last < first
        or not math.isfinite(span)
        or not math.isclose(last - first, span, rel_tol=1e-9, abs_tol=1e-9 * dt)
    ):
        raise ValueError("Named trajectory clock times/count differ from its interval")
    for label, names, widths in (
        ("inputs", input_names, input_widths),
        ("outputs", output_names, output_widths),
    ):
        records = context[label]
        expected = set(names)
        if type(records) is not dict or (
            set(records) != expected
            if label == "inputs" or require_outputs
            else not set(records) <= expected
        ):
            raise ValueError(f"Named trajectory {label} must match declared model ports")
        for name, width in zip(names, widths):
            if name not in records:
                continue
            record = records[name]
            if type(record) is not dict or set(record) != {
                "variables",
                "source",
                "observation",
                "resampled",
                "held_out_samples",
            }:
                raise ValueError(f"Invalid named trajectory {label}[{name!r}] record")
            _signal_names(record["variables"], width)
            observation = record["observation"]
            if type(observation) is not dict or set(observation) != {
                "start_time",
                "last_time",
                "n_samples",
            }:
                raise ValueError("Invalid trajectory observation support")
            n = observation["n_samples"]
            if (
                type(n) is not int
                or n < 1
                or any(
                    type(observation[key]) not in (int, float)
                    for key in ("start_time", "last_time")
                )
                or observation["last_time"] < observation["start_time"]
                or ((n == 1) != (observation["last_time"] == observation["start_time"]))
                or type(record["resampled"]) is not bool
                or type(record["held_out_samples"]) is not int
                or record["held_out_samples"] < 0
                or type(record["source"]) is not dict
            ):
                raise ValueError("Inconsistent named trajectory observation metadata")
            if label == "inputs" and (
                first < observation["start_time"] or last > observation["last_time"]
            ):
                raise ValueError("Shared clock extrapolates an input observation range")
    if last != min(record["observation"]["last_time"] for record in context["inputs"].values()):
        raise ValueError("Shared clock must end at the latest common input observation")
    supervision = context["supervision"]
    if supervision is not None:
        if type(supervision) is not dict or set(supervision) != {
            "start",
            "stop",
            "prediction_leads",
        }:
            raise ValueError("Invalid named trajectory supervision metadata")
        start, stop = supervision["start"], supervision["stop"]
        leads = supervision["prediction_leads"]
        if (
            type(start) is not int
            or type(stop) is not int
            or not 0 <= start < stop <= count
            or type(leads) is not dict
            or set(leads) != set(output_names)
            or any(type(lead) is not int or lead < 1 for lead in leads.values())
            or (
                prediction_leads is not None
                and tuple(leads[name] for name in output_names) != prediction_leads
            )
        ):
            raise ValueError("Invalid named trajectory supervised origin bounds/leads")
        for name in output_names:
            lead = leads[name]
            if stop - 1 + lead >= count or name not in context["outputs"]:
                raise ValueError(
                    "Trajectory target supervision extends beyond observed input history"
                )
            observation = context["outputs"][name]["observation"]
            a = last - dt * (count - 1 - start - lead)
            b = last - dt * (count - stop - lead)
            if (
                a < observation["start_time"] - 1e-9 * dt
                or b > observation["last_time"] + 1e-9 * dt
                or observation["last_time"] > last
            ):
                raise ValueError(
                    "Trajectory supervision extrapolates or includes held-out target observations"
                )
    return copy.deepcopy(context)


def prepare_trajectory_port_history(
    history: Any,
    *,
    input_names: tuple[str, ...],
    input_widths: tuple[int, ...],
    output_names: tuple[str, ...],
    output_widths: tuple[int, ...],
    context: dict[str, Any] | None = None,
    dt: float | None = None,
    method: str = "linear",
    output_systems: tuple[Any, ...] | None = None,
) -> PreparedTrajectoryPorts:
    """Align complete observation history to the fitted clock, preserving labels."""
    requested_dt, method = validate_resampling_options(dt, method)
    if context is not None:
        context = validate_trajectory_ports_context(
            context, input_names, input_widths, output_names, output_widths
        )
        interval = context["clock"]["dt"]
    else:
        interval = requested_dt
    items = _named_values(history, input_names, "history")
    inputs = tuple(
        _read_port(item, width, f"history[{name!r}]")
        for item, name, width in zip(items, input_names, input_widths)
    )
    if context is not None:
        for name, port in zip(input_names, inputs):
            if port.variables != context["inputs"][name]["variables"]:
                raise ValueError(f"History trajectory components/order differ for input {name!r}")
    grid, interval = _shared_grid(inputs, interval)
    values = tuple(
        _sample(port, grid, interval, method, f"history[{name!r}]")
        for name, port in zip(input_names, inputs)
    )
    if context is not None:
        outputs = copy.deepcopy(context["outputs"])
        systems = output_systems or tuple(None for _ in output_names)
    elif len(input_names) == len(output_names) == 1 and input_widths == output_widths:
        outputs = {output_names[0]: _record(inputs[0], grid)}
        systems = (inputs[0].system,)
    else:
        outputs, systems = {}, tuple(None for _ in output_names)
    prepared_context = _context(
        grid, interval, requested_dt, method, input_names, inputs, outputs, None
    )
    prepared_context = validate_trajectory_ports_context(
        prepared_context,
        input_names,
        input_widths,
        output_names,
        output_widths,
        require_outputs=bool(outputs),
    )
    return PreparedTrajectoryPorts(
        grid,
        values,
        (),
        0,
        len(grid),
        prepared_context,
        tuple(port.system for port in inputs),
        systems,
    )


def prepare_trajectory_port_drivers(
    drivers: Any,
    *,
    input_names: tuple[str, ...],
    input_widths: tuple[int, ...],
    last_time: float | None,
    dt: float,
    steps: int,
    method: str = "linear",
    input_records: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, ...]:
    """Query future graph-input ticks, never re-anchor at driver endpoints."""
    interval, method = validate_resampling_options(dt, method)
    if interval is None or last_time is None:
        raise ValueError("Trajectory drivers require timestamped or cached trajectory history")
    if type(steps) is not int or steps < 0:
        raise ValueError("Future driver steps must be a non-negative integer")
    items = _named_values(drivers, input_names, "future_inputs")
    ports = tuple(
        _read_port(item, width, f"future_inputs[{name!r}]")
        for item, name, width in zip(items, input_names, input_widths)
    )
    if input_records is not None:
        for name, port in zip(input_names, ports):
            if port.variables != input_records[name]["variables"]:
                raise ValueError(f"Future trajectory components/order differ for input {name!r}")
    with np.errstate(over="ignore", invalid="ignore"):
        queries = last_time + interval * np.arange(1, steps + 1, dtype=np.float64)
    validate_time_grid(np.concatenate(([last_time], queries)), interval)
    return tuple(
        _sample(port, queries, interval, method, f"future_inputs[{name!r}]")
        for name, port in zip(input_names, ports)
    )


def trajectory_port_forecast_times(
    context: Mapping[str, Any],
    horizon: int,
    *,
    history_length: int,
    warmup_lead: int = 1,
    return_warmup: bool = False,
) -> np.ndarray:
    """Build one shared physical result clock before any graph execution."""
    clock = context["clock"]
    last, dt = clock["last_time"], clock["dt"]
    with np.errstate(over="ignore", invalid="ignore"):
        times = last + dt * np.arange(1, horizon + 1, dtype=np.float64)
    validate_time_grid(np.concatenate(([last], times)), dt)
    if return_warmup:
        prefix = last - dt * np.arange(history_length - warmup_lead - 1, -1, -1, dtype=np.float64)
        times = np.concatenate((prefix, times))
        validate_time_grid(times, dt)
    return np.asarray(times, dtype=np.float64)


def forecast_trajectory_ports(
    values: Mapping[str, np.ndarray],
    context: Mapping[str, Any],
    *,
    output_names: tuple[str, ...],
    times: np.ndarray,
    output_systems: tuple[Any, ...] | None = None,
) -> Any:
    """Render original-unit named predictions with each target's component labels."""
    if set(values) != set(output_names) or set(context["outputs"]) != set(output_names):
        raise ValueError("Trajectory outputs require per-head target component metadata")
    dt, last = context["clock"]["dt"], context["clock"]["last_time"]
    validate_time_grid(times, dt)
    if not len(times):
        raise ValueError("Trajectory forecasts require at least one result sample")
    constructor = trajectory_type()
    systems = output_systems or tuple(None for _ in output_names)
    results = {}
    for name, system in zip(output_names, systems):
        array = np.asarray(values[name])
        record = context["outputs"][name]
        variables = record["variables"]
        if (
            array.ndim != 2
            or array.shape != (len(times), len(variables))
            or array.dtype.kind not in "iuf"
        ):
            raise ValueError(f"Forecast head {name!r} must have shape (time, {len(variables)})")
        if not np.isfinite(array).all():
            raise ValueError(f"Forecast head {name!r} contains nonfinite predictions")
        metadata = {
            "dt": dt,
            "t0": float(times[0]),
            "ic": array[0].copy(),
            "variables": list(variables),
            "source": copy.deepcopy(record["source"]),
            "resdag": {
                "kind": "forecast",
                "output": name,
                "history_last_time": last,
                "sampling": copy.deepcopy(context["clock"]),
            },
        }
        source_metadata = record["source"].get("metadata", {})
        if isinstance(source_metadata, dict) and "field_shape" in source_metadata:
            metadata["field_shape"] = copy.deepcopy(source_metadata["field_shape"])
        results[name] = constructor(
            times.copy(), array, system=system, meta=metadata, variables=variables
        )
    return results[output_names[0]] if len(output_names) == 1 else results
