"""Read the fixed, independently evaluated forecasts used in the documentation.

This module only reads observations and predictions. Figure generation never
integrates a system, fits a reservoir or chooses a case by its forecast score.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SHOWCASE = ROOT / "docs/assets/data/forecast-showcase-2026-10-24.json"
SELECTION = {
    "trajectory": 0,
    "reservoir_seed": 7849201,
    "origin_index": 0,
    "predeclared": True,
}


def _reject_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON constant in forecast showcase: {value}")


def load_showcase(path: Path = SHOWCASE) -> dict:
    """Validate fixed display cases and return their recorded original-unit arrays."""
    record = json.loads(path.read_text(), parse_constant=_reject_constant)
    if record.get("schema_version") != 1 or record.get("selection") != SELECTION:
        raise ValueError("Unsupported forecast showcase schema or display selection")
    if set(record.get("cases", {})) != {"lorenz", "rossler"}:
        raise ValueError("Forecast showcase must contain the two predeclared systems")
    for name, case in record["cases"].items():
        time = np.asarray(case["time"], dtype=np.float64)
        truth = np.asarray(case["truth"], dtype=np.float64)
        prediction = np.asarray(case["prediction"], dtype=np.float64)
        scale = np.asarray(case["training_scale"], dtype=np.float64)
        dt, exponent = case["dt"], case["lambda_max"]
        width = len(case["variables"])
        if (
            time.ndim != 1
            or len(time) < 2
            or truth.shape != (len(time), width)
            or prediction.shape != truth.shape
            or scale.shape != (width,)
            or not all(np.isfinite(array).all() for array in (time, truth, prediction, scale))
            or np.any(scale <= 0)
            or not np.isfinite(dt)
            or dt <= 0
            or not np.isfinite(exponent)
            or exponent <= 0
            or not np.allclose(np.diff(time), dt, rtol=0, atol=1e-10)
        ):
            raise ValueError(f"Invalid forecast showcase values or clock for {name}")
        digest = hashlib.sha256(prediction.astype("<f8", copy=False).tobytes()).hexdigest()
        if digest != case["prediction_sha256"]:
            raise ValueError(f"Forecast prediction hash mismatch for {name}")
    return record


def errors(case: dict) -> np.ndarray:
    """Component-normalized RMS error, using the stored training-only scales."""
    difference = np.asarray(case["prediction"]) - np.asarray(case["truth"])
    return np.sqrt(np.mean((difference / np.asarray(case["training_scale"])) ** 2, axis=-1))


def valid_steps(case: dict, threshold: float = 0.3) -> int:
    """Length of the uninterrupted finite prefix strictly below the threshold."""
    error = errors(case)
    failed = np.flatnonzero(~np.isfinite(error) | (error >= threshold))
    return int(failed[0]) if len(failed) else len(error)


def trajectories(case: dict):
    """Construct public TSDynamics trajectories without changing sample values."""
    import tsdynamics as ts

    kwargs = {"t": np.asarray(case["time"]), "variables": case["variables"]}
    return (
        ts.Trajectory(y=np.asarray(case["truth"]), **kwargs),
        ts.Trajectory(y=np.asarray(case["prediction"]), **kwargs),
    )
