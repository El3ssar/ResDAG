"""ReservoirComputing.jl adapter (CPU, via a Julia subprocess).

Cross-language timing is done *inside* Julia (see ``benchmarks/julia/rc_bench.jl``)
so the JVM-like JIT warmup and process startup never enter the measurement. The
shared series is handed over as a raw little-endian float64 file (NumPy ``(T,dim)``
C-order == Julia ``(dim,T)`` column-major), and the worker prints ``TIMES``/``RMSE``
lines we parse back.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ..config import HParams, Point
from ..timing import RunResult
from .base import Adapter

_BENCH_DIR = Path(__file__).resolve().parents[2]  # .../benchmarks
_JULIA_PROJECT = _BENCH_DIR / "julia"
_JULIA_SCRIPT = _JULIA_PROJECT / "rc_bench.jl"


class ReservoirComputingJLAdapter(Adapter):
    key = "rcjl"
    label = "ReservoirComputing.jl"
    family = "rcjl"
    device = "cpu"

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        if shutil.which("julia") is None:
            return False, "julia not on PATH"
        if not _JULIA_SCRIPT.exists():
            return False, f"worker script missing: {_JULIA_SCRIPT}"
        manifest = _JULIA_PROJECT / "Manifest.toml"
        if not manifest.exists():
            return False, (
                "Julia project not instantiated — run: "
                f"julia --project={_JULIA_PROJECT} -e 'using Pkg; Pkg.instantiate()'"
            )
        return True, ""

    def _run(
        self, series: np.ndarray, hp: HParams, point: Point, op: str, repeats: int, warmups: int
    ) -> RunResult:
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            data_path = f.name
            np.ascontiguousarray(series, dtype="<f8").tofile(data_path)
        try:
            args = [
                "julia",
                f"--project={_JULIA_PROJECT}",
                str(_JULIA_SCRIPT),
                data_path,
                str(hp.dim),
                str(len(series)),
                str(point.reservoir_size),
                str(point.train_len),
                str(point.warmup),
                str(point.horizon),
                repr(hp.spectral_radius),
                repr(hp.leak_rate),
                repr(hp.connectivity),
                repr(hp.input_scaling),
                repr(hp.ridge),
                str(hp.seed),
                op,
                str(repeats),
                str(warmups),
            ]
            proc = subprocess.run(args, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"julia worker failed (op={op}):\n{proc.stderr[-2000:]}")
            times: list[float] = []
            rmse: float | None = None
            for line in proc.stdout.splitlines():
                if line.startswith("TIMES "):
                    body = line[len("TIMES ") :].strip()
                    times = [float(x) for x in body.split(",") if x]
                elif line.startswith("RMSE "):
                    val = line[len("RMSE ") :].strip()
                    try:
                        f_val = float(val)
                        rmse = None if math.isnan(f_val) else f_val
                    except ValueError:
                        rmse = None
            return RunResult(times=times, rmse=rmse, info={"device": "cpu", "dtype": "float64"})
        finally:
            Path(data_path).unlink(missing_ok=True)

    # ReservoirComputing.jl (v0.12, Lux-based) has high per-call cost and low
    # run-to-run variance, and every subprocess re-pays Julia's JIT compile in
    # the discarded warmup. Cap the timed repeats so it doesn't dominate the
    # whole sweep; one post-warmup run is plenty here.
    _MAX_REPEATS = 1
    # ...and skip cells where it is impractically slow (it scales far worse than
    # the others — ~24 s at N=1000, minutes beyond). These show as "—" in the
    # tables; resdag and reservoirpy still cover the full range.
    _MAX_N = 1000
    _MAX_TRAIN_LEN = 10000

    def _skip_reason(self, point: Point) -> str | None:
        if point.reservoir_size > self._MAX_N:
            return f"skipped: N={point.reservoir_size} > {self._MAX_N} (RC.jl too slow)"
        if point.train_len > self._MAX_TRAIN_LEN:
            return f"skipped: train_len={point.train_len} > {self._MAX_TRAIN_LEN} (RC.jl too slow)"
        return None

    def time_train(self, series, hp, point, repeats, warmups) -> RunResult:
        skip = self._skip_reason(point)
        if skip:
            return RunResult(times=[], info={"skipped": skip})
        return self._run(
            series, hp, point, "train", min(repeats, self._MAX_REPEATS), max(warmups, 1)
        )

    def time_forecast(self, series, hp, point, repeats, warmups) -> RunResult:
        skip = self._skip_reason(point)
        if skip:
            return RunResult(times=[], info={"skipped": skip})
        return self._run(
            series, hp, point, "forecast", min(repeats, self._MAX_REPEATS), max(warmups, 1)
        )

    def predict_trajectory(self, series, hp, point) -> np.ndarray:
        skip = self._skip_reason(point)
        if skip:
            raise RuntimeError(skip)
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            data_path = f.name
            np.ascontiguousarray(series, dtype="<f8").tofile(data_path)
        out_path = data_path + ".out"
        try:
            args = [
                "julia",
                f"--project={_JULIA_PROJECT}",
                str(_JULIA_SCRIPT),
                data_path,
                str(hp.dim),
                str(len(series)),
                str(point.reservoir_size),
                str(point.train_len),
                str(point.warmup),
                str(point.horizon),
                repr(hp.spectral_radius),
                repr(hp.leak_rate),
                repr(hp.connectivity),
                repr(hp.input_scaling),
                repr(hp.ridge),
                str(hp.seed),
                "trajectory",
                "1",
                "0",
                out_path,
            ]
            proc = subprocess.run(args, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"julia worker failed (trajectory):\n{proc.stderr[-2000:]}")
            # Julia wrote a (dim, horizon) column-major Float64 array; that flat
            # byte order is exactly NumPy (horizon, dim) C-order.
            flat = np.fromfile(out_path, dtype="<f8")
            return flat.reshape(point.horizon, hp.dim)
        finally:
            Path(data_path).unlink(missing_ok=True)
            Path(out_path).unlink(missing_ok=True)
