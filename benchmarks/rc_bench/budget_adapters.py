"""Explicit common vanilla ESNs for the equal-time-budget comparison.

These recipes deliberately override native initializer defaults. They do not
change the historical matched-workload adapters or public ResDAG defaults.
"""

from __future__ import annotations

import os
import select
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import resource
except ImportError:  # pragma: no cover - Windows has no standard resource module
    resource = None

from .adapters import julia_adapter
from .adapters.resdag_adapter import ResdagCPUAdapter
from .config import HParams, Point


def _peak_rss_bytes() -> int | None:
    if resource is None:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _array_bytes(value) -> int:
    if hasattr(value, "indptr"):
        return value.data.nbytes + value.indices.nbytes + value.indptr.nbytes
    return int(np.asarray(value).nbytes)


@dataclass(frozen=True)
class CommonParams:
    reservoir_size: int
    spectral_radius: float
    leak_rate: float
    input_scaling: float
    bias_scaling: float
    mean_degree: int
    alpha: float

    def hparams(self, seed: int, dim: int) -> HParams:
        return HParams(
            dim=dim,
            spectral_radius=self.spectral_radius,
            leak_rate=self.leak_rate,
            connectivity=self.mean_degree / self.reservoir_size,
            input_scaling=self.input_scaling,
            ridge=self.alpha,
            seed=seed,
        )


@dataclass(frozen=True)
class GaussianRecurrent:
    """Gaussian weights on directed independent edges, including self-loops."""

    density: float

    def __call__(self, n, seed=None):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n, n)) * (rng.random((n, n)) < self.density)


class PythonCandidate:
    def __init__(self, library: str, model: Any, reservoir: Any = None):
        self.library, self.model, self.reservoir = library, model, reservoir

    def forecast(self, history: np.ndarray, horizon: int) -> tuple[np.ndarray, dict]:
        started = time.perf_counter()
        if self.library == "resdag-cpu":
            import torch

            self.model.reset_reservoirs()
            with torch.no_grad():
                prediction = self.model.forecast(
                    torch.as_tensor(history, dtype=torch.float64).unsqueeze(0), horizon=horizon
                )
            prediction = prediction.squeeze(0).detach().cpu().numpy()
        else:
            self.reservoir.reset()
            value = self.model.run(np.ascontiguousarray(history))[-1]
            prediction = np.empty((horizon, history.shape[1]), dtype=np.float64)
            prediction[0] = value
            for index in range(1, horizon):
                value = self.model(value)
                prediction[index] = value
        return np.asarray(prediction, dtype=np.float64), {
            "forecast_seconds": time.perf_counter() - started
        }


class PythonBudgetAdapter:
    def __init__(self, library: str):
        if library not in ("resdag-cpu", "reservoirpy"):
            raise ValueError(f"Unsupported Python comparison library: {library}")
        self.library = library

    def fit(
        self, training: np.ndarray, params: CommonParams, seed: int, washout: int
    ) -> tuple[PythonCandidate, dict]:
        started = time.perf_counter()
        dim = training.shape[1]
        if self.library == "resdag-cpu":
            import torch

            from resdag import ESNLayer, ESNModel, RidgeReadoutLayer, reservoir_input

            inputs = reservoir_input(dim, dtype=torch.float64)
            features = ESNLayer(
                params.reservoir_size,
                feedback_size=dim,
                recurrent_initializer=GaussianRecurrent(params.mean_degree / params.reservoir_size),
                spectral_radius=params.spectral_radius,
                leak_rate=params.leak_rate,
                feedback_initializer=("uniform", {"input_scaling": params.input_scaling}),
                bias=True,
                bias_scaling=params.bias_scaling,
                seed=seed,
                dtype=torch.float64,
            )(inputs)
            readout = RidgeReadoutLayer(
                params.reservoir_size, dim, alpha=params.alpha, bias=False, name="output"
            ).double()
            model = ESNModel(inputs, readout(features))
            build_seconds = time.perf_counter() - started
            fit_started = time.perf_counter()
            point = Point(
                reservoir_size=params.reservoir_size,
                train_len=len(training) - 1,
                warmup=washout,
                horizon=1,
            )
            ResdagCPUAdapter()._fit(model, torch.float64, training, point)
            candidate = PythonCandidate(self.library, model)
            storage_bytes = sum(
                t.numel() * t.element_size() for t in (*model.parameters(), *model.buffers())
            )
            storage = "dense Torch"
        else:
            from reservoirpy.mat_gen import normal, uniform
            from reservoirpy.nodes import Reservoir, Ridge

            reservoir = Reservoir(
                units=params.reservoir_size,
                sr=params.spectral_radius,
                lr=params.leak_rate,
                rc_connectivity=params.mean_degree / params.reservoir_size,
                input_connectivity=1.0,
                input_scaling=params.input_scaling,
                W=normal(loc=0.0, scale=1.0),
                Win=uniform(low=-1.0, high=1.0),
                bias=uniform(low=-params.bias_scaling, high=params.bias_scaling),
                activation="tanh",
                seed=seed,
                dtype=np.float64,
            )
            model = reservoir >> Ridge(ridge=params.alpha, fit_bias=False)
            build_seconds = time.perf_counter() - started
            fit_started = time.perf_counter()
            # ReservoirPy initializes its arrays lazily inside fit. That work
            # remains charged to the candidate, even though recorded as fit.
            model.fit(training[:-1], training[1:], warmup=washout)
            candidate = PythonCandidate(self.library, model, reservoir)
            readout = next(node for node in model.nodes if hasattr(node, "fit_bias"))
            storage_bytes = sum(
                _array_bytes(value)
                for value in (reservoir.W, reservoir.Win, reservoir.bias, readout.Wout)
            )
            storage = "SciPy CSR"
        return candidate, {
            "build_seconds": build_seconds,
            "fit_seconds": time.perf_counter() - fit_started,
            "fit_calls": 1,
            "storage": storage,
            "model_tensor_storage_bytes": storage_bytes,
            "feature_matrix_bytes_estimate": (len(training) - 1 - washout)
            * params.reservoir_size
            * 8,
            "peak_process_rss_bytes_including_startup": _peak_rss_bytes(),
        }

    def close(self) -> None:
        pass


class JuliaCandidate:
    def __init__(self, adapter: JuliaBudgetAdapter, dim: int):
        self.adapter, self.dim = adapter, dim

    def forecast(self, history: np.ndarray, horizon: int) -> tuple[np.ndarray, dict]:
        self.adapter.write(history)
        output = self.adapter.directory / "forecast.bin"
        result = self.adapter.command(
            ["FORECAST", self.adapter.data, self.dim, len(history), horizon, output]
        )
        prediction = np.fromfile(output, dtype="<f8").reshape(horizon, self.dim)
        return prediction, {"worker_forecast_seconds": result[0]}


class JuliaBudgetAdapter:
    """One isolated standard Julia subprocess, no per-candidate startup charge."""

    library = "rcjl"

    def __init__(self, project: Path | None = None, storage: str = "dense"):
        if storage not in ("dense", "sparse"):
            raise ValueError("Julia storage must be dense or sparse")
        self.storage = storage
        executable = shutil.which("julia")
        if executable is None:
            raise RuntimeError("The equal-budget comparison requires julia on PATH")
        self.temporary = tempfile.TemporaryDirectory(prefix="resdag-julia-budget-")
        self.directory = Path(self.temporary.name)
        self.data = self.directory / "data.bin"
        project = project or Path(
            os.environ.get("RESDAG_JULIA_PROJECT", julia_adapter._JULIA_PROJECT)
        )
        worker = Path(__file__).resolve().parents[1] / "julia/rc_budget.jl"
        self.stderr_path = self.directory / "worker.stderr"
        self.stderr = self.stderr_path.open("w")
        self.process = subprocess.Popen(
            [executable, f"--project={project}", "--startup-file=no", str(worker)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
            bufsize=1,
            env=os.environ
            | {
                "JULIA_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "JULIA_PKG_PRECOMPILE_AUTO": "0",
            },
        )
        try:
            greeting = self.readline(timeout=180).split("\t")
        except BaseException:
            self.close()
            raise
        if len(greeting) != 3 or greeting[0] != "READY":
            self.close()
            raise RuntimeError(f"Unexpected Julia budget worker greeting: {greeting}")
        self.versions = {"julia": greeting[1], "ReservoirComputing": greeting[2]}

    def readline(self, timeout: float | None = None) -> str:
        stdout = self.process.stdout
        assert stdout is not None
        if not select.select([stdout], [], [], timeout)[0]:
            self.close()
            raise RuntimeError("Julia budget worker timed out")
        line = stdout.readline().rstrip("\n")
        if not line:
            self.stderr.flush()
            raise RuntimeError(
                f"Julia budget worker exited: {self.stderr_path.read_text()[-4000:]}"
            )
        return line

    def command(self, fields: list) -> list[float]:
        values = [str(value) for value in fields]
        if any("\t" in value or "\n" in value for value in values):
            raise ValueError("Julia worker fields cannot contain tabs or newlines")
        assert self.process.stdin is not None
        self.process.stdin.write("\t".join(values) + "\n")
        self.process.stdin.flush()
        result = self.readline().split("\t")
        if result[0] != "OK":
            raise RuntimeError("Julia candidate failed: " + " ".join(result[1:]))
        return [float(value) for value in result[1:]]

    def write(self, data: np.ndarray) -> None:
        np.ascontiguousarray(data, dtype="<f8").tofile(self.data)

    def fit(
        self, training: np.ndarray, params: CommonParams, seed: int, washout: int
    ) -> tuple[JuliaCandidate, dict]:
        self.write(training)
        result = self.command(
            [
                "FIT",
                self.data,
                training.shape[1],
                len(training),
                params.reservoir_size,
                washout,
                params.spectral_radius,
                params.leak_rate,
                params.mean_degree / params.reservoir_size,
                params.input_scaling,
                params.bias_scaling,
                params.alpha,
                seed,
                self.storage,
            ]
        )
        return JuliaCandidate(self, training.shape[1]), {
            "worker_seconds": result[0],
            "build_seconds": result[1],
            "fit_seconds": result[2],
            "fit_calls": 1,
            "storage": "Julia CSC" if self.storage == "sparse" else "dense Julia Matrix",
            "parameter_tree_heap_bytes": int(result[3]),
            "peak_process_rss_bytes_including_startup": int(result[4]),
            "feature_matrix_bytes_estimate": (len(training) - 1 - washout)
            * params.reservoir_size
            * 8,
        }

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                assert self.process.stdin is not None
                self.process.stdin.write("CLOSE\n")
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait(timeout=5)
        self.stderr.close()
        self.temporary.cleanup()


def prepare_adapter(library: str, *, julia_storage: str = "dense"):
    """Load packages/JIT against fixed synthetic data, outside the search budget."""
    started = time.perf_counter()
    adapter = (
        JuliaBudgetAdapter(storage=julia_storage)
        if library == "rcjl"
        else PythonBudgetAdapter(library)
    )
    if library == "resdag-cpu":
        import torch

        torch.set_num_threads(1)
    params = CommonParams(8, 0.5, 0.5, 0.1, 0.1, 4, 1e-3)
    values = np.sin(np.arange(60, dtype=np.float64).reshape(20, 3) * 0.1)
    try:
        candidate, _ = adapter.fit(values, params, seed=0, washout=3)
        candidate.forecast(values[-3:], 4)
    except BaseException:
        adapter.close()
        raise
    info = {"startup_synthetic_warmup_seconds": time.perf_counter() - started}
    if library == "rcjl":
        info["versions"] = adapter.versions
    return adapter, info


def julia_storage_pilot() -> dict:
    """Fixed synthetic runtime-only selection, with no trajectory or accuracy scores.

    Each layout is warmed once. ABBA measured rounds reduce one-way order bias;
    the selection minimizes total build/fit plus three history+rollout calls.
    """
    adapter, startup = prepare_adapter("rcjl")
    samples = np.sin(np.arange(1203, dtype=np.float64).reshape(401, 3) * 0.01)
    rows = []
    try:
        for width in (300, 600):
            params = CommonParams(width, 0.9, 0.3, 0.2, 0.1, 6, 1e-6)
            for storage in ("dense", "sparse"):
                adapter.storage = storage
                candidate, _ = adapter.fit(samples, params, 0, 20)
                candidate.forecast(samples[-20:], 50)
            for storage in ("dense", "sparse", "sparse", "dense"):
                adapter.storage = storage
                started = time.perf_counter()
                candidate, fit = adapter.fit(samples, params, 1, 20)
                for _ in range(3):
                    candidate.forecast(samples[-20:], 50)
                rows.append(
                    {
                        "width": width,
                        "storage": storage,
                        "elapsed_seconds": time.perf_counter() - started,
                        "fit": fit,
                    }
                )
    finally:
        adapter.close()
    totals = {
        storage: sum(row["elapsed_seconds"] for row in rows if row["storage"] == storage)
        for storage in ("dense", "sparse")
    }
    return {
        "startup": startup,
        "selection": min(totals, key=totals.get),
        "totals_seconds": totals,
        "measurements": rows,
        "data": "fixed synthetic sine; widths300/600;401samples;washout20;3x50step forecasts;ABBA",
        "accuracy_data_accessed": False,
    }
