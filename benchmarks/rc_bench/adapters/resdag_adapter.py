"""resdag adapter (CPU and GPU).

Builds a *matched* ESN: reservoir states -> ridge readout, no input
concatenation, mirroring reservoirpy's ``Reservoir >> Ridge`` and
ReservoirComputing.jl's ``ESN``. CPU runs float64 (to match the NumPy/Julia
baselines); GPU runs float32 (the realistic GPU dtype — consumer cards run
float64 at a small fraction of float32 throughput).
"""

from __future__ import annotations

import numpy as np

from ..config import HParams, Point
from ..timing import RunResult, repeat_timed
from .base import Adapter


def _available() -> tuple[bool, str]:
    try:
        import torch  # noqa: F401

        import resdag  # noqa: F401
    except Exception as e:  # pragma: no cover - environment probe
        return False, f"import failed: {e}"
    return True, ""


class _ResdagBase(Adapter):
    family = "resdag"
    _torch_device = "cpu"
    _dtype_name = "float64"
    # Drive the autoregressive forecast through the torch.compile (cudagraph)
    # path. Worth it on GPU, where per-step kernel-launch overhead — not the
    # tiny matmul — dominates a long closed-loop forecast; on CPU the eager flat
    # step is already launch-free, so leave it off there.
    _compile_forecast = False

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        ok, reason = _available()
        if not ok:
            return ok, reason
        if cls._torch_device == "cuda":
            import torch

            if not torch.cuda.is_available():
                return False, "CUDA not available"
        return True, ""

    def _build(self, hp: HParams, point: Point):
        import pytorch_symbolic as ps
        import torch

        from resdag import ESNModel
        from resdag.layers import ESNLayer
        from resdag.layers.readouts import CGReadoutLayer

        torch.manual_seed(hp.seed)
        dtype = torch.float64 if self._dtype_name == "float64" else torch.float32

        inp = ps.Input((1, hp.dim))
        res = ESNLayer(
            reservoir_size=point.reservoir_size,
            feedback_size=hp.dim,
            spectral_radius=hp.spectral_radius,
            leak_rate=hp.leak_rate,
            activation=hp.activation,
            topology=("erdos_renyi", {"p": hp.connectivity}),
        )(inp)
        out = CGReadoutLayer(point.reservoir_size, hp.dim, alpha=hp.ridge, name="output")(res)
        model = ESNModel(inp, out)
        model = model.to(device=self._torch_device, dtype=dtype)
        return model, dtype

    def _tensor(self, arr: np.ndarray, dtype):
        import torch

        return torch.as_tensor(arr, dtype=dtype, device=self._torch_device).unsqueeze(0)

    def _sync(self):
        if self._torch_device == "cuda":
            import torch

            torch.cuda.synchronize()

    def _fit(self, model, dtype, series: np.ndarray, point: Point):
        from resdag.training import ESNTrainer

        warm = self._tensor(series[: point.warmup], dtype)
        x = self._tensor(series[point.warmup : point.train_len], dtype)
        y = self._tensor(series[point.warmup + 1 : point.train_len + 1], dtype)
        ESNTrainer(model).fit(warmup_inputs=(warm,), train_inputs=(x,), targets={"output": y})

    def time_train(self, series, hp, point, repeats, warmups) -> RunResult:
        def setup():
            return self._build(hp, point)

        def op(ctx):
            model, dtype = ctx
            self._fit(model, dtype, series, point)
            return None

        times, _ = repeat_timed(setup, op, repeats, warmups, sync=self._sync)
        return RunResult(
            times=times, info={"device": self._torch_device, "dtype": self._dtype_name}
        )

    def time_forecast(self, series, hp, point, repeats, warmups) -> RunResult:
        import torch

        model, dtype = self._build(hp, point)
        self._fit(model, dtype, series, point)
        warm = self._tensor(series[: point.warmup], dtype)

        def setup():
            model.reset_reservoirs()
            return model

        def op(m):
            with torch.no_grad():
                return m.forecast(warm, horizon=point.horizon, compile=self._compile_forecast)

        times, last = repeat_timed(setup, op, repeats, warmups, sync=self._sync)

        rmse = None
        if last is not None:
            truth = self._forecast_truth(series, point)
            pred = last.squeeze(0).detach().cpu().numpy()[: len(truth)]
            n = min(len(truth), len(pred), 200)  # short-horizon sanity window
            if n > 0:
                rmse = float(np.sqrt(np.mean((pred[:n] - truth[:n]) ** 2)))
        return RunResult(
            times=times, rmse=rmse, info={"device": self._torch_device, "dtype": self._dtype_name}
        )

    def predict_trajectory(self, series, hp, point) -> np.ndarray:
        import torch

        model, dtype = self._build(hp, point)
        self._fit(model, dtype, series, point)
        warm = self._tensor(series[: point.warmup], dtype)
        model.reset_reservoirs()
        with torch.no_grad():
            out = model.forecast(warm, horizon=point.horizon)
        return out.squeeze(0).detach().cpu().numpy().astype(np.float64)


class ResdagCPUAdapter(_ResdagBase):
    key = "resdag-cpu"
    label = "resdag (CPU)"
    _torch_device = "cpu"
    _dtype_name = "float64"


class ResdagGPUAdapter(_ResdagBase):
    key = "resdag-gpu"
    label = "resdag (GPU)"
    _torch_device = "cuda"
    _dtype_name = "float32"
    _compile_forecast = True
