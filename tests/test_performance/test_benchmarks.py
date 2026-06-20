"""Performance assertions — opt-in, machine-dependent.

Deselected by default (``-m "not benchmark"`` in addopts). Run with::

    pytest tests/test_performance -m benchmark --no-cov

These guard the regressions that motivated the GPU fast paths: at
GPU-favorable sizes, training and forecasting on CUDA must beat the CPU,
and the projected fast path must not be slower than the per-step fallback.
Thresholds are deliberately loose (1.2-2x margins) — they catch order-of-
magnitude regressions, not noise.
"""

import time

import pytest
import torch

import resdag as rd
from resdag.training import ESNTrainer

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _timed(fn, device: str, warmup: int = 1, reps: int = 3) -> float:
    for _ in range(warmup):
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(reps):
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - start) / reps


def _training_setup(device: str, batch: int, reservoir_size: int, train_steps: int):
    torch.manual_seed(0)
    timesteps = train_steps + 400
    t = torch.linspace(0, 100, timesteps)
    data = torch.sin(t).reshape(1, -1, 1).repeat(batch, 1, 3).to(device)
    splits = rd.utils.prepare_esn_data(
        data, warmup_steps=100, train_steps=train_steps, val_steps=200
    )
    model = rd.models.ott_esn(reservoir_size=reservoir_size, feedback_size=3, output_size=3).to(
        device
    )
    return model, splits


@pytest.mark.benchmark
@pytest.mark.gpu
@cuda_required
class TestGPUBeatsCPU:
    """At GPU-favorable sizes the GPU must win — this was broken before 0.5."""

    BATCH = 8
    RESERVOIR = 2000
    TRAIN_STEPS = 1000

    def _times(self, phase: str) -> tuple[float, float]:
        times = {}
        for device in ("cpu", "cuda"):
            model, (warmup, train, target, f_warmup, _val) = _training_setup(
                device, self.BATCH, self.RESERVOIR, self.TRAIN_STEPS
            )
            if phase == "forward":
                times[device] = _timed(lambda: (model.reset_reservoirs(), model(train)), device)
            elif phase == "fit":
                trainer = ESNTrainer(model)
                times[device] = _timed(
                    lambda: trainer.fit((warmup,), (train,), targets={"output": target}),
                    device,
                )
            else:
                times[device] = _timed(lambda: model.forecast(f_warmup, horizon=300), device)
        return times["cpu"], times["cuda"]

    def test_forward_faster_on_gpu(self):
        cpu, cuda = self._times("forward")
        assert cuda < cpu, f"GPU forward slower than CPU: {cuda:.3f}s vs {cpu:.3f}s"

    def test_fit_faster_on_gpu(self):
        cpu, cuda = self._times("fit")
        assert cuda < cpu, f"GPU fit slower than CPU: {cuda:.3f}s vs {cpu:.3f}s"

    def test_forecast_faster_on_gpu(self):
        cpu, cuda = self._times("forecast")
        assert cuda < cpu, f"GPU forecast slower than CPU: {cuda:.3f}s vs {cpu:.3f}s"


@pytest.mark.benchmark
class TestFastPathSpeed:
    """The projected fast path must not lose to the per-step fallback."""

    def test_fast_path_not_slower_than_fallback(self):
        torch.manual_seed(0)
        layer = rd.layers.ESNLayer(reservoir_size=500, feedback_size=3, spectral_radius=0.9)
        cell = layer.cell
        feedback = torch.randn(4, 1000, 3)

        def run_fast():
            layer.reset_state()
            layer(feedback)

        def run_fallback():
            # The per-step path the layer uses for cells without projection.
            state = cell.init_state(feedback.shape[0], feedback.device, feedback.dtype)
            with torch.no_grad():
                for t in range(feedback.shape[1]):
                    _, state = cell([feedback[:, t, :]], state)

        t_fast = _timed(run_fast, "cpu")
        t_fallback = _timed(run_fallback, "cpu")

        assert t_fast < t_fallback * 1.2, f"fast path {t_fast:.3f}s vs fallback {t_fallback:.3f}s"


@pytest.mark.benchmark
class TestFlatForecastSpeed:
    """The flat single-step engine must beat per-step graph re-execution (#254)."""

    def _build(self, device: str, reservoir_size: int = 200):
        torch.manual_seed(0)
        inp = rd.core.reservoir_input(3)
        states = rd.layers.ESNLayer(
            reservoir_size=reservoir_size, feedback_size=3, spectral_radius=0.9
        )(inp)
        out = rd.CGReadoutLayer(reservoir_size, 3, name="output")(states)
        return rd.core.ESNModel(inp, out).to(device)

    @staticmethod
    def _graph_forecast(model, warmup, horizon):
        """The pre-#254 path: one full ``model(...)`` graph walk per step."""
        model.reset_reservoirs()
        warm = model.warmup(warmup, return_outputs=True)
        cur = warm[:, -1:, :]
        buf = torch.empty(
            warmup.shape[0], horizon, warm.shape[-1], dtype=warmup.dtype, device=warmup.device
        )
        with torch.no_grad():
            for t in range(horizon):
                out = model(cur)
                buf[:, t, :] = out.squeeze(1)
                cur = out
        return buf

    def test_flat_beats_graph_reexecution_cpu(self):
        model = self._build("cpu")
        warmup = torch.randn(1, 50, 3, dtype=torch.float32)
        horizon = 2000

        t_graph = _timed(lambda: self._graph_forecast(model, warmup, horizon), "cpu")
        t_flat = _timed(
            lambda: (model.reset_reservoirs(), model.forecast(warmup, horizon=horizon)), "cpu"
        )
        speedup = t_graph / t_flat
        assert speedup >= 1.8, (
            f"flat forecast only {speedup:.2f}x the graph path "
            f"(flat={t_flat:.3f}s, graph={t_graph:.3f}s); expected >= 1.8x at H={horizon}"
        )


@pytest.mark.benchmark
@pytest.mark.gpu
@cuda_required
class TestGramDtypeAuto:
    """float64 Gram formation on consumer GPUs is the classic slowdown —
    the auto default must avoid it."""

    def test_auto_gram_beats_forced_float64_on_gpu(self):
        torch.manual_seed(0)
        X = torch.randn(20_000, 2000, device="cuda")
        y = torch.randn(20_000, 3, device="cuda")

        fast = rd.CGReadoutLayer(2000, 3).to("cuda")
        slow = rd.CGReadoutLayer(2000, 3, gram_dtype=torch.float64).to("cuda")

        t_fast = _timed(lambda: fast.fit(X, y), "cuda")
        t_slow = _timed(lambda: slow.fit(X, y), "cuda")

        assert t_fast < t_slow, f"auto gram {t_fast:.3f}s vs fp64 gram {t_slow:.3f}s"
