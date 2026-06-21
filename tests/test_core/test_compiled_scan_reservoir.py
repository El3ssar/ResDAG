"""Compiled single-step kernel and scan-lowered recurrence (issue #257).

Three execution strategies for the reservoir time loop must agree numerically:

- the default Python loop (``compile_mode="loop"`` / ``"eager"``);
- the :func:`torch._higher_order_ops.scan` lowering (``compile_mode="scan"``),
  which expresses the recurrence as one ``combine_fn`` so ``torch.compile`` sees
  a single graph region instead of one node per timestep;
- :meth:`ESNModel.compile_reservoirs`, which wraps each ``cell.step`` in
  :func:`torch.compile` so the per-step launch overhead amortises across steps
  *without* unrolling the whole-sequence loop.

These checks are fully CPU-testable: the hard bar is ``eager == compiled`` and
``eager == scan`` within tolerance with no NaNs.  The CUDA-only / first-compile
latency acceptance numbers are exercised by the benchmark in ``benchmarks/``,
not here, and are noted as hardware-pending where they cannot run on CPU.  Sizes
are kept small (``N<=80``, ``T<=64``) so the compile tests do not time out.
"""

import sys

import pytest
import pytorch_symbolic as ps
import torch

from resdag import ESNModel
from resdag.layers import CGReadoutLayer, ESNLayer
from resdag.layers.reservoirs import base_reservoir as br
from resdag.layers.reservoirs.base_reservoir import COMPILE_MODES, _scan_available

COMPILE_SUPPORTED = torch.__version__ >= "2.0.0" and sys.version_info < (3, 15)

requires_compile = pytest.mark.skipif(not COMPILE_SUPPORTED, reason="torch.compile not supported")
requires_scan = pytest.mark.skipif(not _scan_available(), reason="scan HOP not available")

TOL = 1e-5


def _build_model(reservoir_size: int = 64, seed: int = 11) -> ESNModel:
    """A small feedback-only ESN model with a CG readout."""
    torch.manual_seed(seed)
    inp = ps.Input((32, 3))
    reservoir = ESNLayer(reservoir_size, feedback_size=3, spectral_radius=0.9, seed=seed)(inp)
    readout = CGReadoutLayer(reservoir_size, 3, name="output")(reservoir)
    return ESNModel(inp, readout)


# ---------------------------------------------------------------------------
# compile_mode plumbing
# ---------------------------------------------------------------------------


class TestCompileModePlumbing:
    """``compile_mode`` is validated and threaded into the layer."""

    @pytest.mark.parametrize("mode", COMPILE_MODES)
    def test_valid_modes_accepted(self, mode: str) -> None:
        """Every documented mode constructs a layer carrying that mode."""
        layer = ESNLayer(16, feedback_size=2, compile_mode=mode)
        assert layer.compile_mode == mode

    def test_invalid_mode_raises(self) -> None:
        """An unknown mode is rejected at construction with a ValueError."""
        with pytest.raises(ValueError, match="compile_mode"):
            ESNLayer(16, feedback_size=2, compile_mode="nope")

    def test_default_is_loop(self) -> None:
        """The historical Python loop stays the default."""
        assert ESNLayer(16, feedback_size=2).compile_mode == "loop"


# ---------------------------------------------------------------------------
# scan path correctness
# ---------------------------------------------------------------------------


@requires_scan
class TestScanCorrectness:
    """The scan lowering reproduces the Python loop within tolerance."""

    def test_feedback_only_matches_loop(self) -> None:
        """Scan output equals the eager loop for a feedback-only reservoir."""
        torch.manual_seed(0)
        fb = torch.randn(3, 50, 4)
        loop = ESNLayer(64, feedback_size=4, spectral_radius=0.9, seed=1).eval()
        scan = ESNLayer(
            64, feedback_size=4, spectral_radius=0.9, seed=1, compile_mode="scan"
        ).eval()
        with torch.no_grad():
            out_loop = loop(fb)
            out_scan = scan(fb)
        assert out_scan.shape == out_loop.shape
        assert not torch.isnan(out_scan).any()
        assert torch.allclose(out_loop, out_scan, atol=TOL)

    def test_driver_and_leak_match_loop(self) -> None:
        """Scan handles driving inputs and leaky integration identically."""
        torch.manual_seed(0)
        fb = torch.randn(2, 40, 4)
        driver = torch.randn(2, 40, 5)
        kw = dict(spectral_radius=0.9, seed=2, leak_rate=0.6)
        loop = ESNLayer(48, feedback_size=4, input_size=5, **kw).eval()
        scan = ESNLayer(48, feedback_size=4, input_size=5, compile_mode="scan", **kw).eval()
        with torch.no_grad():
            assert torch.allclose(loop(fb, driver), scan(fb, driver), atol=TOL)

    def test_final_state_matches_loop(self) -> None:
        """The state left in the buffer after a scan forward matches the loop."""
        torch.manual_seed(0)
        fb = torch.randn(2, 30, 3)
        loop = ESNLayer(32, feedback_size=3, spectral_radius=0.9, seed=3).eval()
        scan = ESNLayer(
            32, feedback_size=3, spectral_radius=0.9, seed=3, compile_mode="scan"
        ).eval()
        with torch.no_grad():
            loop(fb)
            scan(fb)
        assert torch.allclose(loop.state, scan.state, atol=TOL)

    def test_scan_under_torch_compile(self) -> None:
        """A scan layer wrapped in torch.compile still matches eager.

        The scan op is itself the lowered loop, so compiling the *cell.step*
        kernel under it must not change the result.
        """
        torch.manual_seed(0)
        fb = torch.randn(2, 24, 3)
        ref_layer = ESNLayer(
            48, feedback_size=3, spectral_radius=0.9, seed=4, compile_mode="scan"
        ).eval()
        with torch.no_grad():
            ref = ref_layer(fb)
        comp_layer = ESNLayer(
            48, feedback_size=3, spectral_radius=0.9, seed=4, compile_mode="scan"
        ).eval()
        comp_layer.cell.step = torch.compile(comp_layer.cell.step)  # type: ignore[method-assign]
        with torch.no_grad():
            got = comp_layer(fb)
        assert torch.allclose(ref, got, atol=1e-4)


class TestScanFallback:
    """``compile_mode="scan"`` degrades to the Python loop when scan is absent."""

    def test_fallback_matches_loop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With the scan HOP forced unavailable, scan-mode == loop-mode."""
        torch.manual_seed(0)
        fb = torch.randn(2, 30, 3)
        loop = ESNLayer(32, feedback_size=3, spectral_radius=0.9, seed=5).eval()
        with torch.no_grad():
            ref = loop(fb)

        monkeypatch.setattr(br, "_scan_available", lambda: False)
        scan = ESNLayer(
            32, feedback_size=3, spectral_radius=0.9, seed=5, compile_mode="scan"
        ).eval()
        with torch.no_grad():
            got = scan(fb)
        assert torch.allclose(ref, got, atol=TOL)

    def test_empty_sequence_does_not_dispatch_scan(self) -> None:
        """A length-0 sequence returns the empty-output contract, not a scan."""
        layer = ESNLayer(16, feedback_size=2, compile_mode="scan", seed=6).eval()
        empty = torch.zeros(2, 0, 2)
        with torch.no_grad():
            out = layer(empty)
        assert out.shape == (2, 0, 16)


# ---------------------------------------------------------------------------
# compile_reservoirs
# ---------------------------------------------------------------------------


@requires_compile
class TestCompileReservoirs:
    """``ESNModel.compile_reservoirs`` compiles cell.step and stays correct."""

    def test_returns_self(self) -> None:
        """The method returns the model so calls can chain."""
        model = _build_model()
        assert model.compile_reservoirs(mode="default") is model

    def test_compiles_every_reservoir_step(self) -> None:
        """Each reservoir's ``cell.step`` is rebound to a compiled callable."""
        model = _build_model()
        raw_steps = [m.cell.step for m in model.modules() if isinstance(m, br.BaseReservoirLayer)]
        model.compile_reservoirs(mode="default")
        new_steps = [m.cell.step for m in model.modules() if isinstance(m, br.BaseReservoirLayer)]
        assert len(new_steps) == len(raw_steps) >= 1
        for raw, new in zip(raw_steps, new_steps):
            assert new is not raw

    def test_forward_matches_eager(self) -> None:
        """A compiled-step forward equals the eager forward within tolerance."""
        x = torch.randn(1, 32, 3)
        eager = _build_model().eval()
        eager.reset_reservoirs()
        with torch.no_grad():
            base = eager(x)

        compiled = _build_model().eval()
        compiled.reset_reservoirs()
        compiled.compile_reservoirs(mode="default")
        with torch.no_grad():
            got = compiled(x)

        assert got.shape == base.shape
        assert not torch.isnan(got).any()
        assert torch.allclose(base, got, atol=1e-4)

    def test_forecast_matches_eager(self) -> None:
        """An end-to-end forecast equals eager after compile_reservoirs."""
        warm = torch.randn(1, 32, 3)
        eager = _build_model().eval()
        eager.warmup(warm)
        base = eager.forecast(warm, horizon=20)

        compiled = _build_model().eval()
        compiled.compile_reservoirs(mode="default")
        compiled.warmup(warm)
        got = compiled.forecast(warm, horizon=20)

        assert got.shape == base.shape
        assert not torch.isnan(got).any()
        assert torch.allclose(base, got, atol=1e-4)

    def test_default_mode_is_reduce_overhead(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When ``mode`` is omitted, reduce-overhead is requested."""
        captured: dict[str, object] = {}

        def fake_compile(fn: object, **kwargs: object) -> object:
            captured.update(kwargs)
            return fn

        monkeypatch.setattr(torch, "compile", fake_compile)
        _build_model().compile_reservoirs()
        assert captured.get("mode") == "reduce-overhead"

    def test_explicit_kwargs_override_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """User kwargs win over the reduce-overhead default."""
        captured: dict[str, object] = {}

        def fake_compile(fn: object, **kwargs: object) -> object:
            captured.update(kwargs)
            return fn

        monkeypatch.setattr(torch, "compile", fake_compile)
        _build_model().compile_reservoirs(mode="default", fullgraph=True)
        assert captured.get("mode") == "default"
        assert captured.get("fullgraph") is True
