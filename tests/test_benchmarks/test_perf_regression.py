"""Performance-regression gate against committed targets.

This module is the automated counterpart to the cross-library benchmark harness
in ``benchmarks/`` (rendered into ``benchmarks/results/REPORT.md``). The headline
numbers in that report depend on external libraries (reservoirpy,
ReservoirComputing.jl) and on specific hardware, so they cannot gate CI. What
*can* gate CI are the **internal** speedups the fast paths were built to deliver:
each fast path is benchmarked against resdag's own naive path on the same
machine, and the ratio must clear a conservative, hardware-portable floor read
from ``benchmarks/targets.json``.

Three ratios are gated (slugs / tickets):

* ``flat_forecast_vs_graph_reexec`` — the flat single-step forecast engine
  (``core-forecast-flat-inference-engine``, #254) vs a per-step
  pytorch_symbolic graph walk.
* ``vectorized_ngrc_vs_stepwise`` — the vectorized NG-RC feature pass
  (``reservoir-vectorize-ngrc``, #255) vs a Python loop over the single-step
  cell.
* ``fast_spectral_radius_vs_dense_eigvals`` — the power-iteration spectral-radius
  estimate (``init-fast-spectral-radius``, #185) vs the dense ``eigvals``
  eigendecomposition.

All three run on CPU with float32 and require no external library, so they are
reproducible on any machine. They live behind the ``benchmark`` pytest marker
(deselected by default; run with ``pytest -m benchmark``) because they are timing
assertions, and each also carries a built-in correctness check (the fast and
slow paths must agree numerically) so a "fast" path that cheats by computing
something different cannot pass.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import pytest
import torch

import resdag as rd
from resdag.init.topology.base import _dense_eigvals_radius, _power_iteration_radius
from resdag.layers.cells.ngrc_cell import NGCell

# Repo-root-relative path to the committed targets file. This test file lives at
# ``<repo>/tests/test_benchmarks/test_perf_regression.py``; targets.json lives at
# ``<repo>/benchmarks/targets.json``.
_TARGETS_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "targets.json"


def _load_targets() -> dict:
    """Read and parse the committed ``benchmarks/targets.json``."""
    return json.loads(_TARGETS_PATH.read_text())


def _enforced_floor(targets: dict, ratio_key: str) -> float:
    """Return the hard CPU floor for one internal-ratio cell."""
    return float(targets["internal_ratios"][ratio_key]["enforced_min_ratio"])


def _timed(fn: Callable[[], object], warmup: int = 2, reps: int = 7) -> float:
    """Best-of-``reps`` wall time of ``fn`` (CPU), discarding warmup runs.

    The **minimum** is returned, not the mean/median: for a CPU microbenchmark
    the fastest run is the one least disturbed by transient interference (other
    processes, thread oversubscription, GC), so best-of-N isolates the intrinsic
    algorithmic cost.

    Parameters
    ----------
    fn : callable
        Zero-argument callable to time.
    warmup : int, optional
        Discarded warmup iterations (lets lazy allocation / caches settle).
    reps : int, optional
        Timed iterations; the fastest is returned.

    Returns
    -------
    float
        Minimum per-call wall-clock time in seconds.
    """
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def _best_ratio(
    slow: Callable[[], object],
    fast: Callable[[], object],
    *,
    warmup: int = 3,
    reps: int = 9,
) -> tuple[float, float, float]:
    """Best ``slow / fast`` ratio from *interleaved* A/B timing.

    Each rep times ``slow`` then ``fast`` back-to-back, so within a rep both
    paths see the same machine load. The returned ratio is the **maximum** ratio
    over all reps, paired with the ``slow``/``fast`` times from that winning rep.

    Interleaving + best-ratio is what makes the gate robust on a contended CI
    runner. Timing each path in its own block and dividing the two best-of-N
    minima (the obvious approach) is biased by load *asymmetry*: a transient
    spike that lands entirely inside one path's measurement window — but not the
    other's — skews the ratio either way and can flake the assertion. Comparing
    the two paths under the *same* instantaneous load on every rep, then keeping
    the cleanest rep (the one least disturbed for both), isolates the intrinsic
    algorithmic speedup and is monotone in it.

    Parameters
    ----------
    slow, fast : callable
        Zero-argument callables for the naive and fast paths respectively.
    warmup : int, optional
        Discarded warmup iterations per path (lets caches/allocation settle).
    reps : int, optional
        Interleaved timed reps; the best ratio is returned.

    Returns
    -------
    tuple of float
        ``(best_ratio, slow_time, fast_time)`` where the two times are from the
        rep that produced ``best_ratio`` (both in seconds).
    """
    for _ in range(warmup):
        slow()
        fast()
    best_ratio = 0.0
    best_slow = 0.0
    best_fast = 0.0
    for _ in range(reps):
        s0 = time.perf_counter()
        slow()
        t_slow = time.perf_counter() - s0
        f0 = time.perf_counter()
        fast()
        t_fast = time.perf_counter() - f0
        ratio = t_slow / t_fast if t_fast > 0 else 0.0
        if ratio > best_ratio:
            best_ratio, best_slow, best_fast = ratio, t_slow, t_fast
    return best_ratio, best_slow, best_fast


@pytest.mark.benchmark
class TestInternalSpeedupTargets:
    """Each fast path must clear its committed, hardware-portable CPU floor.

    The floors come from ``benchmarks/targets.json`` so the gate and the
    documented target never drift apart: bump the number there to retune the
    gate. Every test also asserts the fast and slow paths agree numerically, so
    the ratio is only meaningful work compared against meaningful work.
    """

    # ---- #254: flat forecast engine vs per-step graph re-execution ---------

    def _build_forecast_model(self, reservoir_size: int = 20) -> rd.core.ESNModel:
        # A deliberately *small* reservoir. The flat engine's win over per-step
        # graph re-execution is the elimination of the fixed per-step
        # pytorch_symbolic graph-walk overhead, which is independent of reservoir
        # size; shrinking the reservoir shrinks the per-step *compute* so that
        # overhead dominates, which isolates exactly the cost #254 removed and
        # makes the gated ratio hardware-portable (it does not depend on how fast
        # the CPU does the matmul). A large reservoir would dilute the ratio with
        # compute and make the floor flake on slower boxes.
        torch.manual_seed(0)
        inp = rd.core.reservoir_input(3)
        states = rd.layers.ESNLayer(
            reservoir_size=reservoir_size, feedback_size=3, spectral_radius=0.9
        )(inp)
        out = rd.CGReadoutLayer(reservoir_size, 3, name="output")(states)
        return rd.core.ESNModel(inp, out)

    @staticmethod
    def _graph_forecast(
        model: rd.core.ESNModel, warmup: torch.Tensor, horizon: int
    ) -> torch.Tensor:
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

    def test_flat_forecast_beats_graph_reexecution(self) -> None:
        targets = _load_targets()
        floor = _enforced_floor(targets, "flat_forecast_vs_graph_reexec")

        model = self._build_forecast_model()
        warmup = torch.randn(1, 50, 3, dtype=torch.float32)
        horizon = 2000

        # Correctness: the flat engine must reproduce the graph-walk trajectory.
        graph_out = self._graph_forecast(model, warmup, horizon)
        model.reset_reservoirs()
        with torch.no_grad():
            flat_out = model.forecast(warmup, horizon=horizon)
        torch.testing.assert_close(flat_out, graph_out, rtol=1e-4, atol=1e-4)

        speedup, t_graph, t_flat = _best_ratio(
            lambda: self._graph_forecast(model, warmup, horizon),
            lambda: (model.reset_reservoirs(), model.forecast(warmup, horizon=horizon)),
            reps=15,
        )
        assert speedup >= floor, (
            f"flat forecast only {speedup:.2f}x the graph path "
            f"(flat={t_flat * 1e3:.1f}ms, graph={t_graph * 1e3:.1f}ms); "
            f"expected >= {floor}x at H={horizon}"
        )

    # ---- #255: vectorized NG-RC vs per-step scan ---------------------------

    @staticmethod
    def _ngrc_stepwise(cell: NGCell, x: torch.Tensor) -> torch.Tensor:
        """Reference path: scan the single-step cell over the sequence."""
        outs: list[torch.Tensor] = []
        state = cell.init_state(x.shape[0], x.device, x.dtype)
        with torch.no_grad():
            for t in range(x.shape[1]):
                feat, state = cell([x[:, t, :]], state)
                outs.append(feat)
        return torch.stack(outs, dim=1)

    def test_vectorized_ngrc_beats_stepwise(self) -> None:
        targets = _load_targets()
        floor = _enforced_floor(targets, "vectorized_ngrc_vs_stepwise")

        torch.manual_seed(0)
        cell = NGCell(input_dim=6, k=3, s=1, p=2)
        x = torch.randn(4, 400, 6)

        # Correctness: the vectorized pass must match the per-step scan exactly.
        with torch.no_grad():
            state = cell.init_state(x.shape[0], x.device, x.dtype)
            vec_out, _ = cell.forward_sequence(x, state)
        step_out = self._ngrc_stepwise(cell, x)
        torch.testing.assert_close(vec_out, step_out, rtol=0.0, atol=0.0)

        def run_vectorized() -> None:
            state = cell.init_state(x.shape[0], x.device, x.dtype)
            with torch.no_grad():
                cell.forward_sequence(x, state)

        speedup, t_step, t_vec = _best_ratio(
            lambda: self._ngrc_stepwise(cell, x),
            run_vectorized,
        )
        assert speedup >= floor, (
            f"vectorized NG-RC only {speedup:.2f}x the step-by-step scan "
            f"(vec={t_vec * 1e3:.2f}ms, step={t_step * 1e3:.2f}ms); "
            f"expected >= {floor}x"
        )

    # ---- #185: fast spectral radius vs dense eigvals -----------------------

    def test_fast_spectral_radius_beats_dense_eigvals(self) -> None:
        targets = _load_targets()
        floor = _enforced_floor(targets, "fast_spectral_radius_vs_dense_eigvals")

        torch.manual_seed(0)
        # A dense matrix routes estimate_spectral_radius through the pure-torch
        # power-iteration path (no scipy / external dependency), so the ratio is
        # fully reproducible. n is large enough that the O(n^3) dense eigvals
        # dominates the O(n^2)-per-step power iteration.
        n = 1500
        matrix = torch.randn(n, n) * 0.1

        # Correctness: the fast estimate must agree with the exact radius.
        r_fast = _power_iteration_radius(matrix)
        r_dense = _dense_eigvals_radius(matrix)
        assert (
            abs(r_fast - r_dense) <= 1e-2 * r_dense
        ), f"fast spectral radius {r_fast:.4f} disagrees with dense {r_dense:.4f}"

        speedup, t_dense, t_fast = _best_ratio(
            lambda: _dense_eigvals_radius(matrix),
            lambda: _power_iteration_radius(matrix),
            warmup=2,
            reps=5,
        )
        assert speedup >= floor, (
            f"fast spectral radius only {speedup:.2f}x dense eigvals "
            f"(fast={t_fast * 1e3:.1f}ms, dense={t_dense * 1e3:.1f}ms); "
            f"expected >= {floor}x at n={n}"
        )


def test_targets_json_is_well_formed() -> None:
    """``benchmarks/targets.json`` parses and defines every gated ratio.

    This is a plain (non-benchmark) test so the normal lane catches a malformed
    or out-of-sync targets file even though the timing assertions are deselected.
    """
    targets = _load_targets()
    assert "internal_ratios" in targets
    required = {
        "flat_forecast_vs_graph_reexec",
        "vectorized_ngrc_vs_stepwise",
        "fast_spectral_radius_vs_dense_eigvals",
    }
    cells = targets["internal_ratios"]
    assert required.issubset(cells), f"missing internal ratios: {required - set(cells)}"
    for key in required:
        cell = cells[key]
        assert float(cell["enforced_min_ratio"]) > 1.0, f"{key} floor must exceed 1x"
        assert "ticket" in cell and "metric" in cell, f"{key} missing provenance fields"
    # The external floors are informational; just confirm they survive a round
    # trip and stay clearly labelled as non-gating.
    assert "external_floors" in targets
    assert targets["external_floors"]["baseline_library"] == "reservoirpy"
