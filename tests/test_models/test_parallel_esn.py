"""Parallel local-state reservoir contracts (resdag.models.parallel_esn).

Pins down the Pathak et al. (2018) parallel / local-state architecture: a single
shared-weight reservoir advanced *batched* over overlapping spatial patches, a
single weight-tied readout predicting each patch's core region, and a spatial-
autoregressive closed-loop forecast that stitches per-patch cores into the full
field and re-windows every step.

Covered here:

* **Geometry** — periodic patch-gather index (wrap across the seam), windowing
  that folds the patch axis into the reservoir batch dimension, and
  stitch/window round-tripping the core regions.
* **Wiring** — exactly one shared reservoir cell and one tied readout; the
  reservoir input width is ``core_width + 2*halo`` and the readout out-width is
  ``core_width``; construction validation on the size arguments.
* **Batched advance** — the ``P`` patches advance as ONE reservoir call over a
  ``batch*P`` leading axis (not a Python loop over patches), and the per-step
  cost is roughly flat per patch as ``P`` grows (fixed per-patch work).
* **Forecasting** — the parallel model forecasts Kuramoto-Sivashinsky with a
  valid-prediction-time clearly better than a single *global* reservoir of
  comparable total size on the same data.
* **Persistence** — ``save_full`` / ``load_full`` round-trips the fitted model
  and reproduces its forecast bit-for-bit.
"""

import pytest
import torch

from resdag import kuramoto_sivashinsky
from resdag.layers.cells import ESNCell
from resdag.layers.readouts import ReadoutLayer
from resdag.metrics import valid_prediction_time
from resdag.models import classic_esn, parallel_esn
from resdag.models.parallel_esn import ParallelESNModel, _patch_gather_index
from resdag.training import ESNTrainer


def _parallel(**kwargs) -> ParallelESNModel:
    """Build a parallel ESN in float64 (the KS fixture / test tensors are float64).

    The factory builds under the ambient default dtype (float32); ``.double()``
    casts the reservoir + readout weights in place — the ``patch_index`` buffer
    stays ``int64`` — so the model matches the float64 inputs the tests feed it.
    """
    return parallel_esn(**kwargs).double()


# --------------------------------------------------------------------------
# Shared KS fixture — one trajectory reused across the (slow) forecast tests.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ks_data() -> torch.Tensor:
    """A reproducible float64 Kuramoto-Sivashinsky trajectory, (1, T, 64)."""
    return kuramoto_sivashinsky(6000, n_grid=64, dtype=torch.float64, seed=1)


def _split(data: torch.Tensor) -> dict[str, torch.Tensor]:
    """Split a KS trajectory into warmup / train / test windows."""
    warmup_len, train_len, test_len = 400, 4000, 800
    a = warmup_len
    b = warmup_len + train_len
    return {
        "warmup": data[:, :a],
        "train_in": data[:, a:b],
        "train_tgt": data[:, a + 1 : b + 1],
        "test_warm": data[:, b : b + warmup_len],
        "test_truth": data[:, b + warmup_len : b + warmup_len + test_len],
        "test_len": test_len,
    }


# --------------------------------------------------------------------------
# Patch geometry: periodic gather index.
# --------------------------------------------------------------------------


class TestPatchGatherIndex:
    """The periodic per-patch gather index that windows a full field."""

    def test_shape_and_partition(self) -> None:
        """P = D/w rows, each of width w + 2g."""
        idx = _patch_gather_index(input_dim=64, core_width=4, halo=6)
        assert idx.shape == (16, 4 + 2 * 6)

    def test_periodic_wrap_across_seam(self) -> None:
        """Boundary patches read across the periodic seam (indices wrap mod D)."""
        idx = _patch_gather_index(input_dim=8, core_width=2, halo=1)
        # Patch 0 core is [0, 1]; with halo 1 it reads [-1, 0, 1, 2] -> wrap -> [7,0,1,2].
        assert idx[0].tolist() == [7, 0, 1, 2]
        # Last patch core is [6, 7]; reads [5, 6, 7, 8] -> wrap -> [5, 6, 7, 0].
        assert idx[-1].tolist() == [5, 6, 7, 0]
        assert int(idx.max()) < 8 and int(idx.min()) >= 0

    def test_cores_tile_the_domain(self) -> None:
        """Concatenating the per-patch core columns covers every cell exactly once."""
        idx = _patch_gather_index(input_dim=12, core_width=3, halo=2)
        # Core columns sit at offsets [halo, halo + core_width) inside each window.
        cores = idx[:, 2 : 2 + 3].reshape(-1)
        assert sorted(cores.tolist()) == list(range(12))

    def test_rejects_non_divisor_core_width(self) -> None:
        with pytest.raises(ValueError, match="divide"):
            _patch_gather_index(input_dim=10, core_width=3, halo=1)

    def test_rejects_bad_sizes(self) -> None:
        with pytest.raises(ValueError, match="core_width"):
            _patch_gather_index(input_dim=8, core_width=0, halo=1)
        with pytest.raises(ValueError, match="halo"):
            _patch_gather_index(input_dim=8, core_width=2, halo=-1)


# --------------------------------------------------------------------------
# Construction / wiring: one shared reservoir, one tied readout.
# --------------------------------------------------------------------------


class TestConstruction:
    """Factory sizing, the shared-weight invariant, and validation."""

    def test_single_shared_reservoir_and_readout(self) -> None:
        """Exactly one reservoir cell and one readout — weights shared across patches."""
        model = _parallel(input_dim=64, core_width=4, halo=6, reservoir_size=100, seed=0)
        cells = [m for m in model.modules() if isinstance(m, ESNCell)]
        readouts = [m for m in model.modules() if isinstance(m, ReadoutLayer)]
        assert len(cells) == 1
        assert len(readouts) == 1

    def test_reservoir_input_and_readout_out_widths(self) -> None:
        """Reservoir feedback width = w + 2g; readout out width = w; in = N."""
        model = _parallel(input_dim=64, core_width=4, halo=6, reservoir_size=100, seed=0)
        assert model.reservoir.feedback_size == 4 + 2 * 6
        assert model.readout.out_features == 4
        assert model.readout.in_features == 100

    def test_n_patches(self) -> None:
        model = _parallel(input_dim=64, core_width=8, halo=2, reservoir_size=50, seed=0)
        assert model.n_patches == 8

    def test_rejects_non_divisor_core_width(self) -> None:
        with pytest.raises(ValueError, match="divide"):
            parallel_esn(input_dim=10, core_width=3, halo=1, reservoir_size=20)

    def test_rejects_negative_halo(self) -> None:
        with pytest.raises(ValueError, match="halo"):
            parallel_esn(input_dim=8, core_width=2, halo=-1, reservoir_size=20)

    def test_direct_ctor_validates_geometry(self) -> None:
        """ParallelESNModel rejects a reservoir/readout whose widths disagree with geometry."""
        from resdag.layers import ESNLayer, RidgeReadoutLayer

        # feedback_size must equal core_width + 2*halo.
        bad_res = ESNLayer(reservoir_size=50, feedback_size=999)
        readout = RidgeReadoutLayer(in_features=50, out_features=4, name="output")
        with pytest.raises(ValueError, match="feedback_size"):
            ParallelESNModel(bad_res, readout, input_dim=64, core_width=4, halo=6)


# --------------------------------------------------------------------------
# Windowing / stitching: fold patches into the batch axis; round-trip cores.
# --------------------------------------------------------------------------


class TestWindowStitch:
    """window() folds patches into the reservoir batch axis; stitch() is its core-inverse."""

    def test_window_folds_patches_into_batch(self) -> None:
        """window (B, T, D) -> (B*P, T, w+2g): patch axis becomes the leading batch axis."""
        model = _parallel(input_dim=64, core_width=4, halo=6, reservoir_size=100, seed=0)
        series = torch.randn(2, 10, 64, dtype=torch.float64)
        win = model.window(series)
        assert win.shape == (2 * 16, 10, 4 + 2 * 6)

    def test_window_stitch_round_trip_on_cores(self) -> None:
        """Windowing then stitching the CORE columns recovers the original field."""
        model = _parallel(input_dim=32, core_width=4, halo=3, reservoir_size=50, seed=0)
        series = torch.randn(3, 5, 32, dtype=torch.float64)
        win = model.window(series)  # (3*8, 5, 4+6)
        # Take the last timestep's core columns for every patch and stitch.
        b = series.shape[0]
        cores = win[:, -1, model.halo : model.halo + model.core_width]  # (3*8, 4)
        cores = cores.reshape(b, model.n_patches, model.core_width)
        stitched = model.stitch(cores)  # (3, 32)
        assert torch.allclose(stitched, series[:, -1, :])

    def test_patches_advance_as_one_batched_op(self) -> None:
        """The reservoir is called ONCE over a batch*P axis, not looped per patch."""
        model = _parallel(input_dim=64, core_width=4, halo=6, reservoir_size=80, seed=0)
        win = model.window(torch.randn(2, 7, 64, dtype=torch.float64))
        states = model.reservoir(win)
        # A single (batch*P, T, N) states tensor is the batched-advance signature.
        assert states.shape == (2 * 16, 7, 80)


# --------------------------------------------------------------------------
# Batched-scaling: per-step cost is ~flat per patch as P grows.
# --------------------------------------------------------------------------


class TestScaling:
    """Growing P (via D) with fixed per-patch work keeps per-patch step cost ~flat."""

    def test_step_time_roughly_flat_per_patch(self) -> None:
        """ms/step/patch drops as P grows — the batched op amortises over patches.

        Timing-based, so the bounds are deliberately generous (the machine may be
        loaded): the falsifiable claim is that ``8x`` the patches does **not**
        cost ``~8x`` the wall time — which a Python loop over patches would.  The
        *structural* proof that patches advance in one op lives in
        :meth:`TestWindowStitch.test_patches_advance_as_one_batched_op`; this only
        guards the scaling that op buys.
        """
        import time

        def per_step_ms(input_dim: int) -> float:
            model = _parallel(input_dim=input_dim, core_width=4, halo=6, reservoir_size=150, seed=0)
            cur = torch.randn(1, input_dim, dtype=torch.float64)
            with torch.no_grad():
                # A couple of warmup passes so allocator / kernel caches are hot.
                for _ in range(3):
                    model.reservoir(model.window(cur.unsqueeze(1)))
                # Best-of-several timing to shed one-off scheduling noise.
                best = float("inf")
                for _ in range(5):
                    t0 = time.perf_counter()
                    for _ in range(20):
                        model.reservoir(model.window(cur.unsqueeze(1)))
                    best = min(best, (time.perf_counter() - t0) / 20 * 1e3)
                return best

        small_d, large_d = 32, 256  # P: 8 -> 64 (8x more patches)
        t_small = per_step_ms(small_d)
        t_large = per_step_ms(large_d)
        p_small, p_large = small_d // 4, large_d // 4
        # 8x the patches must NOT cost ~8x the time (a python loop over patches
        # would); a batched op keeps the growth far sublinear.  5x is a loose
        # ceiling that still separates "batched" from "looped".
        assert t_large < 5.0 * t_small
        # And the per-patch cost falls as P grows (batched amortisation).
        assert (t_large / p_large) < (t_small / p_small)


# --------------------------------------------------------------------------
# Forecasting: parallel beats a comparable single global reservoir on VPT.
# --------------------------------------------------------------------------


class TestForecast:
    """Closed-loop KS forecast; parallel VPT > a comparable global reservoir."""

    def test_forecast_shape_and_finite(self, ks_data: torch.Tensor) -> None:
        s = _split(ks_data)
        model = _parallel(
            input_dim=64,
            core_width=4,
            halo=6,
            reservoir_size=300,
            spectral_radius=0.6,
            readout_alpha=1e-4,
            seed=0,
        )
        model.fit(s["warmup"], s["train_in"], s["train_tgt"])
        pred = model.forecast(s["test_warm"], horizon=200)
        assert pred.shape == (1, 200, 64)
        assert torch.isfinite(pred).all()

    def test_return_warmup_prepends(self, ks_data: torch.Tensor) -> None:
        s = _split(ks_data)
        model = _parallel(
            input_dim=64,
            core_width=4,
            halo=6,
            reservoir_size=200,
            spectral_radius=0.6,
            readout_alpha=1e-4,
            seed=0,
        )
        model.fit(s["warmup"], s["train_in"], s["train_tgt"])
        w = s["test_warm"]
        full = model.forecast(w, horizon=100, return_warmup=True)
        assert full.shape == (1, w.shape[1] + 100, 64)
        assert torch.allclose(full[:, : w.shape[1], :], w)

    def test_forecast_requires_fitted_readout(self) -> None:
        model = _parallel(input_dim=64, core_width=4, halo=6, reservoir_size=50, seed=0)
        with pytest.raises(ValueError, match="fitted"):
            model.forecast(torch.randn(1, 50, 64, dtype=torch.float64), horizon=10)

    def test_parallel_beats_comparable_global_reservoir(self, ks_data: torch.Tensor) -> None:
        """Parallel VPT clearly exceeds a single global reservoir of comparable size.

        A single ``D``-wide global reservoir cannot capture KS's high-dimensional
        chaos at ``D=64`` (its VPT collapses to ~1 step regardless of ``N``),
        while the parallel local-state model holds the attractor for a couple of
        Lyapunov times — the whole point of Pathak et al. (2018).
        """
        s = _split(ks_data)

        # Parallel: P=16 patches x N=300 units, with the factory's tuned defaults
        # (input_scaling=0.1 keeps the tanh reservoir out of saturation).
        pmodel = _parallel(input_dim=64, core_width=4, halo=6, reservoir_size=300, seed=0)
        pmodel.fit(s["warmup"], s["train_in"], s["train_tgt"])
        pred_p = pmodel.forecast(s["test_warm"], horizon=s["test_len"])
        vpt_p = valid_prediction_time(pred_p, s["test_truth"])

        # Global: a single wide reservoir (larger than any *one* patch's reservoir)
        # trained on the same data.  N=2000 is generous and tractable; a global
        # reservoir simply cannot forecast D=64 KS, so a bigger N does not help.
        gmodel = classic_esn(
            reservoir_size=2000,
            feedback_size=64,
            output_size=64,
            spectral_radius=0.6,
            readout_alpha=1e-4,
            seed=0,
        ).double()
        trainer = ESNTrainer(gmodel)
        trainer.fit(
            warmup_inputs=(s["warmup"],),
            train_inputs=(s["train_in"],),
            targets={"output": s["train_tgt"]},
        )
        pred_g = gmodel.forecast(s["test_warm"], horizon=s["test_len"])
        vpt_g = valid_prediction_time(pred_g, s["test_truth"])

        # The parallel local-state model forecasts markedly longer than the single
        # global reservoir on the same data.
        assert vpt_p > vpt_g
        assert vpt_p >= 5 * max(vpt_g, 1)
        # And it holds the KS attractor for a couple of Lyapunov times
        # (~93 steps/Lt on the L=22 domain); >= 100 steps is a safe floor well
        # below the ~244 the tuned defaults reach on this split.
        assert vpt_p >= 100


# --------------------------------------------------------------------------
# Persistence: save_full / load_full round-trip.
# --------------------------------------------------------------------------


class TestPersistence:
    """save_full / load_full reconstructs a fitted model and reproduces its forecast."""

    def test_save_full_load_full_round_trip(self, ks_data: torch.Tensor, tmp_path) -> None:
        s = _split(ks_data)
        model = _parallel(
            input_dim=64,
            core_width=4,
            halo=6,
            reservoir_size=200,
            spectral_radius=0.6,
            readout_alpha=1e-4,
            seed=0,
        )
        model.fit(s["warmup"], s["train_in"], s["train_tgt"])
        pred_before = model.forecast(s["test_warm"], horizon=150)

        path = tmp_path / "parallel.pt"
        model.save_full(path, note="ks")
        restored, meta = ParallelESNModel.load_full(path, return_metadata=True)

        assert meta["note"] == "ks"
        assert restored.input_dim == 64
        assert restored.core_width == 4
        assert restored.halo == 6
        assert restored.readout.is_fitted

        pred_after = restored.forecast(s["test_warm"], horizon=150)
        assert torch.allclose(pred_before, pred_after, atol=1e-10)

    def test_load_full_rejects_non_parallel_payload(self, tmp_path) -> None:
        path = tmp_path / "junk.pt"
        torch.save({"not_a_model": 1}, path)
        with pytest.raises(ValueError, match="ParallelESNModel"):
            ParallelESNModel.load_full(path)
