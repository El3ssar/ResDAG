"""Premade Liquid State Machine factory (resdag.models.lsm).

Pins down the ``lsm`` factory: architecture
(``Input -> LIFReservoir -> Concatenate(Input, trace) -> Readout``), forward
shapes, hyperparameter plumbing (tau_mem / spectral_radius / synaptic_current
reach the LIF cell), a full driven-task train that beats a memoryless baseline
(the "it works" proof for an input-driven reservoir), and save_full / load_full.

Imports are from the concrete module files: the public ``__init__`` exports are
wired separately.
"""

import numpy as np
import torch

from resdag.core import ESNModel
from resdag.layers.cells.lif_cell import LIFCell
from resdag.models.lsm import lsm
from resdag.training import ESNTrainer


def _find_cell(model: ESNModel, cell_type: type) -> object:
    """Return the first reservoir cell of ``cell_type`` inside ``model``."""
    for module in model.modules():
        if isinstance(module, cell_type):
            return module
    raise AssertionError(f"no {cell_type.__name__} found in model")


class TestLSMFactory:
    """The lsm() premade factory."""

    def test_basic_instantiation_and_architecture(self) -> None:
        """Model wires a LIF reservoir, a concatenation, and a ridge readout."""
        model = lsm(reservoir_size=64, feedback_size=2, output_size=3, seed=0)
        names = [name for name, _ in model.named_modules()]
        assert any("LIFReservoirLayer" in name for name in names)
        assert any("Concatenate" in name for name in names)
        assert any("RidgeReadoutLayer" in name for name in names)

    def test_forward_shape(self) -> None:
        model = lsm(reservoir_size=64, feedback_size=2, output_size=3, seed=0)
        out = model(torch.randn(4, 20, 2))
        assert out.shape == (4, 20, 3)

    def test_hyperparameters_reach_cell(self) -> None:
        model = lsm(
            reservoir_size=32,
            feedback_size=2,
            output_size=2,
            tau_mem=25.0,
            v_threshold=0.7,
            synaptic_current=True,
            seed=0,
        )
        cell = _find_cell(model, LIFCell)
        assert cell.tau_mem == 25.0  # type: ignore[attr-defined]
        assert cell.v_threshold == 0.7  # type: ignore[attr-defined]
        assert cell.synaptic_current is True  # type: ignore[attr-defined]

    def test_spectral_radius_reaches_cell(self) -> None:
        from resdag.init.topology import estimate_spectral_radius

        model = lsm(reservoir_size=128, feedback_size=2, output_size=2, spectral_radius=0.8, seed=1)
        cell = _find_cell(model, LIFCell)
        radius = estimate_spectral_radius(cell.weight_rec.detach())  # type: ignore[attr-defined]
        assert radius == np.float64(radius)  # finite
        assert abs(radius - 0.8) < 5e-2

    def test_driving_input_model(self) -> None:
        model = lsm(reservoir_size=40, feedback_size=3, output_size=3, input_size=2, seed=0)
        out = model(torch.randn(2, 12, 3), torch.randn(2, 12, 2))
        assert out.shape == (2, 12, 3)

    def test_output_finite(self) -> None:
        model = lsm(reservoir_size=64, feedback_size=2, output_size=2, seed=0)
        out = model(torch.randn(2, 30, 2))
        assert torch.isfinite(out).all()


class TestLSMDrivenTask:
    """The it-works proof: a driven fading-memory task beats a memoryless baseline."""

    def test_beats_memoryless_baseline(self) -> None:
        """LSM readout on a fading-memory task (tanh of a moving average of the
        drive) beats an instantaneous-linear-on-input baseline by a clear margin.
        This is the input-driven-reservoir contract — not autoregression."""
        torch.manual_seed(0)
        n = 6000
        raw = torch.randn(1, n, 1)
        smooth_kernel = torch.ones(1, 1, 10) / 10.0
        u = torch.nn.functional.conv1d(raw.transpose(1, 2), smooth_kernel, padding=5).transpose(
            1, 2
        )[:, :n]
        u = (u - u.mean()) / u.std()

        win = 6
        avg_kernel = torch.ones(1, 1, win) / win
        moving = torch.nn.functional.conv1d(
            u.transpose(1, 2), avg_kernel, padding=win - 1
        ).transpose(1, 2)[:, :n]
        y = torch.tanh(2.0 * moving)

        warm, tr_end = 400, 4000
        model = lsm(
            reservoir_size=500,
            feedback_size=1,
            output_size=1,
            tau_mem=25.0,
            spectral_radius=1.0,
            trace_decay=0.8,
            v_threshold=0.5,
            bias_scaling=0.6,
            feedback_initializer=("normal", {"input_scaling": 4.0}),
            readout_alpha=1e-2,
            seed=1,
        )
        trainer = ESNTrainer(model)
        trainer.fit(
            warmup_inputs=(u[:, :warm],),
            train_inputs=(u[:, warm:tr_end],),
            targets={"output": y[:, warm:tr_end]},
        )
        model.reset_reservoirs()
        _ = model(u[:, :tr_end])
        pred = model(u[:, tr_end:])
        tgt = y[:, tr_end:]
        lsm_mse = (pred - tgt).pow(2).mean().item()
        base_mse = (tgt - tgt.mean()).pow(2).mean().item()
        lsm_r2 = 1.0 - lsm_mse / base_mse

        # Memoryless linear-on-input baseline.
        uu = u[0, warm:tr_end, 0].numpy()
        yy = y[0, warm:tr_end, 0].numpy()
        design = np.stack([np.ones_like(uu), uu], axis=1)
        weights = np.linalg.lstsq(design, yy, rcond=None)[0]
        ute = u[0, tr_end:, 0].numpy()
        yte = y[0, tr_end:, 0].numpy()
        lin_pred = np.stack([np.ones_like(ute), ute], axis=1) @ weights
        lin_r2 = 1.0 - ((lin_pred - yte) ** 2).mean() / ((yte - yte.mean()) ** 2).mean()

        # The liquid's fading memory adds real predictive value over a memoryless map.
        assert lsm_r2 > 0.5
        assert lsm_r2 > lin_r2 + 0.05


class TestLSMPersistence:
    """save_full / load_full and the packed-state pipeline round-trip."""

    def test_save_full_load_full_roundtrip(self, tmp_path: object) -> None:
        model = lsm(
            reservoir_size=40,
            feedback_size=2,
            output_size=2,
            tau_mem=15.0,
            spectral_radius=0.9,
            seed=3,
        )
        # Fit so the readout carries real weights to round-trip.
        torch.manual_seed(0)
        u = torch.randn(1, 600, 2)
        y = torch.tanh(u.roll(2, dims=1))
        trainer = ESNTrainer(model)
        trainer.fit(
            warmup_inputs=(u[:, :100],),
            train_inputs=(u[:, 100:500],),
            targets={"output": y[:, 100:500]},
        )
        probe = torch.randn(1, 30, 2)
        model.reset_reservoirs()
        out1 = model(probe)

        path = str(tmp_path) + "/lsm_full.pt"  # type: ignore[operator]
        model.save_full(path)
        restored = ESNModel.load_full(path)
        restored.reset_reservoirs()
        out2 = restored(probe)

        assert torch.allclose(out1, out2, atol=1e-5)
        cell = _find_cell(restored, LIFCell)
        assert cell.tau_mem == 15.0  # type: ignore[attr-defined]
