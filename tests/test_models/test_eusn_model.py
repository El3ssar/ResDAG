"""Premade Euler State Network factory (resdag.models.eusn).

Pins down the ``eusn`` factory: architecture
(``Input -> EuSN Reservoir -> Concatenate(Input, Reservoir) -> Readout``),
forward shapes, hyperparameter plumbing (``epsilon`` / ``gamma`` / ``solver``
reach the EuSN cell), a full train + forecast round-trip through the shared
pipeline, and save_full / load_full.

Imports are from the concrete module files: the public ``__init__`` exports are
wired separately.
"""

import pytest
import torch

from resdag.core import ESNModel
from resdag.layers.cells.eusn_cell import EuSNCell
from resdag.models.eusn import eusn
from resdag.training import ESNTrainer


def _find_cell(model: ESNModel, cell_type: type) -> object:
    """Return the first reservoir cell of ``cell_type`` inside ``model``."""
    for module in model.modules():
        if isinstance(module, cell_type):
            return module
    raise AssertionError(f"no {cell_type.__name__} found in model")


class TestEuSNFactory:
    """The eusn() premade factory."""

    def test_basic_instantiation_and_architecture(self) -> None:
        """Model wires an EuSN reservoir, a concatenation, and a ridge readout."""
        model = eusn(reservoir_size=50, feedback_size=2, output_size=3, epsilon=0.01)
        layer_names = [name for name, _ in model.named_modules()]
        assert any("EuSNLayer" in name for name in layer_names)
        assert any("Concatenate" in name for name in layer_names)
        assert any("RidgeReadoutLayer" in name for name in layer_names)

    def test_forward_shape(self) -> None:
        """(batch, time, feedback) -> (batch, time, output)."""
        model = eusn(reservoir_size=50, feedback_size=2, output_size=3, epsilon=0.02)
        out = model(torch.randn(4, 20, 2))
        assert out.shape == (4, 20, 3)

    def test_epsilon_and_gamma_reach_cell(self) -> None:
        """epsilon / gamma are plumbed to the EuSN cell."""
        model = eusn(reservoir_size=32, feedback_size=2, output_size=2, epsilon=0.03, gamma=0.04)
        cell = _find_cell(model, EuSNCell)
        assert cell.epsilon == 0.03  # type: ignore[attr-defined]
        assert cell.gamma == 0.04  # type: ignore[attr-defined]

    def test_solver_reaches_cell(self) -> None:
        """The solver argument reaches the cell."""
        model = eusn(reservoir_size=32, feedback_size=2, output_size=2, solver="rk4")
        cell = _find_cell(model, EuSNCell)
        assert cell.solver == "rk4"  # type: ignore[attr-defined]

    def test_skew_spectral_radius_reaches_cell(self) -> None:
        """spectral_radius scales the skew operator W_r - W_r^T."""
        from resdag.init.topology import estimate_spectral_radius

        model = eusn(
            reservoir_size=128, feedback_size=2, output_size=2, spectral_radius=1.0, seed=1
        )
        cell = _find_cell(model, EuSNCell)
        w = cell.weight_hh.detach()  # type: ignore[attr-defined]
        skew = w - w.t()
        assert estimate_spectral_radius(skew) == pytest.approx(1.0, abs=5e-3)

    def test_driving_input_model(self) -> None:
        """input_size builds a two-input (feedback, driver) model."""
        model = eusn(reservoir_size=40, feedback_size=3, output_size=3, input_size=2)
        out = model(torch.randn(2, 12, 3), torch.randn(2, 12, 2))
        assert out.shape == (2, 12, 3)

    def test_train_and_forecast_finite(self) -> None:
        """Full pipeline: fit the readout, then autoregress a finite forecast."""
        torch.manual_seed(0)
        series = torch.randn(1, 900, 2) * 0.1
        series = torch.cumsum(series, dim=1)
        model = eusn(
            reservoir_size=80,
            feedback_size=2,
            output_size=2,
            epsilon=0.3,
            gamma=0.5,
            spectral_radius=1.0,
            seed=1,
        )
        trainer = ESNTrainer(model)
        trainer.fit(
            warmup_inputs=(series[:, :100],),
            train_inputs=(series[:, 100:699],),
            targets={"output": series[:, 101:700]},
        )
        pred = model.forecast(series[:, 600:700], horizon=50)
        assert pred.shape == (1, 50, 2)
        assert torch.isfinite(pred).all()

    def test_save_full_load_full_roundtrip(self, tmp_path: object) -> None:
        """save_full / load_full reproduces the same forecast."""
        model = eusn(
            reservoir_size=40,
            feedback_size=2,
            output_size=2,
            epsilon=0.02,
            gamma=0.03,
            spectral_radius=1.0,
            seed=3,
        )
        warm = torch.randn(1, 30, 2)
        path = str(tmp_path) + "/eusn_full.pt"  # type: ignore[operator]
        model.save_full(path)
        restored = ESNModel.load_full(path, trusted=True)
        p1 = model.forecast(warm, horizon=20)
        p2 = restored.forecast(warm, horizon=20)
        assert torch.allclose(p1, p2, atol=1e-5)
        # gamma / epsilon survive the round-trip.
        cell = _find_cell(restored, EuSNCell)
        assert cell.gamma == 0.03  # type: ignore[attr-defined]
        assert cell.epsilon == 0.02  # type: ignore[attr-defined]
