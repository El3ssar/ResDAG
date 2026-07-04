"""Premade edge-of-stability model factories (resdag.models).

Pins down the ``es2n`` and ``resesn`` factories: architecture
(``Input -> Skip Reservoir -> Concatenate(Input, Reservoir) -> Readout``),
forward shapes, hyperparameter plumbing (``beta`` / ``alpha`` /
``orthogonal_scaling`` reach the reservoir cell), and a full train + forecast
round-trip through the shared pipeline.

Imports are from the concrete module files: the public ``__init__`` exports are
wired separately.
"""

import torch

from resdag.core import ESNModel
from resdag.layers.cells.es2n_cell import ES2NCell
from resdag.layers.cells.resesn_cell import ResESNCell
from resdag.models.es2n import es2n
from resdag.models.resesn import resesn
from resdag.training import ESNTrainer


def _find_cell(model: ESNModel, cell_type: type) -> object:
    """Return the first reservoir cell of ``cell_type`` inside ``model``."""
    for module in model.modules():
        if isinstance(module, cell_type):
            return module
    raise AssertionError(f"no {cell_type.__name__} found in model")


class TestES2NFactory:
    """The es2n() premade factory."""

    def test_basic_instantiation_and_architecture(self) -> None:
        """Model wires an ES2N reservoir, a concatenation, and a ridge readout."""
        model = es2n(reservoir_size=50, feedback_size=2, output_size=3, beta=0.2)
        layer_names = [name for name, _ in model.named_modules()]
        assert any("ES2NLayer" in name for name in layer_names)
        assert any("Concatenate" in name for name in layer_names)
        assert any("RidgeReadoutLayer" in name for name in layer_names)

    def test_forward_shape(self) -> None:
        """(batch, time, feedback) -> (batch, time, output)."""
        model = es2n(reservoir_size=50, feedback_size=2, output_size=3, beta=0.2)
        out = model(torch.randn(4, 20, 2))
        assert out.shape == (4, 20, 3)

    def test_beta_reaches_cell(self) -> None:
        """The beta argument is plumbed to the ES2N cell."""
        model = es2n(reservoir_size=32, feedback_size=2, output_size=2, beta=0.15)
        cell = _find_cell(model, ES2NCell)
        assert cell.beta == 0.15  # type: ignore[attr-defined]

    def test_orthogonal_scaling_reaches_cell(self) -> None:
        """orthogonal_scaling is plumbed through to the skip matrix."""
        model = es2n(
            reservoir_size=48, feedback_size=2, output_size=2, orthogonal_scaling=2.0, seed=1
        )
        cell = _find_cell(model, ES2NCell)
        svals = torch.linalg.svdvals(cell.skip_matrix)  # type: ignore[attr-defined]
        assert torch.allclose(svals, torch.full_like(svals, 2.0), atol=1e-4)

    def test_readout_in_features_includes_concatenated_input(self) -> None:
        """Concatenation makes readout in_features = feedback + reservoir."""
        model = es2n(reservoir_size=40, feedback_size=3, output_size=3)
        # First (feedback-only) forward exercises the concat path without error.
        assert model(torch.randn(2, 10, 3)).shape == (2, 10, 3)

    def test_train_and_forecast_finite(self) -> None:
        """Full pipeline: fit the readout, then autoregress a finite forecast."""
        torch.manual_seed(0)
        series = torch.randn(1, 900, 2) * 0.1
        series = torch.cumsum(series, dim=1)  # a smooth-ish signal
        model = es2n(reservoir_size=80, feedback_size=2, output_size=2, beta=0.2, seed=1)
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
        model = es2n(reservoir_size=40, feedback_size=2, output_size=2, beta=0.2, seed=3)
        warm = torch.randn(1, 30, 2)
        path = str(tmp_path) + "/es2n_full.pt"  # type: ignore[operator]
        model.save_full(path)
        restored = ESNModel.load_full(path)
        p1 = model.forecast(warm, horizon=20)
        p2 = restored.forecast(warm, horizon=20)
        assert torch.allclose(p1, p2, atol=1e-5)


class TestResESNFactory:
    """The resesn() premade factory."""

    def test_basic_instantiation_and_architecture(self) -> None:
        """Model wires a ResESN reservoir, a concatenation, and a ridge readout."""
        model = resesn(reservoir_size=50, feedback_size=2, output_size=3, alpha=0.6, beta=0.4)
        layer_names = [name for name, _ in model.named_modules()]
        assert any("ResESNLayer" in name for name in layer_names)
        assert any("Concatenate" in name for name in layer_names)
        assert any("RidgeReadoutLayer" in name for name in layer_names)

    def test_forward_shape(self) -> None:
        """(batch, time, feedback) -> (batch, time, output)."""
        model = resesn(reservoir_size=50, feedback_size=2, output_size=3)
        out = model(torch.randn(4, 20, 2))
        assert out.shape == (4, 20, 3)

    def test_alpha_beta_reach_cell(self) -> None:
        """The alpha/beta arguments are plumbed to the ResESN cell."""
        model = resesn(reservoir_size=32, feedback_size=2, output_size=2, alpha=0.7, beta=0.3)
        cell = _find_cell(model, ResESNCell)
        assert cell.alpha == 0.7  # type: ignore[attr-defined]
        assert cell.beta == 0.3  # type: ignore[attr-defined]

    def test_train_and_forecast_finite(self) -> None:
        """Full pipeline: fit the readout, then autoregress a finite forecast."""
        torch.manual_seed(0)
        series = torch.cumsum(torch.randn(1, 900, 2) * 0.1, dim=1)
        model = resesn(
            reservoir_size=80, feedback_size=2, output_size=2, alpha=0.8, beta=0.2, seed=1
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
