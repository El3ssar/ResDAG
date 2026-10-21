"""Hybrid physics-prior ESN factory contracts (resdag.models.hybrid_esn).

Pins down the Pathak-2018 hybrid wiring: a knowledge-based expert K(u) fed into
the reservoir twice — concatenated with the raw input *before* the reservoir
([u, K(u)]) and concatenated with the reservoir state *before* the readout
([h, K(u)]) — with the reservoir input weights coming from the informed
(state/model) split. Also checks the KnowledgeModel wrapper's shape handling and
that the model trains and forecasts finitely.
"""

import pytest
import torch
import torch.nn as nn

from resdag.layers import Concatenate
from resdag.layers.cells import ESNCell
from resdag.layers.readouts import ReadoutLayer, RidgeReadoutLayer
from resdag.models import KnowledgeModel, hybrid_esn
from resdag.training import ESNTrainer


def lorenz_euler(u: torch.Tensor, dt: float = 0.02) -> torch.Tensor:
    """A crude one-step Lorenz Euler expert (feature-wise on the last dim)."""
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    x, y, z = u[..., 0], u[..., 1], u[..., 2]
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return u + dt * torch.stack([dx, dy, dz], dim=-1)


class TestKnowledgeModel:
    """The KnowledgeModel adapter."""

    def test_callable_wrapped_and_applied(self) -> None:
        km = KnowledgeModel(lorenz_euler, model_out_size=3)
        u = torch.randn(4, 20, 3)
        out = km(u)
        assert out.shape == (4, 20, 3)
        assert torch.equal(out, lorenz_euler(u))

    def test_single_step_shape(self) -> None:
        """Works on the (batch, 1, features) slice used during forecasting."""
        km = KnowledgeModel(lorenz_euler, model_out_size=3)
        u = torch.randn(4, 1, 3)
        assert km(u).shape == (4, 1, 3)

    def test_module_expert_registered_as_submodule(self) -> None:
        """An nn.Module expert's parameters move with the wrapper."""

        class LinearExpert(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(3, 3)

            def forward(self, u: torch.Tensor) -> torch.Tensor:
                return self.lin(u)

        km = KnowledgeModel(LinearExpert(), model_out_size=3)
        # The expert's Linear params are discovered through the wrapper.
        assert any(p.numel() > 0 for p in km.parameters())

    def test_different_model_out_size(self) -> None:
        """The expert may change the feature dimension."""

        def project(u: torch.Tensor) -> torch.Tensor:
            return u[..., :1]  # 3 -> 1

        km = KnowledgeModel(project, model_out_size=1)
        assert km(torch.randn(2, 5, 3)).shape == (2, 5, 1)

    def test_wrong_output_dim_raises(self) -> None:
        km = KnowledgeModel(lambda u: u[..., :2], model_out_size=3)
        with pytest.raises(ValueError, match="model_out_size"):
            km(torch.randn(2, 5, 3))

    def test_non_callable_raises(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            KnowledgeModel(42, model_out_size=3)  # type: ignore[arg-type]

    def test_non_positive_model_out_size_raises(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            KnowledgeModel(lorenz_euler, model_out_size=0)


class TestHybridESNConstruction:
    """Wiring and feature dimensions of the hybrid architecture."""

    def test_knowledge_model_present(self) -> None:
        model = hybrid_esn(lorenz_euler, reservoir_size=50, feedback_size=3, output_size=3)
        assert any(isinstance(m, KnowledgeModel) for m in model.modules())

    def test_prebuilt_knowledge_model_used(self) -> None:
        km = KnowledgeModel(lorenz_euler, model_out_size=3)
        model = hybrid_esn(km, reservoir_size=50, feedback_size=3, output_size=3)
        assert km in list(model.modules())

    def test_reservoir_feedback_size_includes_model_block(self) -> None:
        """The reservoir consumes [u, K(u)]: feedback_size + model_out_size."""
        model = hybrid_esn(lorenz_euler, reservoir_size=50, feedback_size=3, output_size=3)
        cell = next(m for m in model.modules() if isinstance(m, ESNCell))
        assert cell.feedback_size == 6  # 3 (u) + 3 (K(u))

    def test_readout_in_features_includes_model_block(self) -> None:
        """The readout sees [h, K(u)]: reservoir_size + model_out_size."""
        model = hybrid_esn(lorenz_euler, reservoir_size=50, feedback_size=3, output_size=3)
        readout = next(m for m in model.modules() if isinstance(m, ReadoutLayer))
        assert readout.in_features == 53  # 50 (h) + 3 (K(u))

    def test_two_concatenate_nodes(self) -> None:
        """One concat before the reservoir, one before the readout."""
        model = hybrid_esn(lorenz_euler, reservoir_size=50, feedback_size=3, output_size=3)
        concats = [m for m in model.modules() if isinstance(m, Concatenate)]
        assert len(concats) >= 2

    def test_informed_initializer_on_feedback_weights(self) -> None:
        """The reservoir feedback weights use the informed (state/model) split."""
        model = hybrid_esn(
            lorenz_euler, reservoir_size=50, feedback_size=3, output_size=3, gamma=0.5
        )
        cell = next(m for m in model.modules() if isinstance(m, ESNCell))
        # The cell keeps the raw feedback-initializer spec: the informed tuple with
        # the K(u) block width and gamma wired through.
        spec = cell.feedback_initializer
        assert isinstance(spec, tuple)
        name, params = spec
        assert name == "informed_init"
        assert params["model_in_size"] == 3  # K(u) block width
        assert params["gamma"] == pytest.approx(0.5)

    def test_informed_split_produces_one_connection_per_row(self) -> None:
        """The informed init wires at most one input connection per reservoir unit."""
        model = hybrid_esn(
            lorenz_euler, reservoir_size=50, feedback_size=3, output_size=3, gamma=0.5
        )
        cell = next(m for m in model.modules() if isinstance(m, ESNCell))
        # weight_feedback has shape (reservoir_size, feedback_size + model_out) and,
        # per informed_init, no row has more than one nonzero entry.
        nnz_per_row = (cell.weight_feedback != 0).sum(dim=1)
        assert bool((nnz_per_row <= 1).all())

    def test_default_readout_is_ridge(self) -> None:
        model = hybrid_esn(lorenz_euler, reservoir_size=50, feedback_size=3, output_size=3)
        readouts = [m for m in model.modules() if isinstance(m, ReadoutLayer)]
        assert readouts and all(isinstance(r, RidgeReadoutLayer) for r in readouts)

    def test_callable_model_out_size_override(self) -> None:
        """A bare callable uses model_out_size (default feedback_size)."""

        def project_expert(u: torch.Tensor) -> torch.Tensor:
            return u[..., :2]

        model = hybrid_esn(
            project_expert,
            reservoir_size=50,
            feedback_size=3,
            output_size=3,
            model_out_size=2,
        )
        cell = next(m for m in model.modules() if isinstance(m, ESNCell))
        assert cell.feedback_size == 5  # 3 (u) + 2 (K(u))
        readout = next(m for m in model.modules() if isinstance(m, ReadoutLayer))
        assert readout.in_features == 52  # 50 + 2

    def test_non_positive_output_size_raises(self) -> None:
        with pytest.raises(ValueError, match="output_size"):
            hybrid_esn(lorenz_euler, reservoir_size=50, feedback_size=3, output_size=0)


class TestHybridESNForward:
    """Forward, training, and autoregressive forecast."""

    def test_forward_shape(self) -> None:
        model = hybrid_esn(lorenz_euler, reservoir_size=50, feedback_size=3, output_size=3)
        x = torch.randn(2, 20, 3)
        assert model(x).shape == (2, 20, 3)

    def test_fit_and_forecast_finite(self) -> None:
        """A hybrid ESN fits and forecasts finitely.

        The expert is re-evaluated on the model's own output at each step, so the
        physics prior stays live throughout the autoregressive rollout.
        """
        torch.manual_seed(0)
        model = hybrid_esn(
            lorenz_euler,
            reservoir_size=100,
            feedback_size=3,
            output_size=3,
            gamma=0.5,
            spectral_radius=0.9,
            readout_alpha=1e-4,
        )
        # A short smooth trajectory keeps the Euler expert well-conditioned.
        t = torch.linspace(0, 6, 260).view(1, -1, 1)
        series = 0.3 * torch.cat([torch.sin(t), torch.cos(t), torch.sin(2 * t)], dim=-1)
        warmup = series[:, :60, :]
        train_in = series[:, 60:259, :]
        target = series[:, 61:260, :]

        ESNTrainer(model).fit(
            warmup_inputs=(warmup,),
            train_inputs=(train_in,),
            targets={"output": target},
        )
        preds = model.forecast(warmup, horizon=40)

        assert preds.shape == (1, 40, 3)
        assert torch.isfinite(preds).all()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestHybridESNGPU:
    """Hybrid ESN runs on GPU."""

    def test_forward_on_gpu(self) -> None:
        model = hybrid_esn(lorenz_euler, reservoir_size=50, feedback_size=3, output_size=3).cuda()
        x = torch.randn(2, 20, 3).cuda()
        out = model(x)
        assert out.is_cuda
        assert out.shape == (2, 20, 3)


def test_knowledge_model_rejects_mismatched_time_axes() -> None:
    expert = KnowledgeModel(lambda x: x[:, :1].expand(-1, 4, -1), model_out_size=2)
    with pytest.raises(ValueError, match="same batch/time"):
        expert(torch.randn(3, 5, 2))
