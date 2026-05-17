"""SGD-via-PyTorch training path for trainable ESN models (Phase 4.1).

These tests verify that:

1. ``ESNModel`` is a first-class ``nn.Module`` — gradients propagate
   through ``ESNLayer(trainable=True)`` and through trainable readouts.
2. A standard ``torch.optim.Adam`` loop drives the loss strictly down.
3. The frozen path (``trainable=False`` everywhere) leaves
   ``model.parameters(requires_grad=True)`` empty so SGD becomes a
   no-op — fail loudly if anyone accidentally re-enables grads on a
   classical ESN.

There is no SGDTrainer class; the API contract here is simply
"``ESNModel`` works in any vanilla PyTorch training loop."
"""

import pytest
import torch

import resdag as rd
from resdag.core import ESNModel, reservoir_input
from resdag.layers import CGReadoutLayer, ESNLayer


def _build_trainable_model(reservoir_size: int = 40, feedback_size: int = 3) -> ESNModel:
    """Tiny end-to-end trainable ESN model.

    Both the reservoir *and* its readout have ``trainable=True`` so every
    parameter is exposed to autograd.
    """
    inp = reservoir_input(feedback_size)
    res = ESNLayer(
        reservoir_size=reservoir_size,
        feedback_size=feedback_size,
        spectral_radius=0.9,
        trainable=True,
    )(inp)
    out = CGReadoutLayer(
        in_features=reservoir_size,
        out_features=feedback_size,
        trainable=True,
        name="output",
    )(res)
    return ESNModel(inp, out)


def _build_frozen_model(reservoir_size: int = 40, feedback_size: int = 3) -> ESNModel:
    """Standard frozen-reservoir ESN (classical ESN training territory)."""
    inp = reservoir_input(feedback_size)
    res = ESNLayer(
        reservoir_size=reservoir_size,
        feedback_size=feedback_size,
        spectral_radius=0.9,
        trainable=False,
    )(inp)
    out = CGReadoutLayer(
        in_features=reservoir_size,
        out_features=feedback_size,
        trainable=False,
        name="output",
    )(res)
    return ESNModel(inp, out)


class TestSGDPath:
    """Adam loop must drive loss strictly downward for trainable=True."""

    def test_trainable_model_has_grads(self) -> None:
        model = _build_trainable_model()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        assert len(trainable_params) > 0, (
            "trainable=True model exposed no autograd-tracked parameters"
        )

    def test_sgd_loop_decreases_loss(self) -> None:
        """Train a small ESN for ~50 steps and require loss to drop ~30%+."""
        torch.manual_seed(0)
        model = _build_trainable_model()

        # Synthetic regression task: shifted-noise teacher signal.
        x = torch.randn(8, 30, 3)
        y = torch.roll(x, shifts=-1, dims=1)  # predict next step

        optim = torch.optim.Adam(
            (p for p in model.parameters() if p.requires_grad), lr=1e-2
        )
        criterion = torch.nn.MSELoss()

        losses: list[float] = []
        for _ in range(50):
            model.reset_reservoirs()
            optim.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optim.step()
            losses.append(loss.item())

        # Both ends must be finite and the trajectory must have dropped.
        assert all(torch.isfinite(torch.tensor(L)).item() for L in losses)
        # Allow a generous threshold — we just want to prove gradients flow
        # and the optimiser actually moves the parameters.
        assert losses[-1] < 0.7 * losses[0], (
            f"loss did not decrease: start={losses[0]:.4f}, end={losses[-1]:.4f}"
        )


class TestFrozenPath:
    """Frozen models must expose zero autograd-tracked parameters."""

    def test_frozen_model_has_no_grads(self) -> None:
        model = _build_frozen_model()
        trainable = [name for name, p in model.named_parameters() if p.requires_grad]
        assert trainable == [], (
            f"Frozen ESN unexpectedly has trainable parameters: {trainable}"
        )

    def test_classical_training_still_works_on_frozen(self) -> None:
        """ESNTrainer.fit must still update the frozen readout weights via the
        algebraic path even though requires_grad is False."""
        model = _build_frozen_model()
        # Grab the CG readout directly so we can check is_fitted.
        readouts = [m for _, m in model.named_modules() if isinstance(m, CGReadoutLayer)]
        assert len(readouts) == 1
        readout = readouts[0]
        assert not readout.is_fitted

        warmup = torch.randn(2, 30, 3)
        train = torch.randn(2, 80, 3)
        target = torch.randn(2, 80, 3)

        rd.ESNTrainer(model).fit(
            warmup_inputs=(warmup,),
            train_inputs=(train,),
            targets={"output": target},
        )
        assert readout.is_fitted


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestSGDPathGPU:
    def test_sgd_loop_on_gpu(self) -> None:
        torch.manual_seed(0)
        device = torch.device("cuda")
        model = _build_trainable_model(reservoir_size=30).to(device)

        x = torch.randn(4, 25, 3, device=device)
        y = torch.roll(x, shifts=-1, dims=1)

        optim = torch.optim.Adam(
            (p for p in model.parameters() if p.requires_grad), lr=1e-2
        )
        criterion = torch.nn.MSELoss()

        loss0: float | None = None
        loss1: float | None = None
        for step in range(40):
            model.reset_reservoirs()
            optim.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optim.step()
            if step == 0:
                loss0 = loss.item()
            loss1 = loss.item()

        assert loss0 is not None and loss1 is not None
        assert loss1 < loss0
