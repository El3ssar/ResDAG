"""Customize, calibrate, train and reuse reservoir families on a small signal.

Run: uv run python examples/16_custom_reservoirs.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from torch import nn

import resdag as rd
from resdag.init import register_recurrent_initializer


@register_recurrent_initializer("example_cycle", prescaled=True, gain=0.8)
def example_cycle(n, gain=0.8, *, dtype=None, device=None):
    """A custom initializer usable by name in any recurrent-weight spec."""
    return gain * torch.eye(n, dtype=dtype, device=device).roll(1, dims=0)


def make_model(reservoir):
    reference = next(reservoir.parameters())
    inp = rd.reservoir_input(2, dtype=reference.dtype, device=reference.device)
    head = rd.SVDReadoutLayer(reservoir.cell.output_size, 2, alpha=1e-5, name="output").to(
        reference
    )
    return rd.ESNModel(inp, head(reservoir(inp)))


def main():
    torch.manual_seed(4)
    t = torch.arange(700, dtype=torch.float64) * 0.05
    series = torch.stack((t.sin(), t.cos()), dim=-1).unsqueeze(0)
    warmup, train, target, history, truth = rd.data.prepare_esn_data(
        series,
        warmup_steps=50,
        train_steps=500,
        val_steps=100,
    )

    # Calibration sees training inputs only, before fitting a readout.
    esn = rd.ESNLayer(
        48,
        feedback_size=2,
        recurrent_initializer="example_cycle",
        feedback_initializer=("normal", {"scale": 0.2}),
        bias_initializer=nn.init.zeros_,
        dtype=torch.float64,
        seed=4,
    )
    report = rd.calibrate_intrinsic_plasticity(
        esn,
        torch.cat((warmup, train), dim=1),
        warmup=50,
        target_std=0.2,
        learning_rate=1e-3,
    )
    print(
        f"Calibrated activity std: {report.activity_std_before:.3f} -> "
        f"{report.activity_std_after:.3f}"
    )

    # Other dynamics use the same training, forecast and integration interfaces.
    reservoirs = {
        "custom calibrated ESN": esn,
        "oscillator": rd.CoRNNLayer(48, feedback_size=2, dt=0.05, dtype=torch.float64, seed=4),
        "Legendre memory": rd.LegendreDelayLayer(
            order=12,
            feedback_size=2,
            theta=80,
            dtype=torch.float64,
        ),
        "PyTorch LSTM": rd.TorchRecurrentLayer(
            nn.LSTMCell(2, 32, dtype=torch.float64),
            feedback_size=2,
        ),
    }
    for name, reservoir in reservoirs.items():
        model = make_model(reservoir)
        rd.ESNTrainer(model).fit(warmup, train, target)
        model.eval()
        prediction = model.forecast(history, horizon=truth.shape[1])
        assert prediction.shape == truth.shape and torch.isfinite(prediction).all()
        error = (prediction - truth).square().mean().sqrt().item()
        print(f"{name}: held-out forecast RMSE {error:.6f}")

        # Save, restore, then use the entire fitted model in a bigger network.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            model.save_full(path)
            restored = rd.ESNModel.load_full(path)
            torch.testing.assert_close(
                restored.forecast(history, horizon=truth.shape[1]),
                prediction,
            )
            pipeline = nn.Sequential(restored.as_module(), nn.Linear(2, 1).double())
            inputs = train.detach().clone().requires_grad_()
            pipeline(inputs).square().mean().backward()
            assert inputs.grad is not None and torch.isfinite(inputs.grad).all()

    print("All families trained, forecast, restored, and passed upstream gradients.")


if __name__ == "__main__":
    main()
