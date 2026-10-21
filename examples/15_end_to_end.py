"""Train, evaluate, save and compose a reservoir model with ordinary PyTorch.

Run from the repository: uv run python examples/15_end_to_end.py
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn

import resdag as rd


def main() -> None:
    torch.manual_seed(0)
    # Low-level interfaces use (batch, time, features), with matching dtype/device.
    t = torch.arange(1200, dtype=torch.float64) * 0.05
    data = torch.stack((t.sin(), t.cos()), dim=-1).unsqueeze(0)
    warmup, train, target, history, truth, stats = rd.data.prepare_esn_data(
        data,
        warmup_steps=100,
        train_steps=800,
        val_steps=200,
        normalize=True,
        norm_method="standard",
        return_stats=True,
    )
    model = rd.classic_esn(64, feedback_size=2, output_size=2, readout="svd").double()
    rd.ESNTrainer(model).fit(warmup, train, target)
    model.eval()
    prediction = model.forecast(history, horizon=truth.shape[1])
    assert prediction.shape == truth.shape
    assert stats is not None
    prediction_original = rd.data.denormalize_data(prediction, "standard", stats)
    truth_original = rd.data.denormalize_data(truth, "standard", stats)
    rmse = (prediction_original - truth_original).square().mean().sqrt().item()
    print(f"Held-out forecast RMSE: {rmse:.6f}")

    # Full model: convenient for trusted files. Save normalization alongside it.
    with TemporaryDirectory() as folder:
        path = Path(folder) / "model.pt"
        model.save_full(path, normalization=stats, norm_method="standard")
        restored, metadata = rd.ESNModel.load_full(path, return_metadata=True, map_location="cpu")
        assert metadata["norm_method"] == "standard"
        torch.testing.assert_close(restored.forecast(history, horizon=200), prediction)

    # Reuse the complete trained reservoir + readout as a normal PyTorch module.
    # Independent windows start from zero state, so batches do not affect one another.
    head = nn.Linear(2, 1).double()
    pipeline = nn.Sequential(model.as_module(), head)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.01)
    inputs = train[:, :100]
    labels = target[:, :100, :1]
    for _ in range(10):
        optimizer.zero_grad()
        loss = (pipeline(inputs) - labels).square().mean()
        loss.backward()
        optimizer.step()
    print(f"Neural head training loss: {loss.item():.6f}")

    # For ordered chunks, carry state explicitly; the model keeps no hidden state here.
    whole, _ = model.forward_stateless(inputs)
    left, states = model.forward_stateless(inputs[:, :40])
    right, _ = model.forward_stateless(inputs[:, 40:], states=states)
    torch.testing.assert_close(torch.cat((left, right), dim=1), whole)
    print("Checkpoint, PyTorch composition and chunk continuity verified.")


if __name__ == "__main__":
    main()
