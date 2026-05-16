"""ESN as a building block in a larger PyTorch pipeline (Phase 4.2).

Demonstrates that an :class:`resdag.ESNModel` is a first-class
``torch.nn.Module``: it nests cleanly into ``nn.Sequential``, plays well
with vanilla autograd, and can be trained end-to-end with any PyTorch
optimiser.

Two scenarios are shown:

1. **Frozen reservoir, trainable downstream head**
   Classical ESN feature extractor (``trainable=False``) followed by a
   small MLP head trained with Adam. Reservoir weights stay fixed; only
   the head moves.

2. **Fully trainable end-to-end**
   ``ESNLayer(trainable=True)`` with a regression head, trained jointly
   via backpropagation through time. Useful when the reservoir
   initialisation isn't ideal for the task.

Both scenarios use the same synthetic next-step prediction task on a
band-limited noise signal so the loss curves are directly comparable.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from resdag import ESNLayer, ESNModel, reservoir_input


def make_dataset(
    n_batches: int = 32,
    seq_len: int = 60,
    feature_dim: int = 3,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Smooth synthetic signal where target = next-step input."""
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(n_batches, seq_len + 1, feature_dim, generator=g)
    # Light smoothing so the task isn't pure noise.
    kernel = torch.tensor([0.25, 0.5, 0.25])
    smooth = torch.zeros_like(raw)
    for d in range(feature_dim):
        col = raw[:, :, d]
        smooth[:, :, d] = torch.nn.functional.conv1d(
            col.unsqueeze(1),
            kernel.view(1, 1, 3),
            padding=1,
        ).squeeze(1)
    x = smooth[:, :-1, :]
    y = smooth[:, 1:, :]
    return x, y


# ---------------------------------------------------------------------------
# Scenario 1: Frozen reservoir + trainable MLP head
# ---------------------------------------------------------------------------


def scenario_frozen_reservoir() -> None:
    print("\n" + "=" * 60)
    print("Scenario 1: frozen reservoir + trainable MLP head")
    print("=" * 60)

    torch.manual_seed(42)
    x, y = make_dataset()

    reservoir_size = 80
    feature_dim = 3

    # ESNModel acts as a feature extractor: shape preserved (B, T, R)
    inp = reservoir_input(feature_dim)
    res = ESNLayer(
        reservoir_size=reservoir_size,
        feedback_size=feature_dim,
        spectral_radius=0.9,
        trainable=False,  # frozen reservoir weights
    )(inp)
    feature_model = ESNModel(inp, res)

    # Trainable MLP head consumes the reservoir features per timestep
    head = nn.Sequential(
        nn.Linear(reservoir_size, 32),
        nn.Tanh(),
        nn.Linear(32, feature_dim),
    )

    # Confirm the partitioning of trainable parameters
    frozen_n = sum(p.numel() for p in feature_model.parameters())
    head_n = sum(p.numel() for p in head.parameters() if p.requires_grad)
    grad_frozen = [p for p in feature_model.parameters() if p.requires_grad]
    print(f"Reservoir params (all frozen) : {frozen_n}  (with grad: {len(grad_frozen)})")
    print(f"Head params (all trainable)   : {head_n}")

    # Only the head's parameters are passed to the optimiser
    optim = torch.optim.Adam(head.parameters(), lr=1e-2)
    criterion = nn.MSELoss()

    for step in range(80):
        feature_model.reset_reservoirs()
        feats = feature_model(x)  # (B, T, R)  — frozen graph, no grads
        pred = head(feats)
        loss = criterion(pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if step % 20 == 0 or step == 79:
            print(f"  step {step:3d} | loss {loss.item():.5f}")

    # Sanity: reservoir weight matrix is unchanged after training.
    res_cell = next(m for m in feature_model.modules() if isinstance(m, ESNLayer))
    print(
        f"Reservoir w_hh frozen? requires_grad={res_cell.weight_hh.requires_grad}, "
        f"any grad attached={res_cell.weight_hh.grad is not None}"
    )


# ---------------------------------------------------------------------------
# Scenario 2: Fully trainable end-to-end (BPTT through the reservoir)
# ---------------------------------------------------------------------------


def scenario_trainable_e2e() -> None:
    print("\n" + "=" * 60)
    print("Scenario 2: fully trainable end-to-end (BPTT)")
    print("=" * 60)

    torch.manual_seed(42)
    x, y = make_dataset()

    reservoir_size = 60
    feature_dim = 3

    inp = reservoir_input(feature_dim)
    res = ESNLayer(
        reservoir_size=reservoir_size,
        feedback_size=feature_dim,
        spectral_radius=0.9,
        trainable=True,  # reservoir + readout learn jointly
    )(inp)
    head = nn.Linear(reservoir_size, feature_dim)(res)
    model = ESNModel(inp, head)

    grad_n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_n = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {grad_n} / {total_n} total")

    optim = torch.optim.Adam(model.parameters(), lr=5e-3)
    criterion = nn.MSELoss()

    for step in range(80):
        model.reset_reservoirs()
        pred = model(x)
        loss = criterion(pred, y)
        optim.zero_grad()
        loss.backward()
        # BPTT through a recurrent reservoir benefits from gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optim.step()
        if step % 20 == 0 or step == 79:
            print(f"  step {step:3d} | loss {loss.item():.5f}")


# ---------------------------------------------------------------------------
# Scenario 3: ESNModel inside nn.Sequential
# ---------------------------------------------------------------------------


def scenario_nn_sequential() -> None:
    """ESNModel composes inside nn.Sequential too — the typical pattern when
    using PyTorch's built-in containers."""
    print("\n" + "=" * 60)
    print("Scenario 3: ESNModel inside nn.Sequential")
    print("=" * 60)

    torch.manual_seed(42)
    x, y = make_dataset()
    reservoir_size = 64
    feature_dim = 3

    inp = reservoir_input(feature_dim)
    res = ESNLayer(
        reservoir_size=reservoir_size,
        feedback_size=feature_dim,
        spectral_radius=0.9,
        trainable=False,
    )(inp)
    feature_model = ESNModel(inp, res)

    pipeline = nn.Sequential(
        feature_model,
        nn.Linear(reservoir_size, feature_dim),
    )

    grad_params = [p for p in pipeline.parameters() if p.requires_grad]
    optim = torch.optim.Adam(grad_params, lr=1e-2)
    criterion = nn.MSELoss()

    for step in range(60):
        feature_model.reset_reservoirs()
        pred = pipeline(x)
        loss = criterion(pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if step % 20 == 0 or step == 59:
            print(f"  step {step:3d} | loss {loss.item():.5f}")


if __name__ == "__main__":
    scenario_frozen_reservoir()
    scenario_trainable_e2e()
    scenario_nn_sequential()
    print("\n" + "=" * 60)
    print("All pipeline-integration scenarios completed.")
    print("=" * 60)
