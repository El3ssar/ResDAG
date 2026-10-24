"""12 — Reservoir as a frozen feature extractor for a torch head.

``ReservoirFeatureExtractor`` packages a reservoir as a plain
``torch.nn.Module`` that drops straight into ``nn.Sequential`` ahead of any
trainable head. The reservoir is frozen by default, so a single optimizer
over ``model.parameters()`` trains only the head — the classic
"random features + learned readout" recipe, now a one-liner.

Two heads share one extractor pattern:

A. Regression head — next-step prediction of Lorenz-63 (the Path B result
   from ``05_training_paths.py``, now via the adapter).
B. Classification head — label each window by which of two regimes it came
   from, trained with cross-entropy on the same frozen features.

Expected runtime: ~5 s on CPU.
"""

import torch
import torch.nn as nn

from resdag import ESNLayer, ESNModel, ReservoirFeatureExtractor, lorenz, reservoir_input
from resdag.data import normalize_data

RESERVOIR_SIZE = 200


def regression_head(*, quick: bool = False) -> None:
    """A. Frozen extractor + Adam-trained MLP head for next-step prediction."""
    print("=" * 70)
    print("A: frozen reservoir features -> gradient-trained regression head")
    print("=" * 70)

    torch.manual_seed(42)
    size = 24 if quick else RESERVOIR_SIZE
    washout, train_end, val_end = (12, 92, 116) if quick else (200, 1400, 1800)
    steps = 3 if quick else 300
    raw = lorenz(val_end + 1, seed=42, normalize=None)  # raw (1, 1801, 3)
    # The held-out suffix never enters the fitted input/target scale.
    _, stats = normalize_data(raw[:, washout:train_end], method="standard")
    data, _ = normalize_data(raw, method="standard", stats=stats)
    warmup = data[:, :washout]
    train_in = data[:, washout:train_end]
    train_tgt = data[:, washout + 1 : train_end + 1]
    val_in = data[:, train_end:val_end]
    val_tgt = data[:, train_end + 1 : val_end + 1]

    # Drop the extractor straight into nn.Sequential. Frozen by default.
    model = nn.Sequential(
        ReservoirFeatureExtractor(size, feedback_size=3, spectral_radius=0.8),
        nn.Linear(size, 64),
        nn.Tanh(),
        nn.Linear(64, 3),
    )
    extractor = model[0]
    head = model[1:]
    print(f"reservoir frozen: {extractor.is_frozen}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params (head only): {trainable:,}")

    # Frozen features: compute once — the main speed win of a frozen base.
    with torch.no_grad():
        extractor.on_epoch_start()  # epoch-reset hook (alias of reset_state)
        feats = extractor(torch.cat([warmup, train_in], dim=1))[:, warmup.shape[1] :]

    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    for step in range(steps):
        loss = criterion(head(feats), train_tgt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == steps - 1:
            print(f"  step {step:3d} | train loss {loss.item():.5f}")

    # Teacher-forced one-step validation error.
    with torch.no_grad():
        extractor.on_epoch_start()
        val_feats = model[:1](torch.cat([warmup, val_in], dim=1))
        val_pred = head(val_feats)[:, warmup.shape[1] :]
    mse = torch.mean((val_pred - val_tgt) ** 2).item()
    assert torch.isfinite(val_pred).all()
    print(f"val one-step MSE = {mse:.2e} (training-standardized units)")


def classification_head(*, quick: bool = False) -> None:
    """B. The same frozen extractor feeding a cross-entropy classification head."""
    print("\n" + "=" * 70)
    print("B: frozen reservoir features -> torch classification head")
    print("=" * 70)

    torch.manual_seed(0)
    # Two regimes: a slow sine and a fast sine, 3 channels each. The task is to
    # label every timestep's window by which regime produced it.
    n, length = (4, 16) if quick else (64, 80)
    size = 24 if quick else RESERVOIR_SIZE
    steps = 3 if quick else 150
    t = torch.linspace(0, 6 * torch.pi, length)
    slow = torch.stack([torch.sin(t * (1 + 0.1 * i)) for i in range(3)], dim=-1)
    fast = torch.stack([torch.sin(t * (3 + 0.1 * i)) for i in range(3)], dim=-1)
    x_slow = slow.unsqueeze(0).repeat(n, 1, 1) + 0.05 * torch.randn(n, length, 3)
    x_fast = fast.unsqueeze(0).repeat(n, 1, 1) + 0.05 * torch.randn(n, length, 3)
    x = torch.cat([x_slow, x_fast], dim=0)  # (2n, length, 3)
    labels = torch.cat([torch.zeros(n), torch.ones(n)]).long()  # (2n,)

    extractor = ReservoirFeatureExtractor(size, feedback_size=3, spectral_radius=0.9)
    head = nn.Linear(size, 2)  # 2-class logits

    # Use the last-timestep feature vector as the sequence summary.
    with torch.no_grad():
        extractor.reset_state()
        summary = extractor(x)[:, -1, :]  # (2n, RESERVOIR_SIZE)

    optimizer = torch.optim.Adam(head.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    for step in range(steps):
        logits = head(summary)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step == steps - 1:
            acc = (logits.argmax(-1) == labels).float().mean().item()
            print(f"  step {step:3d} | loss {loss.item():.4f} | train acc {acc:.3f}")
    assert extractor.is_frozen and torch.isfinite(logits).all()
    print(f"reservoir frozen throughout: {extractor.is_frozen}")


def reuse_existing_model(*, quick: bool = False) -> None:
    """C. Reuse the reservoir of an existing ESNModel — shared, not copied."""
    print("\n" + "=" * 70)
    print("C: from_model reuses an existing ESNModel's reservoir (shared)")
    print("=" * 70)

    inp = reservoir_input(3)
    states = ESNLayer(24 if quick else RESERVOIR_SIZE, feedback_size=3)(inp)
    esn = ESNModel(inp, states)

    extractor = ReservoirFeatureExtractor.from_model(esn)
    src = next(m for m in esn.modules() if hasattr(m, "weight_hh"))
    shared = next(m for m in extractor.modules() if isinstance(m, ESNLayer))
    assert shared.weight_hh is src.weight_hh
    print(f"shares recurrent weights: {shared.weight_hh is src.weight_hh}")


def main() -> None:
    from _example_options import parse_options

    options = parse_options(__doc__)
    regression_head(quick=options.quick)
    classification_head(quick=options.quick)
    reuse_existing_model(quick=options.quick)


if __name__ == "__main__":
    main()
