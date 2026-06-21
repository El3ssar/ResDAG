"""14 — Streaming DataLoader: windowed SGD and algebraic-over-DataLoader fits.

resdag models are plain ``torch.nn.Module``s, so they slot into the ordinary
PyTorch data stack. ``TimeSeriesWindowDataset`` slices a long trajectory into
fixed-length windows and ``make_dataloader`` wraps it in a standard
``DataLoader`` whose batches are ``(B, window_len, D)`` — exactly what
``ESNLayer`` / ``ESNModel`` consume.

This example shows three things on a shared Lorenz-63 task:

A. The canonical SGD loop — frozen reservoir + trainable head, minibatched
   over the loader. ``reset_reservoirs()`` per window keeps state from leaking
   across windows; ``detach_state_between_calls`` (on by default) handles the
   cross-batch autograd graph hygiene.
B. The same loop with a *trainable* reservoir (full BPTT through the windows).
C. The algebraic-over-DataLoader path: stream each batch into
   ``IncrementalRidgeReadout.partial_fit``, then ``finalize`` once — a one-solve
   ridge fit that never holds the whole state matrix in memory.

Expected runtime: ~10 s on CPU.
"""

import torch
import torch.nn as nn

from resdag import ESNLayer, lorenz, make_dataloader
from resdag.layers import IncrementalRidgeReadout

RESERVOIR_SIZE = 200
WINDOW_LEN = 250
WASHOUT = 50


def main() -> None:
    torch.manual_seed(42)

    # One long Lorenz-63 trajectory, shaped (T, D). The dataset slices it into
    # overlapping windows; the loader stacks them to (B, window_len, D).
    series = lorenz(4000, seed=42)[0]  # (4000, 3)
    print(f"source series: {tuple(series.shape)} (time, features)")

    loader = make_dataloader(
        series,
        batch_size=16,
        window_len=WINDOW_LEN,
        horizon=1,  # one-step-ahead forecasting target
        stride=WINDOW_LEN // 2,  # 50% window overlap
        washout=WASHOUT,
        shuffle=True,
    )
    n_windows = len(loader.dataset)  # type: ignore[arg-type]
    x0, y0, washout = next(iter(loader))
    print(f"loader: {n_windows} windows, batch {tuple(x0.shape)} -> target {tuple(y0.shape)}")
    print(f"washout per window: {washout} steps (excluded from the loss)\n")

    criterion = nn.MSELoss()

    # ------------------------------------------------------------------
    # Path A: frozen reservoir + SGD-trained linear head
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Path A: frozen reservoir features + SGD-trained head over the loader")
    print("=" * 70)
    torch.manual_seed(42)
    reservoir_a = ESNLayer(RESERVOIR_SIZE, feedback_size=3, spectral_radius=0.9, trainable=False)
    head_a = nn.Linear(RESERVOIR_SIZE, 3)
    optim_a = torch.optim.Adam(head_a.parameters(), lr=5e-3)

    for epoch in range(6):
        epoch_loss = 0.0
        n_batches = 0
        for x, y, wo in loader:
            reservoir_a.reset_state()  # each window is independent
            pred = head_a(reservoir_a(x))
            loss = criterion(pred[:, wo:], y[:, wo:])  # skip the washout transient
            optim_a.zero_grad()
            loss.backward()
            optim_a.step()
            epoch_loss += loss.item()
            n_batches += 1
        print(f"  epoch {epoch} | mean train MSE {epoch_loss / n_batches:.5f}")

    # ------------------------------------------------------------------
    # Path B: fully-trainable reservoir (BPTT through each window)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Path B: trainable reservoir (full BPTT) over the loader")
    print("=" * 70)
    torch.manual_seed(42)
    reservoir_b = ESNLayer(RESERVOIR_SIZE, feedback_size=3, spectral_radius=0.9, trainable=True)
    head_b = nn.Linear(RESERVOIR_SIZE, 3)
    params_b = list(reservoir_b.parameters()) + list(head_b.parameters())
    optim_b = torch.optim.Adam(params_b, lr=2e-3)

    for epoch in range(4):
        epoch_loss = 0.0
        n_batches = 0
        for x, y, wo in loader:
            reservoir_b.reset_state()
            pred = head_b(reservoir_b(x))
            loss = criterion(pred[:, wo:], y[:, wo:])
            optim_b.zero_grad()
            loss.backward()
            optim_b.step()
            epoch_loss += loss.item()
            n_batches += 1
        print(f"  epoch {epoch} | mean train MSE {epoch_loss / n_batches:.5f}")

    # ------------------------------------------------------------------
    # Path C: algebraic over the DataLoader (incremental ridge)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Path C: algebraic over the loader — IncrementalRidgeReadout")
    print("=" * 70)
    print("Stream partial_fit per batch, finalize once — no whole-state matrix.")
    torch.manual_seed(42)
    reservoir_c = ESNLayer(RESERVOIR_SIZE, feedback_size=3, spectral_radius=0.9, trainable=False)
    readout_c = IncrementalRidgeReadout(RESERVOIR_SIZE, 3, name="output", alpha=1e-4)

    readout_c.reset_accumulators()
    with torch.no_grad():
        for x, y, wo in loader:
            reservoir_c.reset_state()
            states = reservoir_c(x)
            readout_c.partial_fit(states[:, wo:], y[:, wo:])  # accumulate stats
    print(f"  accumulated {readout_c.n_seen} post-washout samples across all batches")
    readout_c.finalize()  # one ridge solve from the running sufficient statistics

    with torch.no_grad():
        x, y, wo = next(iter(loader))
        reservoir_c.reset_state()
        pred = readout_c(reservoir_c(x))
        mse_c = torch.mean((pred[:, wo:] - y[:, wo:]) ** 2).item()
    print(f"  fitted (is_fitted={readout_c.is_fitted}); held-out one-step MSE {mse_c:.5f}")

    print("\nTakeaway: the same windowed loader drives SGD (A/B) and a one-solve")
    print("algebraic fit (C). Reach for C when the reservoir is fixed and you want")
    print("the classic ridge readout without materialising every state at once.")


if __name__ == "__main__":
    main()
