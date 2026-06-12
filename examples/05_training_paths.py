"""05 — Training paths: algebraic, frozen + SGD head, and full BPTT.

ResDAG models are plain ``torch.nn.Module``s, so three training styles
coexist:

A. Algebraic (the reservoir-computing classic): freeze everything, fit the
   readout in one ridge-regression solve via ``ESNTrainer``.
B. Frozen reservoir + gradient head: use the reservoir as a fixed feature
   extractor and train any PyTorch head with an optimizer.
C. Full BPTT: ``ESNLayer(trainable=True)`` and end-to-end backprop through
   the recurrence.

All three are run on the same next-step prediction task and compared.

Expected runtime: ~10 s on CPU.
"""

import time

import torch
import torch.nn as nn

from resdag import ESNLayer, ESNModel, reservoir_input
from resdag.layers import CGReadoutLayer
from resdag.training import ESNTrainer


def generate_lorenz(n_steps: int, dt: float = 0.01, seed: int = 42) -> torch.Tensor:
    """Integrate Lorenz-63 (Euler). Returns (1, n_steps, 3), normalized."""
    torch.manual_seed(seed)
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    xyz = torch.empty(n_steps, 3)
    xyz[0] = torch.tensor([1.0, 1.0, 1.05])
    for t in range(1, n_steps):
        x, y, z = xyz[t - 1]
        dxyz = torch.stack((sigma * (y - x), x * (rho - z) - y, x * y - beta * z))
        xyz[t] = xyz[t - 1] + dt * dxyz
    xyz = (xyz - xyz.mean(0)) / xyz.std(0)
    return xyz.unsqueeze(0)  # (1, n_steps, 3)


RESERVOIR_SIZE = 200


def main() -> None:
    torch.manual_seed(42)

    # Shared task: one-step-ahead prediction of Lorenz-63.
    data = generate_lorenz(1801)  # (1, 1801, 3) = (batch, time, features)
    warmup = data[:, :200]  # (1, 200, 3)
    train_in = data[:, 200:1400]  # (1, 1200, 3)
    train_tgt = data[:, 201:1401]  # next step of train_in
    val_in = data[:, 1400:1800]  # (1, 400, 3)
    val_tgt = data[:, 1401:1801]

    results: list[tuple[str, int, float, float]] = []  # path, trainable params, time, MSE

    def teacher_forced_mse(model: ESNModel, head: nn.Module | None = None) -> float:
        """One-step prediction error on the validation window."""
        model.reset_reservoirs()
        with torch.no_grad():
            out = model(torch.cat([warmup, val_in], dim=1))
            if head is not None:
                out = head(out)
            out = out[:, warmup.shape[1] :]  # drop the warmup window
        return torch.mean((out - val_tgt) ** 2).item()

    # ------------------------------------------------------------------
    # Path A: algebraic fitting (ESNTrainer + CGReadoutLayer)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Path A: algebraic — ESNTrainer fits the readout in one solve")
    print("=" * 70)

    inp = reservoir_input(3)  # symbolic input, any sequence length
    states = ESNLayer(RESERVOIR_SIZE, feedback_size=3, spectral_radius=0.8)(inp)
    out = CGReadoutLayer(RESERVOIR_SIZE, 3, name="output", alpha=1e-6)(states)
    model_a = ESNModel(inp, out)

    t0 = time.perf_counter()
    ESNTrainer(model_a).fit(
        warmup_inputs=(warmup,),
        train_inputs=(train_in,),
        targets={"output": train_tgt},
    )
    time_a = time.perf_counter() - t0

    mse_a = teacher_forced_mse(model_a)
    print(f"fitted in {time_a:.3f} s, val one-step MSE = {mse_a:.2e}")
    print("No optimizer, no epochs, no gradients — and it can forecast (06).")
    results.append(("A algebraic", 0, time_a, mse_a))

    # ------------------------------------------------------------------
    # Path B: frozen reservoir + SGD-trained head
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Path B: frozen reservoir features + gradient-trained MLP head")
    print("=" * 70)
    print("Useful when the head is nonlinear or the loss is not least-squares.")

    torch.manual_seed(42)
    inp = reservoir_input(3)
    states = ESNLayer(RESERVOIR_SIZE, feedback_size=3, spectral_radius=0.8, trainable=False)(inp)
    extractor = ESNModel(inp, states)  # headless: outputs (batch, time, RESERVOIR_SIZE)

    head_b = nn.Sequential(
        nn.Linear(RESERVOIR_SIZE, 64),
        nn.Tanh(),
        nn.Linear(64, 3),
    )
    trainable_b = sum(p.numel() for p in head_b.parameters())
    frozen = all(not p.requires_grad for p in extractor.parameters())
    print(f"reservoir frozen: {frozen}; head has {trainable_b} trainable params")

    # Compute the (fixed) features once — the big speed win of a frozen base.
    with torch.no_grad():
        extractor.reset_reservoirs()
        feats = extractor(torch.cat([warmup, train_in], dim=1))[:, warmup.shape[1] :]

    optimizer = torch.optim.Adam(head_b.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    t0 = time.perf_counter()
    for step in range(300):
        pred = head_b(feats)  # (1, 1200, 3)
        loss = criterion(pred, train_tgt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == 299:
            print(f"  step {step:3d} | train loss {loss.item():.5f}")
    time_b = time.perf_counter() - t0

    mse_b = teacher_forced_mse(extractor, head=head_b)
    print(f"trained in {time_b:.2f} s, val one-step MSE = {mse_b:.2e}")
    results.append(("B frozen + SGD head", trainable_b, time_b, mse_b))

    # ------------------------------------------------------------------
    # Path C: full BPTT through the reservoir
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Path C: full BPTT — ESNLayer(trainable=True), end-to-end backprop")
    print("=" * 70)
    print("The recurrent/input weights now receive gradients through time.")

    torch.manual_seed(42)
    reservoir_c = ESNLayer(RESERVOIR_SIZE, feedback_size=3, spectral_radius=0.8, trainable=True)
    head_c = nn.Linear(RESERVOIR_SIZE, 3)
    params_c = list(reservoir_c.parameters()) + list(head_c.parameters())
    trainable_c = sum(p.numel() for p in params_c if p.requires_grad)
    print(f"trainable params: {trainable_c} (reservoir included)")

    optimizer = torch.optim.Adam(params_c, lr=1e-3)

    t0 = time.perf_counter()
    for step in range(30):  # BPTT is expensive — few steps, short window
        reservoir_c.reset_state()
        pred = head_c(reservoir_c(train_in[:, :400]))  # truncate for speed
        loss = criterion(pred, train_tgt[:, :400])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 10 == 0 or step == 29:
            print(f"  step {step:3d} | train loss {loss.item():.5f}")
    time_c = time.perf_counter() - t0

    with torch.no_grad():
        reservoir_c.reset_state()
        out = head_c(reservoir_c(torch.cat([warmup, val_in], dim=1)))[:, warmup.shape[1] :]
    mse_c = torch.mean((out - val_tgt) ** 2).item()
    print(f"trained in {time_c:.2f} s, val one-step MSE = {mse_c:.2e}")
    results.append(("C full BPTT", trainable_c, time_c, mse_c))

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Comparison (same reservoir size, same task)")
    print("=" * 70)
    header = f"{'path':<22} {'grad params':>12} {'time [s]':>10} {'val MSE':>12}"
    print(header)
    print("-" * len(header))
    for name, n_params, secs, mse in results:
        print(f"{name:<22} {n_params:>12,} {secs:>10.2f} {mse:>12.2e}")
    print("-" * len(header))
    print("Rule of thumb: start with A. Reach for B when you need a nonlinear")
    print("head or custom loss; reach for C only when the random reservoir")
    print("provably is not good enough — it is orders of magnitude slower.")


if __name__ == "__main__":
    main()
