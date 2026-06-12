"""03 — Functional API: building reservoir DAGs by hand.

The premade factories (02) cover common architectures; for everything else
you wire layers together with the ``pytorch_symbolic`` functional API and
wrap the graph in ``ESNModel``.

What it shows
-------------
1. Minimal model: Input -> ESNLayer -> CGReadoutLayer
2. Input-driven model: feedback + exogenous driver
3. Ott-style state augmentation by hand (what ``ott_esn`` builds for you)
4. Parallel reservoirs merged by ``Concatenate``
5. Multi-readout DAG trained with one ``ESNTrainer.fit`` call

Expected runtime: ~5 s on CPU.
"""

import torch

from resdag.core import ESNModel, Input
from resdag.layers import CGReadoutLayer, Concatenate, ESNLayer, SelectiveExponentiation
from resdag.training import ESNTrainer


def main() -> None:
    torch.manual_seed(42)

    # ------------------------------------------------------------------
    # 1. Minimal model
    # ------------------------------------------------------------------
    print("=" * 70)
    print("1. Minimal model: Input -> ESNLayer -> CGReadoutLayer")
    print("=" * 70)

    # Input shape is (timesteps, features); batch is implicit. The time size
    # is a placeholder — any sequence length is accepted at call time.
    inp = Input((100, 3))
    states = ESNLayer(reservoir_size=200, feedback_size=3)(inp)
    out = CGReadoutLayer(in_features=200, out_features=3, name="output")(states)
    model = ESNModel(inp, out)

    x = torch.randn(4, 100, 3)  # (batch, time, features)
    y = model(x)
    print(f"forward: {tuple(x.shape)} -> {tuple(y.shape)}")

    # ------------------------------------------------------------------
    # 2. Input-driven model (feedback + driver)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. Input-driven model: feedback + exogenous driver")
    print("=" * 70)
    print("Convention: the FIRST input is always the feedback signal;")
    print("any further inputs are drivers (known exogenous series).")

    feedback = Input((100, 1))
    driver = Input((100, 5))
    states = ESNLayer(reservoir_size=150, feedback_size=1, input_size=5)(feedback, driver)
    out = CGReadoutLayer(150, 1, name="output")(states)
    driven_model = ESNModel([feedback, driver], out)

    fb = torch.randn(2, 100, 1)  # (batch, time, feedback_dim)
    dr = torch.randn(2, 100, 5)  # (batch, time, driver_dim)
    y = driven_model(fb, dr)
    print(f"forward: feedback {tuple(fb.shape)} + driver {tuple(dr.shape)} -> {tuple(y.shape)}")

    # ------------------------------------------------------------------
    # 3. Ott-style augmentation by hand
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. Ott-style state augmentation (this is what ott_esn builds)")
    print("=" * 70)

    inp = Input((100, 3))
    states = ESNLayer(reservoir_size=200, feedback_size=3)(inp)
    # Square the even-indexed reservoir units (breaks the tanh odd symmetry)
    augmented = SelectiveExponentiation(index=0, exponent=2.0)(states)
    # Readout sees the raw input next to the augmented states
    features = Concatenate()(inp, augmented)  # (batch, time, 3 + 200)
    out = CGReadoutLayer(in_features=3 + 200, out_features=3, name="output")(features)
    ott_style = ESNModel(inp, out)

    y = ott_style(torch.randn(2, 100, 3))
    print(f"forward: (2, 100, 3) -> {tuple(y.shape)}")
    print("Swap SelectiveExponentiation for rd.layers.Power(p) and you have")
    print("power_augmented; any nn.Module fits in the graph.")

    # ------------------------------------------------------------------
    # 4. Parallel reservoirs
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("4. Parallel reservoirs merged by Concatenate")
    print("=" * 70)
    print("Two reservoirs with different timescales read the same input;")
    print("the readout mixes fast and slow features.")

    inp = Input((100, 3))
    fast = ESNLayer(120, feedback_size=3, leak_rate=1.0, spectral_radius=0.7)(inp)
    slow = ESNLayer(120, feedback_size=3, leak_rate=0.2, spectral_radius=0.95)(inp)
    merged = Concatenate()(fast, slow)  # (batch, time, 240)
    out = CGReadoutLayer(240, 3, name="output")(merged)
    parallel = ESNModel(inp, out)

    y = parallel(torch.randn(2, 100, 3))
    print(f"forward: (2, 100, 3) -> fast (.,.,120) + slow (.,.,120) -> {tuple(y.shape)}")

    # ------------------------------------------------------------------
    # 5. Multi-readout DAG + training
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("5. Multi-readout DAG trained in one fit() call")
    print("=" * 70)
    print("Readout names are the keys of the targets dict. ESNTrainer fits")
    print("each readout in graph execution order, so the 'coords' output is")
    print("already fitted when the second reservoir consumes it.")

    inp = Input((100, 3))
    res1 = ESNLayer(200, feedback_size=3)(inp)
    coords = CGReadoutLayer(200, 3, name="coords")(res1)  # intermediate prediction
    res2 = ESNLayer(100, feedback_size=3)(coords)  # stacked reservoir reads it
    energy = CGReadoutLayer(100, 1, name="energy")(res2)  # auxiliary scalar output
    multi = ESNModel(inp, outputs=[coords, energy])

    # Synthetic task: coords = next step of the input, energy = squared norm
    series = torch.cumsum(0.1 * torch.randn(2, 701, 3), dim=1)  # (batch, time, 3)
    warmup_in = series[:, :200]
    train_in = series[:, 200:700]
    coords_target = series[:, 201:701]  # next-step prediction
    energy_target = (train_in**2).sum(dim=-1, keepdim=True)  # (2, 500, 1)

    ESNTrainer(multi).fit(
        warmup_inputs=(warmup_in,),
        train_inputs=(train_in,),
        targets={"coords": coords_target, "energy": energy_target},
    )

    multi.reset_reservoirs()  # always reset before reusing a stateful model
    y_coords, y_energy = multi(series[:, :300])
    print(f"outputs: coords {tuple(y_coords.shape)}, energy {tuple(y_energy.shape)}")

    print("\nDone. Topologies and initializers for these layers: 04_topologies_and_initializers.py")


if __name__ == "__main__":
    main()
