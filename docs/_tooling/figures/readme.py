"""README diagrams: two hand-wired DAGs that show composition, not factories."""

from _style import FIG


def main() -> None:
    from resdag import CGReadoutLayer, Concatenate, ESNModel, Power, reservoir_input
    from resdag.layers import ESNLayer

    out = FIG / "readme"
    out.mkdir(parents=True, exist_ok=True)

    # 1. Two reservoirs on different timescales, read out together
    inp = reservoir_input(3)
    fast = ESNLayer(64, feedback_size=3, leak_rate=1.0, spectral_radius=0.9)(inp)
    slow = ESNLayer(64, feedback_size=3, leak_rate=0.2, spectral_radius=0.9)(inp)
    merged = Concatenate()(fast, slow)
    head = CGReadoutLayer(128, 3, name="output")(merged)
    ESNModel(inp, head).plot_model(
        show_shapes=False, rankdir="LR",
        save_path=str(out / "parallel_timescales.svg"), format="svg")

    # 2. One reservoir, augmented features, two readout heads
    inp2 = reservoir_input(3)
    states = ESNLayer(96, feedback_size=3, spectral_radius=0.9)(inp2)
    augmented = Concatenate()(states, Power(exponent=2.0)(states))
    pos = CGReadoutLayer(192, 3, name="position")(augmented)
    energy = CGReadoutLayer(192, 1, name="energy")(augmented)
    ESNModel(inp2, [pos, energy]).plot_model(
        show_shapes=False, rankdir="LR",
        save_path=str(out / "augmented_two_heads.svg"), format="svg")

    print("  wrote readme/parallel_timescales.svg, readme/augmented_two_heads.svg")


if __name__ == "__main__":
    main()
