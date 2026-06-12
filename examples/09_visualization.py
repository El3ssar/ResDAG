"""09 — Visualization: inspecting model architecture.

What it shows
-------------
1. ``model.summary()`` — Keras-style text table (from pytorch_symbolic)
2. ``model.plot_model()`` — graphviz rendering of the DAG, with
   show_shapes / show_trainable / rankdir options
3. Rendering to a file (svg/png/pdf) instead of opening a viewer

Expected runtime: ~5 s on CPU. Requires the ``graphviz`` Python package
(a core dependency); if the system ``dot`` binary is missing, plot_model
prints the DOT source as a fallback instead of failing.
"""

import tempfile
from pathlib import Path

import torch

from resdag.core import ESNModel, Input
from resdag.layers import CGReadoutLayer, Concatenate, ESNLayer, SelectiveExponentiation
from resdag.models import classic_esn, ott_esn


def main() -> None:
    torch.manual_seed(42)
    tmpdir = Path(tempfile.mkdtemp(prefix="resdag_plots_"))

    # ------------------------------------------------------------------
    # 1. Text summaries
    # ------------------------------------------------------------------
    print("=" * 70)
    print("1. model.summary()")
    print("=" * 70)

    model = classic_esn(reservoir_size=100, feedback_size=1, output_size=1)
    print("\nclassic_esn:")
    model.summary()

    model_ott = ott_esn(reservoir_size=200, feedback_size=3, output_size=3)
    print("\nott_esn (note the SelectiveExponentiation + Concatenate nodes):")
    model_ott.summary()

    # ------------------------------------------------------------------
    # 2. Graphviz plots to file
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. model.plot_model(save_path=...)")
    print("=" * 70)
    print("Without save_path the diagram opens in a viewer (or displays")
    print("inline in Jupyter); with save_path it renders to disk.")

    model_ott.plot_model(save_path=tmpdir / "ott_esn.svg")

    # show_shapes annotates edges, show_trainable marks frozen/trainable
    # nodes with a padlock, rankdir="LR" lays the graph out left-to-right.
    model_ott.plot_model(
        save_path=tmpdir / "ott_esn_detailed.png",
        format="png",
        show_shapes=True,
        show_trainable=True,
        rankdir="LR",
    )

    # ------------------------------------------------------------------
    # 3. A DAG worth plotting
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. A custom multi-branch DAG")
    print("=" * 70)

    inp = Input((100, 3))
    fast = ESNLayer(120, feedback_size=3, leak_rate=1.0)(inp)
    slow = ESNLayer(120, feedback_size=3, leak_rate=0.2)(inp)
    squared = SelectiveExponentiation(index=0, exponent=2.0)(fast)
    merged = Concatenate()(squared, slow)
    coords = CGReadoutLayer(240, 3, name="coords")(merged)
    energy = CGReadoutLayer(240, 1, name="energy")(merged)
    dag = ESNModel(inp, outputs=[coords, energy])

    dag.summary()
    dag.plot_model(save_path=tmpdir / "custom_dag.svg", show_shapes=True)

    print("\nRendered files:")
    for f in sorted(tmpdir.iterdir()):
        print(f"  {f}  ({f.stat().st_size / 1024:.1f} KiB)")
    print("\nOpen them in any browser / image viewer.")


if __name__ == "__main__":
    main()
