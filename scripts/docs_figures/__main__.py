"""Run every figure generator in order.

Usage::

    .venv/bin/python -m scripts.docs_figures
"""

from __future__ import annotations

from ._common import OUT, apply_style
from . import (
    activations,
    architectures,
    hpo,
    initializers,
    lorenz,
    signals,
    sine,
    topologies,
    topology_comparison,
)


def main() -> None:
    apply_style()
    print(f"Writing figures to {OUT}\n")
    architectures.main()
    signals.main()
    activations.main()
    sine.main()
    lorenz.main()
    topologies.main()
    topology_comparison.main()
    initializers.main()
    hpo.main()
    print("\nDone.")


if __name__ == "__main__":
    main()
