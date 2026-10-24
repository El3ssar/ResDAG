"""Small, explicit execution budgets shared by the numbered examples."""

from __future__ import annotations

import argparse
from typing import TypeVar

T = TypeVar("T")


class ExampleOptions(argparse.Namespace):
    quick: bool
    plot: bool

    def choose(self, quick: T, full: T) -> T:
        return quick if self.quick else full


def parse_options(description: str | None, *, plots: bool = False) -> ExampleOptions:
    """Keep full tutorial defaults, with an opt-in execution smoke budget."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run every section with small data/models and a few optimizer steps; "
        "checks execution, not forecast quality or performance",
    )
    if plots:
        parser.add_argument("--plot", action="store_true", help="display optional matplotlib plots")
    options = parser.parse_args(namespace=ExampleOptions())
    if options.quick:
        import torch

        torch.set_num_threads(1)
        print("Quick execution smoke: reduced sizes; no quality or performance claim.")
    return options
