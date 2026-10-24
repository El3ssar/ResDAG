"""Temporary inference mode without changing a caller's mixed module modes."""

from collections.abc import Iterator
from contextlib import contextmanager

from torch import nn


@contextmanager
def evaluating(module: nn.Module) -> Iterator[None]:
    """Enter eval mode and restore each existing module's training flag.

    Restoring only the root with `train(previous)` would also overwrite
    children deliberately kept in a different mode, such as frozen backbones.
    This context does not change gradient tracking or recurrent states.
    """
    modes = [(child, child.training) for child in module.modules()]
    try:
        module.eval()
        yield
    finally:
        for child, training in modes:
            child.training = training
