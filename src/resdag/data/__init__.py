"""
Streaming / SGD Data Pipeline
=============================

PyTorch-native data plumbing for the SGD and streaming-algebraic training paths.
Where :mod:`resdag.utils.data` returns fixed whole-tensor warmup/train/val
splits, this package slices a long trajectory (or many) into fixed-length,
batchable windows behind a standard :class:`torch.utils.data.DataLoader`.

Classes
-------
TimeSeriesWindowDataset
    Sliding-window ``Dataset`` over a ``(T, D)`` tensor, a ``(B, T, D)`` tensor,
    or a list of variable-length trajectories. Yields
    ``(input_window, target_window, washout)`` triples that never straddle a
    trajectory boundary.

Functions
---------
make_dataloader
    Wrap :class:`TimeSeriesWindowDataset` in a ``DataLoader`` whose batches are
    ``(B, window_len, D)`` and feed straight into ``ESNLayer`` / ``ESNModel``.

Examples
--------
>>> import torch
>>> from resdag.data import make_dataloader
>>> series = torch.randn(2000, 3)
>>> loader = make_dataloader(series, batch_size=16, window_len=200, washout=50)
>>> for x, y, washout in loader:  # doctest: +SKIP
...     model.reset_reservoirs()
...     pred = model(x)
...     loss = mse(pred[:, washout:], y[:, washout:])

See Also
--------
resdag.utils.data.prepare_esn_data : Whole-tensor split for the algebraic path.
resdag.layers.IncrementalRidgeReadout : Streaming ridge readout for the loader.
"""

from .dataset import TimeSeriesWindowDataset, make_dataloader

__all__ = [
    "TimeSeriesWindowDataset",
    "make_dataloader",
]


def __dir__() -> list[str]:
    """Restrict ``dir()`` / tab-completion to the public API (:pep:`562`)."""
    return list(__all__)
