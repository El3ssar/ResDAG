"""
Windowed Time-Series Dataset
============================

This module provides the data half of resdag's frictionless plain-PyTorch / SGD
story: a :class:`torch.utils.data.Dataset` that slices a long trajectory (or a
batch of trajectories, or a ragged list of variable-length trajectories) into
fixed-length, batchable windows, and a :func:`make_dataloader` convenience that
wraps it in a :class:`torch.utils.data.DataLoader` with a collate that stacks
windows to a ``(B, window_len, D)`` tensor — exactly the shape
:class:`~resdag.layers.ESNLayer` / :class:`~resdag.core.ESNModel` expect.

Each item is an ``(input_window, target_window)`` pair where the target is the
input shifted forward by ``horizon`` steps (the forecasting default) or sliced
from an external ``targets`` tensor (the regression case). A ``washout`` field
marks the leading steps of every window that should be excluded from the loss
(the reservoir state has not yet synchronised). Because windows never straddle
trajectory boundaries, a per-window ``reset_reservoirs()`` plus a post-washout
mask keeps state from leaking across unrelated trajectories.

Canonical SGD loop
------------------
The intended training loop over a :func:`make_dataloader` is::

    loader = make_dataloader(series, batch_size=16, window_len=200, washout=50)
    for epoch in range(n_epochs):
        for x, y, washout in loader:           # x, y: (B, window_len, D)
            model.reset_reservoirs()           # each window is independent
            pred = model(x)                    # (B, window_len, D_out)
            loss = mse(pred[:, washout:], y[:, washout:])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

``ESNLayer.detach_state_between_calls`` (on by default) severs the autograd
graph at each forward-call boundary, so even without the per-window
``reset_reservoirs()`` consecutive minibatches do not raise "backward through
the graph a second time". Pair this loader with
:class:`~resdag.layers.IncrementalRidgeReadout` for an *algebraic*
over-:class:`DataLoader` path — call ``partial_fit(states, targets)`` per batch,
then ``finalize()`` once.

See Also
--------
resdag.utils.data.prepare_esn_data : Whole-tensor warmup/train/val splitting.
resdag.layers.IncrementalRidgeReadout : Streaming ridge readout for the loader.
resdag.layers.ESNLayer.detach_state_between_calls : Cross-batch graph hygiene.
"""

from collections.abc import Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from resdag.utils.data.prepare import NormMethod

__all__ = ["TimeSeriesWindowDataset", "make_dataloader"]

# A batch item: (input_window, target_window, washout_for_this_window).
WindowItem = tuple[torch.Tensor, torch.Tensor, int]
# A collated batch: stacked inputs/targets plus the (shared) washout length.
WindowBatch = tuple[torch.Tensor, torch.Tensor, int]


def _as_trajectory_list(
    series: torch.Tensor | Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    """Normalise any accepted ``series`` form to a list of ``(T, D)`` tensors.

    Accepts a single ``(T, D)`` tensor, a batched ``(B, T, D)`` tensor (each of
    the ``B`` slices becomes one trajectory), or an already-ragged sequence of
    ``(T, D)`` tensors of possibly differing length. Every trajectory must be
    2-D ``(T, D)`` and share the same feature dimension ``D``.

    Parameters
    ----------
    series : torch.Tensor or sequence of torch.Tensor
        ``(T, D)`` tensor, ``(B, T, D)`` tensor, or a list of ``(T, D)``
        tensors of varying length.

    Returns
    -------
    list of torch.Tensor
        One ``(T, D)`` tensor per trajectory.

    Raises
    ------
    ValueError
        If ``series`` is empty, has an unsupported number of dimensions, or its
        trajectories disagree on the feature dimension.
    TypeError
        If ``series`` is neither a tensor nor a sequence of tensors.
    """
    if isinstance(series, torch.Tensor):
        if series.dim() == 2:
            trajectories = [series]
        elif series.dim() == 3:
            trajectories = [series[i] for i in range(series.shape[0])]
        else:
            raise ValueError(
                f"series tensor must be 2-D (T, D) or 3-D (B, T, D), got {series.dim()}-D "
                f"with shape {tuple(series.shape)}."
            )
    elif isinstance(series, Sequence):
        if len(series) == 0:
            raise ValueError("series sequence is empty; provide at least one trajectory.")
        trajectories = []
        for i, traj in enumerate(series):
            if not isinstance(traj, torch.Tensor):
                raise TypeError(f"series[{i}] must be a torch.Tensor, got {type(traj).__name__}.")
            if traj.dim() != 2:
                raise ValueError(
                    f"series[{i}] must be a 2-D (T, D) trajectory, got {traj.dim()}-D "
                    f"with shape {tuple(traj.shape)}."
                )
            trajectories.append(traj)
    else:
        raise TypeError(
            f"series must be a torch.Tensor or a sequence of torch.Tensor, "
            f"got {type(series).__name__}."
        )

    feature_dim = trajectories[0].shape[-1]
    for i, traj in enumerate(trajectories):
        if traj.shape[-1] != feature_dim:
            raise ValueError(
                f"all trajectories must share the feature dimension; trajectory 0 has "
                f"D={feature_dim} but trajectory {i} has D={traj.shape[-1]}."
            )
    return trajectories


class TimeSeriesWindowDataset(Dataset[WindowItem]):
    """Sliding-window :class:`~torch.utils.data.Dataset` over time-series data.

    Slices one or more trajectories into fixed-length windows suitable for SGD
    (or streaming-algebraic) training of a reservoir model. Each item is an
    ``(input_window, target_window, washout)`` triple where ``input_window`` and
    ``target_window`` both have shape ``(window_len, D)``:

    - **Forecasting** (default, no external ``targets``): the target is the input
      shifted forward by ``horizon`` steps, so window ``i`` covers source steps
      ``[start, start + window_len)`` for the input and
      ``[start + horizon, start + horizon + window_len)`` for the target.
    - **Regression** (external ``targets`` given): the target is the matching
      window sliced from ``targets`` at the same source positions as the input
      (``horizon`` is ignored — supply pre-aligned targets).

    Windows are generated **per trajectory** and never straddle a trajectory
    boundary, so feeding the loader's batches with a per-batch
    ``reset_reservoirs()`` (or relying on
    :attr:`~resdag.layers.BaseReservoirLayer.detach_state_between_calls`) keeps
    reservoir state from leaking across unrelated trajectories. The first
    ``washout`` steps of every window are reported in the returned triple so the
    caller can exclude them from the loss while the reservoir synchronises.

    Parameters
    ----------
    series : torch.Tensor or sequence of torch.Tensor
        Source data. A ``(T, D)`` tensor (one trajectory), a ``(B, T, D)``
        tensor (``B`` trajectories), or a list of ``(T, D)`` tensors of varying
        length (ragged trajectories).
    window_len : int
        Number of timesteps in each input/target window. Must be positive.
    horizon : int, default=1
        Forecast offset: the target window is the input window shifted forward
        by ``horizon`` steps. Ignored when ``targets`` is provided. Must be
        non-negative.
    stride : int, default=1
        Step between successive window start positions within a trajectory. Must
        be positive.
    washout : int, default=0
        Number of leading steps in each window the caller should exclude from
        the loss (transient reservoir warmup). Reported per item; must satisfy
        ``0 <= washout < window_len``.
    targets : torch.Tensor or sequence of torch.Tensor, optional
        External regression targets aligned step-for-step with ``series``. Must
        have the same form and per-trajectory length as ``series`` (the feature
        dimension may differ). When given, the dataset returns input/target
        windows sliced from the *same* source positions and ``horizon`` is
        ignored.
    stats : dict, optional
        Pre-computed normalization statistics from
        :func:`resdag.utils.data.normalize_data`. When given, every window is
        normalized on access with the matching method. Requires ``norm_method``.
    norm_method : {"minmax", "standard", "noncentered", "meanpreserving"}, optional
        Normalization method matching ``stats``. Required when ``stats`` is
        given, ignored otherwise.

    Attributes
    ----------
    window_len : int
        Window length.
    horizon : int
        Forecast offset (forecasting mode only).
    stride : int
        Window start spacing.
    washout : int
        Leading steps to exclude from the loss.
    feature_dim : int
        Input feature dimension ``D``.
    target_dim : int
        Target feature dimension (equals ``feature_dim`` in forecasting mode).

    Raises
    ------
    ValueError
        For invalid ``window_len``/``horizon``/``stride``/``washout``, mismatched
        ``targets``, or an incomplete ``stats``/``norm_method`` pair.

    Examples
    --------
    Forecasting windows over a single ``(T, D)`` series:

    >>> import torch
    >>> from resdag.data import TimeSeriesWindowDataset
    >>> series = torch.randn(1000, 3)
    >>> ds = TimeSeriesWindowDataset(series, window_len=200, horizon=1, washout=50)
    >>> x, y, washout = ds[0]
    >>> x.shape, y.shape, washout
    (torch.Size([200, 3]), torch.Size([200, 3]), 50)

    See Also
    --------
    make_dataloader : Wrap this dataset in a batched ``DataLoader``.
    resdag.layers.IncrementalRidgeReadout : Algebraic over-``DataLoader`` fit.
    """

    def __init__(
        self,
        series: torch.Tensor | Sequence[torch.Tensor],
        window_len: int,
        horizon: int = 1,
        stride: int = 1,
        washout: int = 0,
        targets: torch.Tensor | Sequence[torch.Tensor] | None = None,
        stats: dict[str, torch.Tensor] | None = None,
        norm_method: NormMethod | None = None,
    ) -> None:
        if window_len <= 0:
            raise ValueError(f"window_len must be positive, got {window_len}.")
        if horizon < 0:
            raise ValueError(f"horizon must be non-negative, got {horizon}.")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}.")
        if not 0 <= washout < window_len:
            raise ValueError(
                f"washout must satisfy 0 <= washout < window_len, got washout={washout}, "
                f"window_len={window_len}."
            )
        if (stats is None) != (norm_method is None):
            raise ValueError(
                "stats and norm_method must be provided together; got "
                f"stats={'set' if stats is not None else None}, norm_method={norm_method!r}."
            )

        self.window_len = window_len
        self.horizon = horizon
        self.stride = stride
        self.washout = washout
        self._stats = stats
        self._norm_method = norm_method

        self._inputs = _as_trajectory_list(series)
        self.feature_dim = self._inputs[0].shape[-1]

        self._has_external_targets = targets is not None
        if targets is not None:
            self._targets: list[torch.Tensor] = _as_trajectory_list(targets)
            if len(self._targets) != len(self._inputs):
                raise ValueError(
                    f"series has {len(self._inputs)} trajectories but targets has "
                    f"{len(self._targets)}; they must match."
                )
            for i, (x_traj, y_traj) in enumerate(zip(self._inputs, self._targets)):
                if x_traj.shape[0] != y_traj.shape[0]:
                    raise ValueError(
                        f"trajectory {i}: series length ({x_traj.shape[0]}) and targets "
                        f"length ({y_traj.shape[0]}) must match."
                    )
            self.target_dim = self._targets[0].shape[-1]
            # Targets are pre-aligned; no forecast offset is applied.
            self._offset = 0
        else:
            self._targets = self._inputs
            self.target_dim = self.feature_dim
            self._offset = horizon

        # Precompute (trajectory index, window start) for every valid window so
        # __getitem__ is O(1) and windows never cross a trajectory boundary.
        # A window of source span [start, start + window_len) plus the target
        # offset must fit inside the trajectory: start + window_len + offset <= T.
        self._index: list[tuple[int, int]] = []
        for t_idx, traj in enumerate(self._inputs):
            max_start = traj.shape[0] - self.window_len - self._offset
            start = 0
            while start <= max_start:
                self._index.append((t_idx, start))
                start += self.stride

    def __len__(self) -> int:
        """Return the total number of windows across all trajectories."""
        return len(self._index)

    def _normalize(self, window: torch.Tensor) -> torch.Tensor:
        """Normalize ``window`` in place-free fashion using the stored ``stats``.

        Imported lazily so the dataset module does not pull the preparation
        utilities (and their type machinery) at import time.
        """
        if self._stats is None or self._norm_method is None:
            return window
        from resdag.utils.data import normalize_data

        normalized, _ = normalize_data(
            window.unsqueeze(0), method=self._norm_method, stats=self._stats
        )
        return normalized.squeeze(0)

    def __getitem__(self, idx: int) -> WindowItem:
        """Return the ``idx``-th window as ``(input, target, washout)``.

        Parameters
        ----------
        idx : int
            Window index in ``[0, len(self))``.

        Returns
        -------
        input_window : torch.Tensor
            Input window of shape ``(window_len, feature_dim)``.
        target_window : torch.Tensor
            Target window of shape ``(window_len, target_dim)`` — the input
            shifted by ``horizon`` (forecasting) or the matching slice of
            ``targets`` (regression).
        washout : int
            Number of leading steps to exclude from the loss for this window.
        """
        traj_idx, start = self._index[idx]
        x = self._inputs[traj_idx][start : start + self.window_len]
        y_start = start + self._offset
        y = self._targets[traj_idx][y_start : y_start + self.window_len]
        return self._normalize(x), self._normalize(y), self.washout


def _collate_windows(batch: list[WindowItem]) -> WindowBatch:
    """Stack a list of window items into a batched ``(B, window_len, D)`` tuple.

    The per-window ``washout`` is identical across the dataset (it is a dataset
    parameter), so the collated batch carries it once as a scalar rather than a
    per-item tensor.

    Parameters
    ----------
    batch : list of (torch.Tensor, torch.Tensor, int)
        Items produced by :meth:`TimeSeriesWindowDataset.__getitem__`.

    Returns
    -------
    inputs : torch.Tensor
        Stacked inputs of shape ``(B, window_len, feature_dim)``.
    targets : torch.Tensor
        Stacked targets of shape ``(B, window_len, target_dim)``.
    washout : int
        The shared washout length for the batch.
    """
    inputs = torch.stack([item[0] for item in batch], dim=0)
    targets = torch.stack([item[1] for item in batch], dim=0)
    washout = batch[0][2]
    return inputs, targets, washout


def make_dataloader(
    series: torch.Tensor | Sequence[torch.Tensor],
    batch_size: int,
    window_len: int,
    horizon: int = 1,
    stride: int = 1,
    washout: int = 0,
    targets: torch.Tensor | Sequence[torch.Tensor] | None = None,
    stats: dict[str, torch.Tensor] | None = None,
    norm_method: NormMethod | None = None,
    shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
    **dataloader_kwargs: object,
) -> DataLoader[WindowItem]:
    """Build a windowed :class:`~torch.utils.data.DataLoader` over ``series``.

    Convenience wrapper that constructs a :class:`TimeSeriesWindowDataset` from
    ``series`` and the windowing parameters, then returns a
    :class:`~torch.utils.data.DataLoader` whose batches are
    ``(inputs, targets, washout)`` with ``inputs``/``targets`` of shape
    ``(B, window_len, D)`` — ready to feed straight into
    :class:`~resdag.layers.ESNLayer` / :class:`~resdag.core.ESNModel.forward`.

    Parameters
    ----------
    series : torch.Tensor or sequence of torch.Tensor
        Source data; see :class:`TimeSeriesWindowDataset`.
    batch_size : int
        Number of windows per batch.
    window_len : int
        Window length (timesteps per input/target window).
    horizon : int, default=1
        Forecast offset (ignored when ``targets`` is given).
    stride : int, default=1
        Spacing between window starts within a trajectory.
    washout : int, default=0
        Leading steps per window to exclude from the loss.
    targets : torch.Tensor or sequence of torch.Tensor, optional
        External regression targets aligned with ``series``.
    stats : dict, optional
        Pre-computed normalization statistics (with ``norm_method``).
    norm_method : str, optional
        Normalization method matching ``stats``.
    shuffle : bool, default=False
        Whether to shuffle window order each epoch.
    drop_last : bool, default=False
        Whether to drop the final, smaller-than-``batch_size`` batch.
    num_workers : int, default=0
        Number of worker processes for data loading.
    **dataloader_kwargs : object
        Extra keyword arguments forwarded to :class:`~torch.utils.data.DataLoader`
        (e.g. ``pin_memory``, ``generator``). ``collate_fn`` is supplied here and
        must not be overridden.

    Returns
    -------
    torch.utils.data.DataLoader
        Loader yielding ``(inputs, targets, washout)`` batches.

    Examples
    --------
    >>> import torch
    >>> from resdag.data import make_dataloader
    >>> series = torch.randn(1000, 3)
    >>> loader = make_dataloader(series, batch_size=16, window_len=200, washout=50)
    >>> x, y, washout = next(iter(loader))
    >>> x.shape  # (B, window_len, D)
    torch.Size([16, 200, 3])

    See Also
    --------
    TimeSeriesWindowDataset : The underlying windowing dataset.
    """
    dataset = TimeSeriesWindowDataset(
        series,
        window_len=window_len,
        horizon=horizon,
        stride=stride,
        washout=washout,
        targets=targets,
        stats=stats,
        norm_method=norm_method,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=_collate_windows,
        **dataloader_kwargs,  # type: ignore[arg-type]
    )
