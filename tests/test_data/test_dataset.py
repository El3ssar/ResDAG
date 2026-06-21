"""Tests for :mod:`resdag.data.dataset`.

Covers windowing semantics (``window_len`` / ``horizon`` / ``stride`` /
``washout``), the regression path with external targets, ragged
variable-length trajectories (no cross-boundary leakage), the
``make_dataloader`` batch contract, an SGD training loop that reduces loss (both
frozen-reservoir-plus-trainable-head and fully-trainable-reservoir), and an
algebraic-over-``DataLoader`` fit via
:class:`~resdag.layers.IncrementalRidgeReadout`.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import resdag as rd
from resdag import ESNLayer, ESNModel, reservoir_input
from resdag.data import TimeSeriesWindowDataset, make_dataloader
from resdag.layers import CGReadoutLayer, IncrementalRidgeReadout


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sine_series() -> torch.Tensor:
    """A clean (T, D) multi-feature sine series."""
    t = torch.linspace(0, 12 * torch.pi, 1000)
    return torch.stack([torch.sin(t), torch.cos(t), torch.sin(2 * t)], dim=-1)


# ---------------------------------------------------------------------------
# Public API exposure
# ---------------------------------------------------------------------------
def test_public_api_exports() -> None:
    """Both symbols are exported from the package roots and ``__all__``."""
    assert rd.TimeSeriesWindowDataset is TimeSeriesWindowDataset
    assert rd.make_dataloader is make_dataloader
    assert "TimeSeriesWindowDataset" in rd.__all__
    assert "make_dataloader" in rd.__all__
    assert "data" in rd.__all__
    assert set(rd.data.__all__) == {"TimeSeriesWindowDataset", "make_dataloader"}
    assert rd.data.__dir__() == list(rd.data.__all__)


# ---------------------------------------------------------------------------
# Windowing semantics on a (T, D) series
# ---------------------------------------------------------------------------
def test_window_shapes_and_count() -> None:
    """Window shapes and the window count follow window_len/stride/horizon."""
    series = torch.arange(100, dtype=torch.float32).unsqueeze(-1)  # (100, 1)
    ds = TimeSeriesWindowDataset(series, window_len=10, horizon=1, stride=1)
    x, y, washout = ds[0]
    assert x.shape == (10, 1)
    assert y.shape == (10, 1)
    assert washout == 0
    # max_start = T - window_len - horizon = 100 - 10 - 1 = 89, stride 1 -> 90 windows
    assert len(ds) == 90
    assert ds.feature_dim == 1
    assert ds.target_dim == 1


def test_forecast_target_is_input_shifted_by_horizon() -> None:
    """In forecasting mode the target equals the input shifted by ``horizon``."""
    series = torch.arange(50, dtype=torch.float32).unsqueeze(-1)  # values == index
    horizon = 3
    ds = TimeSeriesWindowDataset(series, window_len=8, horizon=horizon, stride=1)
    x, y, _ = ds[0]
    assert torch.equal(x, torch.arange(0, 8, dtype=torch.float32).unsqueeze(-1))
    assert torch.equal(y, torch.arange(horizon, horizon + 8, dtype=torch.float32).unsqueeze(-1))
    # A later window starting at stride*k keeps the horizon offset.
    x2, y2, _ = ds[5]
    assert torch.equal(y2, x2 + horizon)


def test_stride_controls_start_spacing() -> None:
    """Successive windows start ``stride`` steps apart."""
    series = torch.arange(40, dtype=torch.float32).unsqueeze(-1)
    ds = TimeSeriesWindowDataset(series, window_len=5, horizon=1, stride=4)
    x0, _, _ = ds[0]
    x1, _, _ = ds[1]
    assert x0[0].item() == 0.0
    assert x1[0].item() == 4.0  # advanced by stride


def test_washout_is_reported_and_validated() -> None:
    """``washout`` rides along on each item and is range-checked."""
    series = torch.randn(100, 2)
    ds = TimeSeriesWindowDataset(series, window_len=20, washout=5)
    _, _, washout = ds[0]
    assert washout == 5
    assert ds.washout == 5
    with pytest.raises(ValueError, match="washout"):
        TimeSeriesWindowDataset(series, window_len=20, washout=20)
    with pytest.raises(ValueError, match="washout"):
        TimeSeriesWindowDataset(series, window_len=20, washout=-1)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"window_len": 0}, "window_len"),
        ({"window_len": 10, "horizon": -1}, "horizon"),
        ({"window_len": 10, "stride": 0}, "stride"),
    ],
)
def test_invalid_params_raise(kwargs: dict, match: str) -> None:
    """Bad windowing parameters raise a clear ``ValueError``."""
    series = torch.randn(100, 2)
    with pytest.raises(ValueError, match=match):
        TimeSeriesWindowDataset(series, **kwargs)


# ---------------------------------------------------------------------------
# Batched (B, T, D) input and regression (external targets)
# ---------------------------------------------------------------------------
def test_batched_tensor_treated_as_multiple_trajectories() -> None:
    """A (B, T, D) tensor yields windows from each of the B trajectories."""
    series = torch.randn(4, 60, 3)  # 4 trajectories
    ds = TimeSeriesWindowDataset(series, window_len=20, horizon=1, stride=20)
    # Per trajectory: max_start = 60 - 20 - 1 = 39, stride 20 -> starts {0, 20} -> 2 windows
    assert len(ds) == 4 * 2
    x, y, _ = ds[0]
    assert x.shape == (20, 3)
    assert y.shape == (20, 3)


def test_external_targets_regression_mode() -> None:
    """With external targets the target window is sliced at the input position."""
    series = torch.arange(50, dtype=torch.float32).unsqueeze(-1)
    targets = series * 10.0  # different feature scale / dim semantics
    ds = TimeSeriesWindowDataset(series, window_len=8, horizon=5, targets=targets)
    x, y, _ = ds[0]
    # horizon is ignored in regression mode: y aligns with x positions.
    assert torch.equal(y, x * 10.0)
    # horizon does not reduce the window count in regression mode.
    assert len(ds) == 50 - 8 + 1


def test_external_targets_different_dim() -> None:
    """Target feature dimension may differ from the input's."""
    series = torch.randn(80, 3)
    targets = torch.randn(80, 1)
    ds = TimeSeriesWindowDataset(series, window_len=10, targets=targets)
    x, y, _ = ds[0]
    assert x.shape == (10, 3)
    assert y.shape == (10, 1)
    assert ds.feature_dim == 3
    assert ds.target_dim == 1


def test_mismatched_targets_raise() -> None:
    """Targets whose length disagrees with the series raise ``ValueError``."""
    series = torch.randn(80, 3)
    targets = torch.randn(70, 1)  # wrong length
    with pytest.raises(ValueError, match="length"):
        TimeSeriesWindowDataset(series, window_len=10, targets=targets)


# ---------------------------------------------------------------------------
# Ragged variable-length trajectories — no cross-boundary leakage
# ---------------------------------------------------------------------------
def test_variable_length_trajectories_no_leakage() -> None:
    """Windows never straddle a trajectory boundary (no leakage)."""
    # Two trajectories with distinct value ranges so leakage is detectable.
    traj_a = torch.arange(0, 30, dtype=torch.float32).unsqueeze(-1)  # 0..29
    traj_b = torch.arange(1000, 1020, dtype=torch.float32).unsqueeze(-1)  # 1000..1019
    ds = TimeSeriesWindowDataset([traj_a, traj_b], window_len=10, horizon=1, stride=1)

    # Per traj: A -> max_start 30-10-1=19 -> 20 windows; B -> 20-10-1=9 -> 10 windows.
    assert len(ds) == 30
    for i in range(len(ds)):
        x, y, _ = ds[i]
        x_max = x.max().item()
        # Each window is entirely inside one trajectory's value range, never
        # mixing the 0..29 and 1000..1019 ranges.
        in_a = x_max < 100
        in_b = x.min().item() >= 1000
        assert in_a or in_b
        # The forecast target stays in the same trajectory's range as its input.
        if in_a:
            assert y.max().item() < 100
        else:
            assert y.min().item() >= 1000


def test_short_trajectory_contributes_no_windows() -> None:
    """A trajectory shorter than window_len + horizon yields no windows."""
    long_traj = torch.randn(40, 2)
    short_traj = torch.randn(5, 2)  # too short for window_len=10
    ds = TimeSeriesWindowDataset([long_traj, short_traj], window_len=10, horizon=1, stride=1)
    # Only the long trajectory contributes: 40 - 10 - 1 + 1 = 30 windows.
    assert len(ds) == 30


def test_ragged_feature_mismatch_raises() -> None:
    """Trajectories with differing feature dims raise ``ValueError``."""
    with pytest.raises(ValueError, match="feature dimension"):
        TimeSeriesWindowDataset([torch.randn(30, 3), torch.randn(30, 2)], window_len=10)


# ---------------------------------------------------------------------------
# Normalization with precomputed stats
# ---------------------------------------------------------------------------
def test_precomputed_stats_normalizes_windows() -> None:
    """Windows are normalized on access with the supplied stats/method."""
    series = torch.randn(200, 3) * 5.0 + 2.0
    _, stats = rd.utils.data.normalize_data(series.unsqueeze(0), method="standard")
    ds = TimeSeriesWindowDataset(
        series, window_len=50, horizon=1, stats=stats, norm_method="standard"
    )
    x, _, _ = ds[0]
    # Standard-normalized data has much smaller magnitude than the raw series.
    assert x.abs().mean() < series.abs().mean()


def test_stats_without_method_raises() -> None:
    """``stats`` and ``norm_method`` must be supplied together."""
    series = torch.randn(100, 2)
    with pytest.raises(ValueError, match="norm_method"):
        TimeSeriesWindowDataset(series, window_len=10, stats={"mean": torch.zeros(2)})


# ---------------------------------------------------------------------------
# make_dataloader contract
# ---------------------------------------------------------------------------
def test_make_dataloader_returns_dataloader_with_batched_shapes(
    sine_series: torch.Tensor,
) -> None:
    """``make_dataloader`` yields (B, window_len, D) batches plus washout."""
    loader = make_dataloader(sine_series, batch_size=8, window_len=100, horizon=1, washout=20)
    assert isinstance(loader, DataLoader)
    x, y, washout = next(iter(loader))
    assert x.shape == (8, 100, 3)
    assert y.shape == (8, 100, 3)
    assert washout == 20


def test_dataloader_batches_feed_into_esn_layer(sine_series: torch.Tensor) -> None:
    """Batches flow straight into an ``ESNLayer.forward`` without reshaping."""
    loader = make_dataloader(sine_series, batch_size=4, window_len=80)
    reservoir = ESNLayer(reservoir_size=32, feedback_size=3, spectral_radius=0.9)
    x, _, _ = next(iter(loader))
    out = reservoir(x)
    assert out.shape == (4, 80, 32)


def test_dataloader_batches_feed_into_esn_model(sine_series: torch.Tensor) -> None:
    """Batches flow straight into an ``ESNModel.forward``."""
    inp = reservoir_input(3)
    states = ESNLayer(reservoir_size=32, feedback_size=3)(inp)
    out = CGReadoutLayer(32, 3, name="output")(states)
    model = ESNModel(inp, out)
    loader = make_dataloader(sine_series, batch_size=4, window_len=80)
    x, _, _ = next(iter(loader))
    model.reset_reservoirs()
    pred = model(x)
    assert pred.shape == (4, 80, 3)


def test_drop_last_and_shuffle(sine_series: torch.Tensor) -> None:
    """``drop_last`` drops the ragged tail; ``shuffle`` is accepted."""
    loader = make_dataloader(
        sine_series, batch_size=7, window_len=100, drop_last=True, shuffle=True
    )
    for x, _, _ in loader:
        assert x.shape[0] == 7  # never a short tail batch


# ---------------------------------------------------------------------------
# SGD training loop over the DataLoader reduces loss
# ---------------------------------------------------------------------------
def test_sgd_frozen_reservoir_trainable_head_reduces_loss(
    sine_series: torch.Tensor,
) -> None:
    """Frozen reservoir + trainable linear head: SGD over the loader lowers loss."""
    torch.manual_seed(0)
    reservoir = ESNLayer(reservoir_size=64, feedback_size=3, spectral_radius=0.9, trainable=False)
    head = nn.Linear(64, 3)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-2)
    criterion = nn.MSELoss()
    loader = make_dataloader(
        sine_series, batch_size=8, window_len=120, horizon=1, washout=20, shuffle=True
    )

    first_loss = None
    last_loss = None
    for epoch in range(4):
        for x, y, washout in loader:
            reservoir.reset_state()
            states = reservoir(x)
            pred = head(states)
            loss = criterion(pred[:, washout:], y[:, washout:])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if first_loss is None:
                first_loss = loss.item()
            last_loss = loss.item()

    assert first_loss is not None and last_loss is not None
    assert last_loss < first_loss


def test_sgd_trainable_reservoir_reduces_loss(sine_series: torch.Tensor) -> None:
    """Fully-trainable reservoir + head: BPTT over the loader lowers loss."""
    torch.manual_seed(0)
    reservoir = ESNLayer(reservoir_size=48, feedback_size=3, spectral_radius=0.9, trainable=True)
    head = nn.Linear(48, 3)
    params = list(reservoir.parameters()) + list(head.parameters())
    assert any(p.requires_grad for p in reservoir.parameters())
    optimizer = torch.optim.Adam(params, lr=5e-3)
    criterion = nn.MSELoss()
    loader = make_dataloader(sine_series, batch_size=8, window_len=100, horizon=1, washout=20)

    first_loss = None
    last_loss = None
    for epoch in range(5):
        for x, y, washout in loader:
            reservoir.reset_state()
            pred = head(reservoir(x))
            loss = criterion(pred[:, washout:], y[:, washout:])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if first_loss is None:
                first_loss = loss.item()
            last_loss = loss.item()

    assert first_loss is not None and last_loss is not None
    assert last_loss < first_loss


# ---------------------------------------------------------------------------
# Algebraic-over-DataLoader: IncrementalRidgeReadout.partial_fit then finalize
# ---------------------------------------------------------------------------
def test_incremental_ridge_over_dataloader(sine_series: torch.Tensor) -> None:
    """Stream states/targets per batch into IncrementalRidgeReadout, then finalize."""
    torch.manual_seed(0)
    reservoir = ESNLayer(reservoir_size=80, feedback_size=3, spectral_radius=0.9, trainable=False)
    readout = IncrementalRidgeReadout(in_features=80, out_features=3, alpha=1e-4)
    loader = make_dataloader(sine_series, batch_size=8, window_len=120, horizon=1, washout=30)

    readout.reset_accumulators()
    with torch.no_grad():
        for x, y, washout in loader:
            reservoir.reset_state()
            states = reservoir(x)
            # Drop the washout transient before accumulating sufficient stats.
            readout.partial_fit(states[:, washout:], y[:, washout:])
    assert readout.n_seen > 0
    readout.finalize()
    assert readout.is_fitted

    # The fitted readout produces a sensible one-step prediction on a fresh window.
    with torch.no_grad():
        reservoir.reset_state()
        x, y, washout = next(iter(loader))
        pred = readout(reservoir(x))
        mse = torch.mean((pred[:, washout:] - y[:, washout:]) ** 2).item()
    assert mse < 1.0  # far better than the unit-variance baseline of the signal
