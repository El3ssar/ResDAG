"""Tests for resdag.utils.data.io loaders.

Focuses on the (timesteps, features) shape convention, especially the
single-row / single-column CSV and 1D npy/npz/nc edge cases (issue #141).
"""

import importlib
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from resdag.utils.data import load_and_prepare
from resdag.utils.data.io import (
    _ensure_3d,
    _torch_to_numpy_dtype,
    list_files,
    load_csv,
    load_file,
    load_nc,
    load_npy,
    load_npz,
    save_csv,
    save_nc,
    save_npy,
    save_npz,
)


def _write_csv(path: Path, rows: list[list[float]]) -> None:
    """Write ``rows`` as a headerless comma-delimited CSV at ``path``."""
    np.savetxt(path, np.asarray(rows, dtype=np.float64), delimiter=",")


def test_load_csv_single_row_multi_col(tmp_path: Path) -> None:
    """A 1-row N-col CSV is one timestep of N features -> (1, 1, N)."""
    csv_path = tmp_path / "single_row.csv"
    _write_csv(csv_path, [[1.0, 2.0, 3.0]])

    out = load_csv(csv_path)

    assert out.shape == (1, 1, 3)
    assert torch.allclose(out[0, 0], torch.tensor([1.0, 2.0, 3.0], dtype=out.dtype))


def test_load_csv_single_col_multi_row(tmp_path: Path) -> None:
    """A 1-col T-row CSV is T timesteps of one feature -> (1, T, 1)."""
    csv_path = tmp_path / "single_col.csv"
    _write_csv(csv_path, [[1.0], [2.0], [3.0], [4.0]])

    out = load_csv(csv_path)

    assert out.shape == (1, 4, 1)
    assert torch.allclose(out[0, :, 0], torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=out.dtype))


def test_load_csv_single_cell(tmp_path: Path) -> None:
    """A 1x1 CSV stays (1, 1, 1) rather than collapsing to a scalar."""
    csv_path = tmp_path / "single_cell.csv"
    _write_csv(csv_path, [[42.0]])

    out = load_csv(csv_path)

    assert out.shape == (1, 1, 1)
    assert out[0, 0, 0].item() == pytest.approx(42.0)


def test_load_csv_multi_row_multi_col_unchanged(tmp_path: Path) -> None:
    """Regular (T, D) grids keep the time axis first, features second."""
    csv_path = tmp_path / "grid.csv"
    rows = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
    _write_csv(csv_path, rows)

    out = load_csv(csv_path)

    assert out.shape == (1, 4, 3)
    assert torch.allclose(out[0], torch.tensor(rows, dtype=out.dtype))


def test_load_npy_1d_warns_and_is_univariate(tmp_path: Path) -> None:
    """A 1D npy is interpreted as a univariate series (T,) -> (1, T, 1), with a warning."""
    npy_path = tmp_path / "series.npy"
    np.save(npy_path, np.arange(4, dtype=np.float64))

    with pytest.warns(UserWarning, match="univariate series"):
        out = load_npy(npy_path)

    assert out.shape == (1, 4, 1)
    assert torch.allclose(out[0, :, 0], torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=out.dtype))


def test_load_npy_2d_no_warning(tmp_path: Path) -> None:
    """A 2D npy (T, D) loads to (1, T, D) without warning."""
    npy_path = tmp_path / "matrix.npy"
    np.save(npy_path, np.arange(6, dtype=np.float64).reshape(3, 2))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = load_npy(npy_path)

    assert out.shape == (1, 3, 2)


def test_load_npz_1d_warns(tmp_path: Path) -> None:
    """A 1D array stored in an npz also emits the univariate warning."""
    npz_path = tmp_path / "series.npz"
    np.savez(npz_path, data=np.arange(5, dtype=np.float64))

    with pytest.warns(UserWarning, match="univariate series"):
        out = load_npz(npz_path)

    assert out.shape == (1, 5, 1)


# ---------------------------------------------------------------------------
# save -> load round-trips through load_file / save_* (csv, npy, npz)
# ---------------------------------------------------------------------------


def _grid() -> torch.Tensor:
    """A representative (1, 5, 3) tensor for round-trip checks."""
    return torch.arange(15, dtype=torch.float64).reshape(1, 5, 3)


def test_save_load_csv_round_trip(tmp_path: Path) -> None:
    """``save_csv`` squeezes B=1 and ``load_file`` restores the (1, T, D) tensor."""
    data = _grid()
    path = tmp_path / "rt.csv"
    save_csv(data, path)

    out = load_file(path, dtype=torch.float64)
    assert out.shape == data.shape
    assert torch.allclose(out, data)


def test_save_load_npy_round_trip(tmp_path: Path) -> None:
    """``save_npy`` / ``load_file`` round-trips a 3D tensor unchanged."""
    data = _grid()
    path = tmp_path / "rt.npy"
    save_npy(data, path)

    out = load_file(path, dtype=torch.float64)
    assert out.shape == data.shape
    assert torch.allclose(out, data)


def test_save_load_npz_round_trip(tmp_path: Path) -> None:
    """``save_npz`` / ``load_file`` round-trips via the default ``data`` key."""
    data = _grid()
    path = tmp_path / "rt.npz"
    save_npz(data, path)

    out = load_file(path, dtype=torch.float64)
    assert out.shape == data.shape
    assert torch.allclose(out, data)


def test_save_load_npz_custom_key_round_trip(tmp_path: Path) -> None:
    """A custom npz key is honoured by both the saver and ``load_file``."""
    data = _grid()
    path = tmp_path / "keyed.npz"
    save_npz(data, path, key="series")

    out = load_file(path, dtype=torch.float64, key="series")
    assert torch.allclose(out, data)


def test_save_csv_rejects_non_2d() -> None:
    """A multi-batch tensor cannot be flattened to CSV and raises ``ValueError``."""
    data = torch.zeros(2, 5, 3)
    with pytest.raises(ValueError, match="2D"):
        save_csv(data, "unused.csv")


# ---------------------------------------------------------------------------
# load_file dispatch + dtype handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_load_file_respects_dtype(tmp_path: Path, dtype: torch.dtype) -> None:
    """``load_file`` casts to the requested dtype across formats."""
    data = _grid()
    npy_path = tmp_path / "typed.npy"
    save_npy(data, npy_path)

    out = load_file(npy_path, dtype=dtype)
    assert out.dtype == dtype


def test_load_file_defaults_to_global_dtype(tmp_path: Path) -> None:
    """With ``dtype=None`` the loader uses ``torch.get_default_dtype()``."""
    data = _grid()
    npy_path = tmp_path / "default.npy"
    save_npy(data, npy_path)

    out = load_file(npy_path)
    assert out.dtype == torch.get_default_dtype()


def test_load_file_unsupported_extension_raises(tmp_path: Path) -> None:
    """An unknown extension raises ``ValueError`` listing the supported ones."""
    bogus = tmp_path / "data.parquet"
    bogus.write_text("not real data")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        load_file(bogus)


def test_load_file_extension_is_case_insensitive(tmp_path: Path) -> None:
    """The extension dispatch lowercases the suffix (``.NPY`` -> ``.npy``)."""
    data = _grid()
    path = tmp_path / "upper.NPY"
    # Write to an explicit handle: ``np.save`` would otherwise append ``.npy``
    # to a path whose suffix it doesn't already recognize.
    with open(path, "wb") as fh:
        np.save(fh, data.numpy())

    out = load_file(path, dtype=torch.float64)
    assert torch.allclose(out, data)


def test_load_npz_missing_key_raises(tmp_path: Path) -> None:
    """Requesting an absent npz key raises ``KeyError`` with the available keys."""
    path = tmp_path / "wrongkey.npz"
    np.savez(path, payload=np.arange(6, dtype=np.float64).reshape(3, 2))

    with pytest.raises(KeyError, match="not found"):
        load_npz(path, key="data")
    # The dispatch path surfaces the same error.
    with pytest.raises(KeyError, match="not found"):
        load_file(path, key="data")


# ---------------------------------------------------------------------------
# Internal helpers: dtype mapping and shape coercion guards
# ---------------------------------------------------------------------------


def test_torch_to_numpy_dtype_rejects_unsupported() -> None:
    """Integer torch dtypes are not in the float-only mapping."""
    with pytest.raises(ValueError, match="Unsupported dtype"):
        _torch_to_numpy_dtype(torch.int64)


def test_ensure_3d_rejects_4d() -> None:
    """A 4D array cannot be coerced to (B, T, D) and raises ``ValueError``."""
    with pytest.raises(ValueError, match="unsupported shape"):
        _ensure_3d(np.zeros((1, 2, 3, 4)), "test")


def test_ensure_3d_passes_through_3d() -> None:
    """An already-3D array is returned unchanged."""
    arr = np.zeros((2, 3, 4))
    assert _ensure_3d(arr, "test") is arr


# ---------------------------------------------------------------------------
# NetCDF helpers degrade gracefully when xarray is absent
# ---------------------------------------------------------------------------

_HAS_XARRAY = importlib.util.find_spec("xarray") is not None


@pytest.mark.skipif(_HAS_XARRAY, reason="xarray installed; ImportError path not taken")
def test_load_nc_without_xarray_raises(tmp_path: Path) -> None:
    """``load_nc`` raises a helpful ``ImportError`` when xarray is missing."""
    with pytest.raises(ImportError, match="xarray is required"):
        load_nc(tmp_path / "data.nc")


@pytest.mark.skipif(_HAS_XARRAY, reason="xarray installed; ImportError path not taken")
def test_save_nc_without_xarray_raises(tmp_path: Path) -> None:
    """``save_nc`` raises a helpful ``ImportError`` when xarray is missing."""
    with pytest.raises(ImportError, match="xarray is required"):
        save_nc(_grid(), tmp_path / "data.nc")


@pytest.mark.skipif(not _HAS_XARRAY, reason="xarray not installed")
def test_save_load_nc_round_trip(tmp_path: Path) -> None:
    """``save_nc`` / ``load_file`` round-trips a 3D tensor when xarray is present."""
    data = _grid()
    path = tmp_path / "rt.nc"
    save_nc(data, path)

    out = load_file(path, dtype=torch.float64)
    assert out.shape == data.shape
    assert torch.allclose(out, data)


# ---------------------------------------------------------------------------
# list_files
# ---------------------------------------------------------------------------


def test_list_files_returns_sorted_files_only(tmp_path: Path) -> None:
    """``list_files`` returns files (not subdirectories), sorted by path."""
    (tmp_path / "b.csv").write_text("1\n")
    (tmp_path / "a.npy").write_text("x")
    (tmp_path / "sub").mkdir()

    files = list_files(tmp_path)
    names = [f.name for f in files]
    assert names == ["a.npy", "b.csv"]
    assert all(f.is_file() for f in files)


def test_list_files_filters_by_extension(tmp_path: Path) -> None:
    """Extension filtering normalizes case and a missing leading dot."""
    (tmp_path / "x.csv").write_text("1\n")
    (tmp_path / "y.NPY").write_text("x")
    (tmp_path / "z.npz").write_text("x")

    # Mix a dotted lowercase ext with a bare uppercase ext.
    files = list_files(tmp_path, extensions=[".csv", "NPY"])
    names = sorted(f.name for f in files)
    assert names == ["x.csv", "y.NPY"]


def test_list_files_no_filter_returns_all(tmp_path: Path) -> None:
    """With no extension filter every file is returned."""
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "b.bin").write_text("x")

    files = list_files(tmp_path)
    assert len(files) == 2


# ---------------------------------------------------------------------------
# load_and_prepare: end-to-end load + split, with multi-file batching
# ---------------------------------------------------------------------------


def _write_series_npy(path: Path, offset: float, timesteps: int = 120) -> torch.Tensor:
    """Write a (T, 2) ramp offset by ``offset`` and return the (1, T, 2) tensor."""
    base = np.arange(timesteps, dtype=np.float64)[:, None] + np.array([0.0, 100.0])
    arr = base + offset
    np.save(path, arr)
    return torch.from_numpy(arr).reshape(1, timesteps, 2)


def test_load_and_prepare_single_file(tmp_path: Path) -> None:
    """A single path loads, splits, and matches the in-memory split exactly."""
    path = tmp_path / "single.npy"
    data = _write_series_npy(path, offset=0.0)

    warmup, train, target, f_warmup, val = load_and_prepare(
        str(path),
        warmup_steps=20,
        train_steps=60,
        val_steps=20,
        dtype=torch.float64,
    )

    assert warmup.shape == (1, 20, 2)
    assert train.shape == (1, 60, 2)
    assert val.shape == (1, 20, 2)
    # target is the one-step-ahead of train.
    assert torch.allclose(target[:, 0, :], data[:, 21, :])


def test_load_and_prepare_two_files_batched(tmp_path: Path) -> None:
    """Two paths concatenate along the batch dim before splitting."""
    p0 = tmp_path / "traj0.npy"
    p1 = tmp_path / "traj1.npy"
    d0 = _write_series_npy(p0, offset=0.0)
    d1 = _write_series_npy(p1, offset=1000.0)

    warmup, train, target, f_warmup, val = load_and_prepare(
        [str(p0), str(p1)],
        warmup_steps=20,
        train_steps=60,
        val_steps=20,
        dtype=torch.float64,
    )

    # Batch dimension is 2 (one per file).
    assert warmup.shape[0] == 2
    assert train.shape == (2, 60, 2)
    # Each batch row corresponds to its source trajectory.
    assert torch.allclose(warmup[0], d0[:, :20, :].squeeze(0))
    assert torch.allclose(warmup[1], d1[:, :20, :].squeeze(0))


def test_load_and_prepare_three_files_with_stats(tmp_path: Path) -> None:
    """Multi-file load with normalization returns the fitted stats (6-tuple)."""
    paths = []
    for i in range(3):
        p = tmp_path / f"t{i}.npy"
        _write_series_npy(p, offset=float(i) * 500.0)
        paths.append(str(p))

    result = load_and_prepare(
        paths,
        warmup_steps=20,
        train_steps=60,
        val_steps=20,
        normalize=True,
        norm_method="minmax",
        return_stats=True,
        dtype=torch.float64,
    )

    assert len(result) == 6
    *splits, stats = result
    assert splits[0].shape[0] == 3  # three batched trajectories
    assert isinstance(stats, dict)
    assert set(stats) == {"min", "range"}


def test_load_and_prepare_requires_a_path() -> None:
    """An empty / ``None`` path list raises ``ValueError``."""
    with pytest.raises(ValueError, match="At least one data path"):
        load_and_prepare(None, warmup_steps=1, train_steps=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="At least one data path"):
        load_and_prepare([], warmup_steps=1, train_steps=1)
