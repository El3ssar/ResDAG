"""Tests for resdag.utils.data.io loaders.

Focuses on the (timesteps, features) shape convention, especially the
single-row / single-column CSV and 1D npy/npz/nc edge cases (issue #141).
"""

import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from resdag.utils.data.io import load_csv, load_npy, load_npz


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
