"""File I/O utilities for loading and saving time series data.

Supports multiple file formats: CSV, NPY, NPZ, and NetCDF.
All load functions return tensors with shape (B, T, D) where:
- B = batch dimension (typically 1 for single files)
- T = time steps
- D = feature dimension
"""

from pathlib import Path

import numpy as np
import torch

# Type alias for path-like objects
PathLike = str | Path


def _ensure_3d(data: np.ndarray, source: str) -> np.ndarray:
    """Ensure data has shape (B, T, D).

    Parameters
    ----------
    data : np.ndarray
        Input data of shape (T,), (T, D), or (B, T, D).
    source : str
        Source description for error messages.

    Returns
    -------
    np.ndarray
        Data reshaped to (B, T, D).

    Raises
    ------
    ValueError
        If data has more than 3 dimensions.
    """
    if data.ndim == 1:
        # (T,) -> (1, T, 1)
        return data.reshape(1, -1, 1)
    elif data.ndim == 2:
        # (T, D) -> (1, T, D)
        return data.reshape(1, data.shape[0], data.shape[1])
    elif data.ndim == 3:
        return data
    else:
        raise ValueError(
            f"{source} has unsupported shape {data.shape}. "
            "Expected 1D (T,), 2D (T, D), or 3D (B, T, D)."
        )


def load_csv(path: PathLike, delimiter: str = ",") -> torch.Tensor:
    """Load time series data from a CSV file.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.
    delimiter : str, default=","
        Field delimiter.

    Returns
    -------
    torch.Tensor
        Data tensor with shape (1, T, D).

    Notes
    -----
    Expects headerless CSV with numeric values only.
    """
    data = np.loadtxt(path, delimiter=delimiter, dtype=np.float32)
    data = _ensure_3d(data, f"CSV file '{path}'")
    return torch.from_numpy(data)


def load_npy(path: PathLike) -> torch.Tensor:
    """Load time series data from a NumPy .npy file.

    Parameters
    ----------
    path : str or Path
        Path to the .npy file.

    Returns
    -------
    torch.Tensor
        Data tensor with shape (B, T, D).
    """
    data = np.load(path).astype(np.float32)
    data = _ensure_3d(data, f"NPY file '{path}'")
    return torch.from_numpy(data)


def load_npz(path: PathLike, key: str = "data") -> torch.Tensor:
    """Load time series data from a NumPy .npz file.

    Parameters
    ----------
    path : str or Path
        Path to the .npz file.
    key : str, default="data"
        Key to access the data array in the .npz file.

    Returns
    -------
    torch.Tensor
        Data tensor with shape (B, T, D).

    Raises
    ------
    KeyError
        If the specified key is not found in the file.
    """
    data_dict = np.load(path)
    if key not in data_dict:
        available = list(data_dict.keys())
        raise KeyError(f"Key '{key}' not found in '{path}'. Available keys: {available}")

    data = data_dict[key].astype(np.float32)
    data = _ensure_3d(data, f"NPZ file '{path}'")
    return torch.from_numpy(data)


def load_nc(path: PathLike) -> torch.Tensor:
    """Load time series data from a NetCDF (.nc) file.

    Parameters
    ----------
    path : str or Path
        Path to the .nc file.

    Returns
    -------
    torch.Tensor
        Data tensor with shape (B, T, D).

    Notes
    -----
    Requires xarray to be installed.
    """
    try:
        import xarray as xr
    except ImportError as e:
        raise ImportError(
            "xarray is required to load NetCDF files. Install with: pip install xarray netcdf4"
        ) from e

    data = xr.open_dataarray(path).to_numpy().astype(np.float32)
    data = _ensure_3d(data, f"NetCDF file '{path}'")
    return torch.from_numpy(data)


def load_file(path: PathLike, **kwargs) -> torch.Tensor:
    """Load time series data from a file, detecting format from extension.

    Parameters
    ----------
    path : str or Path
        Path to the data file. Supported extensions: .csv, .npy, .npz, .nc
    **kwargs
        Additional arguments passed to the specific loader (e.g., key for .npz).

    Returns
    -------
    torch.Tensor
        Data tensor with shape (B, T, D).

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    loaders = {
        ".csv": load_csv,
        ".npy": load_npy,
        ".npz": load_npz,
        ".nc": load_nc,
    }

    if suffix not in loaders:
        raise ValueError(
            f"Unsupported file extension '{suffix}' for '{path}'. Supported: {list(loaders.keys())}"
        )

    loader = loaders[suffix]

    # Only pass kwargs if the loader accepts them
    if suffix == ".npz":
        return loader(path, **kwargs)
    return loader(path)


def save_csv(data: torch.Tensor, path: PathLike, delimiter: str = ",") -> None:
    """Save data to a CSV file.

    Parameters
    ----------
    data : torch.Tensor
        Data to save. Will be squeezed to 2D (T, D) if 3D with B=1.
    path : str or Path
        Path to save the CSV file.
    delimiter : str, default=","
        Field delimiter.

    Raises
    ------
    ValueError
        If data cannot be represented as 2D.
    """
    arr = data.detach().cpu().numpy()

    # Squeeze batch dim if B=1
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr.squeeze(0)

    if arr.ndim != 2:
        raise ValueError(f"CSV saving requires 2D data, got shape {arr.shape}")

    np.savetxt(path, arr, delimiter=delimiter)


def save_npy(data: torch.Tensor, path: PathLike) -> None:
    """Save data to a NumPy .npy file.

    Parameters
    ----------
    data : torch.Tensor
        Data to save.
    path : str or Path
        Path to save the .npy file.
    """
    np.save(path, data.detach().cpu().numpy())


def save_npz(data: torch.Tensor, path: PathLike, key: str = "data") -> None:
    """Save data to a NumPy .npz file.

    Parameters
    ----------
    data : torch.Tensor
        Data to save.
    path : str or Path
        Path to save the .npz file.
    key : str, default="data"
        Key to use when storing the data.
    """
    np.savez(path, **{key: data.detach().cpu().numpy()})


def save_nc(data: torch.Tensor, path: PathLike) -> None:
    """Save data to a NetCDF (.nc) file.

    Parameters
    ----------
    data : torch.Tensor
        Data to save.
    path : str or Path
        Path to save the .nc file.

    Notes
    -----
    Requires xarray to be installed.
    """
    try:
        import xarray as xr
    except ImportError as e:
        raise ImportError(
            "xarray is required to save NetCDF files. Install with: pip install xarray netcdf4"
        ) from e

    arr = data.detach().cpu().numpy()
    xr.DataArray(arr).to_netcdf(path)


def list_files(directory: PathLike, extensions: list[str] | None = None) -> list[Path]:
    """List files in a directory, optionally filtering by extension.

    Parameters
    ----------
    directory : str or Path
        Path to the directory.
    extensions : list of str, optional
        List of extensions to filter (e.g., [".csv", ".npy"]).
        If None, returns all files.

    Returns
    -------
    list[Path]
        List of file paths (excluding directories).
    """
    directory = Path(directory)
    files = [f for f in directory.iterdir() if f.is_file()]

    if extensions:
        # Normalize extensions to lowercase with leading dot
        exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
        files = [f for f in files if f.suffix.lower() in exts]

    return sorted(files)
