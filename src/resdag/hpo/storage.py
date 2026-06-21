"""Optuna storage backend resolution for HPO studies.

This module handles the creation and configuration of Optuna storage backends
used to persist hyperparameter optimization studies. It supports in-memory,
journal file, and SQLite storage, with automatic selection based on the number
of workers and user-provided paths.

See Also
--------
resdag.hpo.run : High-level HPO orchestrator that consumes these utilities.
resdag.hpo.runners : Trial execution backends that reconnect to storage.
"""

import atexit
import logging
import os
import sqlite3
import tempfile
from typing import Literal, overload

import optuna

from optuna.storages import JournalStorage  # isort: skip

try:  # noqa: E402
    from optuna.storages.journal import JournalFileBackend as _JournalFileBackend
except ImportError:
    from optuna.storages import (
        JournalFileStorage as _JournalFileBackend,  # type: ignore[attr-defined,assignment]
    )

__all__ = ["resolve_storage", "enable_sqlite_wal", "register_temp_storage_cleanup"]

logger = logging.getLogger(__name__)


def register_temp_storage_cleanup(path: str) -> None:
    """Schedule deletion of an auto-created temporary storage file at exit.

    The file is removed when the interpreter shuts down via :mod:`atexit`.
    Missing files are ignored so the handler is safe to register even if the
    file was already cleaned up.

    Parameters
    ----------
    path : str
        Filesystem path of the temporary storage file to remove at exit.
    """

    def _cleanup() -> None:
        try:
            os.remove(path)
            logger.info(f"Removed temporary journal storage: {path}")
        except FileNotFoundError:
            pass
        except OSError as exc:  # pragma: no cover - best-effort cleanup
            logger.warning(f"Could not remove temporary journal storage '{path}': {exc}")

    atexit.register(_cleanup)


@overload
def resolve_storage(
    storage: str | None,
    n_workers: int,
    *,
    return_path: Literal[False] = ...,
) -> optuna.storages.BaseStorage | None: ...


@overload
def resolve_storage(
    storage: str | None,
    n_workers: int,
    *,
    return_path: Literal[True],
) -> tuple[optuna.storages.BaseStorage | None, str | None]: ...


def resolve_storage(
    storage: str | None,
    n_workers: int,
    *,
    return_path: bool = False,
) -> optuna.storages.BaseStorage | None | tuple[optuna.storages.BaseStorage | None, str | None]:
    """Resolve a user-provided storage specifier into an Optuna storage object.

    For multi-worker runs the default backend is ``JournalFileStorage``
    (append-only, safe for concurrent processes).  When no storage path is
    given for a multi-worker study, a temporary journal file is created
    automatically and scheduled for deletion at interpreter exit.

    Parameters
    ----------
    storage : str or None
        Storage specifier.  Accepted forms:

        - ``None`` : in-memory for single-worker, temp journal for multi-worker.
        - Path ending in ``".db"`` or prefixed ``"sqlite:///"`` : SQLite via
          ``RDBStorage`` with a 30 s busy timeout.
        - Any other string : treated as a journal file path.

    n_workers : int
        Number of parallel worker processes.  When ``> 1`` and *storage* is
        ``None``, a temporary journal file is created.
    return_path : bool, default=False
        When ``True``, return a ``(storage, path)`` tuple where *path* is the
        concrete filesystem path or ``sqlite:///`` URL that workers can use to
        reconnect (``None`` for in-memory storage).  This avoids reaching into
        Optuna's private ``_backend._file_path`` attribute to recover the
        auto-created temp-journal path.

    Returns
    -------
    optuna.storages.BaseStorage or None
        Configured storage object, or ``None`` for in-memory storage.  When
        *return_path* is ``True``, a ``(storage, path)`` tuple is returned
        instead.

    Notes
    -----
    If an existing SQLite file is empty (0 bytes), it is assumed to be a
    leftover from a crashed run and is removed so Optuna can recreate it.

    Auto-created temporary journals (the ``storage=None``, ``n_workers > 1``
    case) are registered for deletion via :mod:`atexit` so they do not leak.

    Examples
    --------
    >>> resolve_storage(None, n_workers=1)  # in-memory
    >>> resolve_storage("study.log", n_workers=4)  # journal file
    >>> resolve_storage("sqlite:///study.db", n_workers=1)  # SQLite
    >>> store, path = resolve_storage(None, n_workers=4, return_path=True)
    """
    storage_obj, path = _resolve_storage_with_path(storage, n_workers)
    if return_path:
        return storage_obj, path
    return storage_obj


def _resolve_storage_with_path(
    storage: str | None,
    n_workers: int,
) -> tuple[optuna.storages.BaseStorage | None, str | None]:
    """Resolve storage and report the path workers should reconnect to.

    Parameters
    ----------
    storage : str or None
        Storage specifier (see :func:`resolve_storage`).
    n_workers : int
        Number of parallel worker processes.

    Returns
    -------
    tuple[optuna.storages.BaseStorage or None, str or None]
        The configured storage object and the concrete path/URL workers can
        reconnect with (``None`` for in-memory storage).
    """
    if storage is None:
        if n_workers > 1:
            tmp = tempfile.NamedTemporaryFile(suffix=".log", prefix="resdag_hpo_", delete=False)
            tmp.close()
            register_temp_storage_cleanup(tmp.name)
            logger.info(f"Created temporary journal storage: {tmp.name}")
            return JournalStorage(_JournalFileBackend(tmp.name)), tmp.name
        return None, None

    # User-provided path — detect type
    if storage.startswith("sqlite:///") or storage.endswith(".db"):
        db_path = (
            storage.removeprefix("sqlite:///") if storage.startswith("sqlite:///") else storage
        )
        # Warn about stale empty files from crashed runs
        if os.path.exists(db_path) and os.path.getsize(db_path) == 0:
            logger.warning(
                f"Found empty database file '{db_path}' (likely from a crashed run). "
                "Removing it so Optuna can create a fresh database."
            )
            os.remove(db_path)
        url = storage if storage.startswith("sqlite:///") else f"sqlite:///{storage}"
        rdb = optuna.storages.RDBStorage(
            url=url,
            engine_kwargs={
                "connect_args": {
                    "timeout": 30,
                    "check_same_thread": False,
                },
            },
        )
        return rdb, storage

    # Treat as journal file path (e.g., "study.log")
    return JournalStorage(_JournalFileBackend(storage)), storage


def enable_sqlite_wal(storage_path: str) -> None:
    """Enable WAL journal mode on a SQLite database for better concurrency.

    Write-Ahead Logging (WAL) mode allows concurrent reads while a write is in
    progress, which is essential for multi-process HPO with SQLite storage.
    ``PRAGMA synchronous=NORMAL`` is also set for improved write throughput
    without sacrificing crash safety under WAL.

    Parameters
    ----------
    storage_path : str
        Path or ``sqlite:///`` URL pointing to the SQLite database file.

    Notes
    -----
    This function is a no-op if *storage_path* does not look like a SQLite
    path (i.e., does not end with ``".db"`` and does not start with
    ``"sqlite:///"``) .

    Examples
    --------
    >>> enable_sqlite_wal("study.db")
    >>> enable_sqlite_wal("sqlite:///study.db")
    """
    if not (storage_path.endswith(".db") or storage_path.startswith("sqlite:///")):
        return

    db_path = (
        storage_path.removeprefix("sqlite:///")
        if storage_path.startswith("sqlite:///")
        else storage_path
    )
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.commit()
    conn.close()
