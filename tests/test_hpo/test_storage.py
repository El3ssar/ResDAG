"""Tests for HPO storage resolution and temp-file lifecycle (AC3)."""

import os
import tempfile
from pathlib import Path

import pytest

optuna = pytest.importorskip("optuna")

from resdag.hpo.storage import (  # noqa: E402
    register_temp_storage_cleanup,
    resolve_storage,
)


class TestResolveStoragePath:
    """``resolve_storage(..., return_path=True)`` exposes the reconnect path."""

    def test_in_memory_single_worker(self):
        """Single-worker None storage is in-memory and has no path."""
        store, path = resolve_storage(None, n_workers=1, return_path=True)
        assert store is None
        assert path is None

    def test_default_return_is_storage_only(self):
        """Without return_path the call still returns a bare storage object."""
        store = resolve_storage(None, n_workers=1)
        assert store is None  # in-memory

    def test_journal_path_roundtrip(self):
        """A journal path is returned verbatim alongside the storage object."""
        with tempfile.TemporaryDirectory() as tmp:
            journal = str(Path(tmp) / "study.log")
            store, path = resolve_storage(journal, n_workers=4, return_path=True)
            assert path == journal
            assert isinstance(store, optuna.storages.JournalStorage)

    def test_sqlite_path_roundtrip(self):
        """A SQLite specifier is returned verbatim alongside an RDBStorage."""
        with tempfile.TemporaryDirectory() as tmp:
            spec = f"sqlite:///{Path(tmp) / 'study.db'}"
            store, path = resolve_storage(spec, n_workers=1, return_path=True)
            assert path == spec
            assert isinstance(store, optuna.storages.RDBStorage)

    def test_temp_journal_path_is_usable_without_private_attrs(self):
        """The auto-created temp journal path is returned (no _backend access)."""
        store, path = resolve_storage(None, n_workers=4, return_path=True)
        try:
            assert isinstance(store, optuna.storages.JournalStorage)
            assert isinstance(path, str)
            assert os.path.exists(path)
            # Path is recovered from the public return value, not _backend._file_path
            assert path.endswith(".log")
        finally:
            if path and os.path.exists(path):
                os.remove(path)


class TestTempStorageCleanup:
    """Auto-created temp journals are scheduled for cleanup (no leak)."""

    def test_cleanup_removes_file(self):
        """The registered atexit handler deletes the temp file when invoked."""
        import atexit
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            target = str(Path(tmp) / "leaky.log")
            Path(target).touch()
            assert os.path.exists(target)

            captured = {}

            def _capture(fn):
                captured["fn"] = fn
                return fn

            with mock.patch.object(atexit, "register", side_effect=_capture):
                register_temp_storage_cleanup(target)

            assert "fn" in captured, "cleanup was not registered with atexit"
            captured["fn"]()  # simulate interpreter exit
            assert not os.path.exists(target), "temp storage file was not removed"

    def test_cleanup_is_idempotent_on_missing_file(self):
        """Cleanup of an already-removed file does not raise."""
        import atexit
        from unittest import mock

        captured = {}
        with mock.patch.object(
            atexit, "register", side_effect=lambda fn: captured.setdefault("fn", fn)
        ):
            register_temp_storage_cleanup("/nonexistent/path/does-not-exist.log")
        # Running the handler on a missing file must be a no-op, not an error.
        captured["fn"]()

    def test_resolve_storage_registers_cleanup_for_temp_journal(self):
        """resolve_storage registers an atexit cleanup for the auto temp journal."""
        import atexit
        from unittest import mock

        registered = []
        with mock.patch.object(atexit, "register", side_effect=lambda fn: registered.append(fn)):
            store, path = resolve_storage(None, n_workers=4, return_path=True)
        try:
            assert registered, "no atexit cleanup was registered for the temp journal"
        finally:
            if path and os.path.exists(path):
                os.remove(path)
