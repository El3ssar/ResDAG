"""Real database ownership checks, including failures before trial execution."""

import gc
import importlib
from contextlib import closing

import pytest

optuna = pytest.importorskip("optuna")
sqlalchemy = pytest.importorskip("sqlalchemy")

from resdag.hpo.storage import _close_storage, enable_sqlite_wal  # noqa: E402


def _owned_database(tmp_path):
    storage = optuna.storages.RDBStorage(f"sqlite:///{tmp_path / 'study.db'}")
    closed = []
    sqlalchemy.event.listen(storage.engine, "close", lambda conn, record: closed.append(conn))
    return storage, closed


def test_close_returns_checked_out_session_then_closes_pool_and_can_reconnect(tmp_path):
    storage, closed = _owned_database(tmp_path)
    study = optuna.create_study(storage=storage)
    study.enqueue_trial({"x": 2})
    study.optimize(lambda trial: trial.suggest_float("x", 0, 3) ** 2, n_trials=1)
    storage.scoped_session().connection()
    old_pool = storage.engine.pool
    assert old_pool.checkedout() >= 1

    _close_storage(storage)

    assert old_pool.checkedout() == 0
    assert closed
    assert storage.engine.pool is not old_pool
    # Disposal releases resources, rather than making the user's study unusable.
    assert study.best_value == 4
    _close_storage(storage)
    _close_storage(storage)


@pytest.mark.parametrize("failure", [ValueError("bad configuration"), KeyboardInterrupt()])
def test_run_hpo_releases_owned_storage_when_setup_fails(tmp_path, monkeypatch, failure):
    run = importlib.import_module("resdag.hpo.run")
    storage, closed = _owned_database(tmp_path)
    monkeypatch.setattr(run, "resolve_storage", lambda *args, **kwargs: (storage, "study.db"))

    def fail_loss(*args):
        raise failure

    monkeypatch.setattr(run, "get_loss", fail_loss)
    try:
        with pytest.raises(type(failure)) as caught:
            run.run_hpo(lambda: None, lambda trial: {}, lambda trial: {}, n_trials=1)
        assert caught.value is failure
        assert closed
        assert storage.engine.pool.checkedout() == 0
    finally:
        _close_storage(storage)


@pytest.mark.parametrize("trial_fails", [False, True])
def test_run_hpo_releases_pool_after_success_or_trial_failure(tmp_path, monkeypatch, trial_fails):
    run = importlib.import_module("resdag.hpo.run")
    storage, closed = _owned_database(tmp_path)
    monkeypatch.setattr(run, "resolve_storage", lambda *args, **kwargs: (storage, "study.db"))

    def objective(trial):
        if trial_fails:
            raise ArithmeticError("trial failed")
        return 3.0

    monkeypatch.setattr(
        run,
        "run_trials",
        lambda study, runner, **kwargs: study.optimize(objective, n_trials=kwargs["n_trials"]),
    )
    try:
        if trial_fails:
            with pytest.raises(ArithmeticError, match="trial failed"):
                run.run_hpo(lambda: None, lambda trial: {}, lambda trial: {}, n_trials=1)
        else:
            study = run.run_hpo(lambda: None, lambda trial: {}, lambda trial: {}, n_trials=1)
            assert closed  # inspect before reading the study opens another connection
            assert study.best_value == 3
        assert closed
    finally:
        _close_storage(storage)
        gc.collect()


def test_disposal_is_attempted_even_when_session_removal_raises(tmp_path, monkeypatch):
    storage, closed = _owned_database(tmp_path)
    original = storage.remove_session

    def broken_session():
        original()
        raise RuntimeError("session cleanup failed")

    monkeypatch.setattr(storage, "remove_session", broken_session)
    try:
        with pytest.raises(RuntimeError, match="session cleanup failed"):
            _close_storage(storage)
        assert closed
    finally:
        original()
        storage.engine.dispose()


@pytest.mark.parametrize("backend", [None, optuna.storages.InMemoryStorage()])
def test_close_accepts_backends_without_rdb_lifecycle(backend):
    _close_storage(backend)


@pytest.mark.parametrize("fail_at", [None, 0, 1, 2])
def test_wal_connection_closes_even_if_setup_or_commit_fails(monkeypatch, fail_at):
    storage = importlib.import_module("resdag.hpo.storage")
    calls = []

    class Connection:
        def execute(self, sql):
            index = len(calls)
            calls.append(sql)
            if index == fail_at:
                raise RuntimeError("setup failed")

        def commit(self):
            calls.append("commit")
            if fail_at == 2:
                raise RuntimeError("setup failed")

        def close(self):
            calls.append("close")

    monkeypatch.setattr(storage.sqlite3, "connect", lambda *args, **kwargs: Connection())
    if fail_at is None:
        enable_sqlite_wal("study.db")
    else:
        with pytest.raises(RuntimeError, match="setup failed"):
            enable_sqlite_wal("study.db")
    assert calls[-1] == "close"


def test_wal_keeps_real_database_readable_after_setup(tmp_path):
    import sqlite3

    path = str(tmp_path / "wal.db")
    enable_sqlite_wal(path)
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


@pytest.mark.parametrize("destination_direction", ["minimize", "maximize"])
def test_path_transfer_releases_owned_pool_on_success_or_direction_error(
    tmp_path, monkeypatch, destination_direction
):
    transfer = importlib.import_module("resdag.hpo.transfer")
    storage, closed = _owned_database(tmp_path)
    source = optuna.create_study(storage=storage, direction="minimize")
    source.add_trial(
        optuna.trial.create_trial(
            value=2,
            params={"x": 0.5},
            distributions={"x": optuna.distributions.FloatDistribution(0, 1)},
        )
    )
    destination = optuna.create_study(direction=destination_direction)
    source.set_user_attr("resdag_tuning_protocol_v1", "same-task")
    destination.set_user_attr("resdag_tuning_protocol_v1", "same-task")
    monkeypatch.setattr(transfer, "resolve_storage", lambda *args, **kwargs: storage)
    try:
        if destination_direction == "minimize":
            assert (
                transfer.transfer_trials(
                    destination, "study.db", mode="scores", protocol_hash="same-task"
                )
                == 1
            )
            assert destination.best_value == 2
        else:
            with pytest.raises(ValueError, match="matching objective directions"):
                transfer.transfer_trials(
                    destination, "study.db", mode="scores", protocol_hash="same-task"
                )
        assert closed
        assert source.best_value == 2  # a closed engine can reconnect
    finally:
        _close_storage(storage)


@pytest.mark.parametrize("study_count", [0, 2])
def test_ambiguous_path_transfer_releases_its_pool(tmp_path, monkeypatch, study_count):
    transfer = importlib.import_module("resdag.hpo.transfer")
    storage, closed = _owned_database(tmp_path)
    for index in range(study_count):
        optuna.create_study(storage=storage, study_name=f"source-{index}")
    destination = optuna.create_study()
    monkeypatch.setattr(transfer, "resolve_storage", lambda *args, **kwargs: storage)
    try:
        with pytest.raises(ValueError, match="pass an explicit optuna.Study"):
            transfer.transfer_trials(
                destination, "study.db", mode="scores", protocol_hash="same-task"
            )
        assert closed
        assert destination.trials == []
    finally:
        _close_storage(storage)


def test_caller_owned_transfer_study_keeps_its_pool(tmp_path):
    from resdag.hpo.transfer import transfer_trials

    storage, closed = _owned_database(tmp_path)
    source = optuna.create_study(storage=storage)
    destination = optuna.create_study()
    try:
        assert transfer_trials(destination, source) == 0
        assert closed == []
        source.set_user_attr("still_usable", True)
        assert source.user_attrs == {"still_usable": True}
    finally:
        _close_storage(storage)


@pytest.mark.parametrize("operation", ["run_setup", "path_transfer"])
def test_cleanup_failure_preserves_primary_error_with_a_note(tmp_path, monkeypatch, operation):
    storage, closed = _owned_database(tmp_path)
    original_remove = storage.remove_session

    def broken_remove():
        original_remove()
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(storage, "remove_session", broken_remove)
    try:
        if operation == "run_setup":
            run = importlib.import_module("resdag.hpo.run")
            primary = ValueError("primary configuration failed")
            monkeypatch.setattr(run, "resolve_storage", lambda *args, **kwargs: (storage, "db"))

            def fail_loss(*args):
                raise primary

            monkeypatch.setattr(run, "get_loss", fail_loss)
            with pytest.raises(ValueError, match="primary configuration failed") as caught:
                run.run_hpo(lambda: None, lambda trial: {}, lambda trial: {}, n_trials=1)
            assert caught.value is primary
        else:
            transfer = importlib.import_module("resdag.hpo.transfer")
            source = optuna.create_study(storage=storage, direction="minimize")
            source.set_user_attr("resdag_tuning_protocol_v1", "same-task")
            destination = optuna.create_study(direction="maximize")
            destination.set_user_attr("resdag_tuning_protocol_v1", "same-task")
            monkeypatch.setattr(transfer, "resolve_storage", lambda *args, **kwargs: storage)
            with pytest.raises(ValueError, match="matching objective directions") as caught:
                transfer.transfer_trials(
                    destination, "db", mode="scores", protocol_hash="same-task"
                )
        assert any("RuntimeError: cleanup failed" in note for note in caught.value.__notes__)
        assert closed
    finally:
        original_remove()
        storage.engine.dispose()


@pytest.mark.parametrize(
    "url",
    [
        "sqlite:///:memory:",
        "sqlite:///",
        "sqlite:///file::memory:?cache=shared&uri=true",
        "sqlite:///file:resdag-memory-lifecycle?mode=memory&cache=shared&uri=true",
    ],
)
def test_returned_sqlite_memory_study_keeps_its_database(url, monkeypatch):
    run = importlib.import_module("resdag.hpo.run")
    storage = optuna.storages.RDBStorage(url)
    closed = []
    sqlalchemy.event.listen(storage.engine, "close", lambda conn, record: closed.append(conn))
    monkeypatch.setattr(run, "resolve_storage", lambda *args, **kwargs: (storage, url))
    monkeypatch.setattr(
        run,
        "run_trials",
        lambda study, runner, **kwargs: study.optimize(
            lambda trial: 2.0, n_trials=kwargs["n_trials"]
        ),
    )
    pool = storage.engine.pool
    try:
        study = run.run_hpo(lambda: None, lambda trial: {}, lambda trial: {}, n_trials=1)
        assert storage.engine.pool is pool
        assert closed == []
        assert study.best_value == 2.0
        assert len(study.trials) == 1
        study.set_user_attr("still_usable", True)
        assert study.user_attrs["still_usable"] is True
    finally:
        _close_storage(storage)
    assert closed  # Explicit teardown can release the backing memory database.


def test_sqlite_memory_pool_is_disposed_after_failed_run(monkeypatch):
    run = importlib.import_module("resdag.hpo.run")
    storage = optuna.storages.RDBStorage("sqlite:///:memory:")
    closed = []
    sqlalchemy.event.listen(storage.engine, "close", lambda conn, record: closed.append(conn))
    monkeypatch.setattr(run, "resolve_storage", lambda *args, **kwargs: (storage, ":memory:"))

    def fail_loss(*args):
        raise ValueError("invalid loss")

    monkeypatch.setattr(run, "get_loss", fail_loss)
    with pytest.raises(ValueError, match="invalid loss"):
        run.run_hpo(lambda: None, lambda trial: {}, lambda trial: {}, n_trials=1)
    assert closed


@pytest.mark.parametrize("uri_query", ["", "&uri=false"])
def test_file_uri_without_enabled_uri_mode_releases_persistent_pool(
    tmp_path, monkeypatch, uri_query
):
    run = importlib.import_module("resdag.hpo.run")
    monkeypatch.chdir(tmp_path)
    url = f"sqlite:///file:ordinary-file?mode=memory{uri_query}"
    storage = optuna.storages.RDBStorage(url)
    closed = []
    sqlalchemy.event.listen(storage.engine, "close", lambda conn, record: closed.append(conn))
    monkeypatch.setattr(run, "resolve_storage", lambda *args, **kwargs: (storage, url))
    monkeypatch.setattr(
        run,
        "run_trials",
        lambda study, runner, **kwargs: study.optimize(
            lambda trial: 2.0, n_trials=kwargs["n_trials"]
        ),
    )
    try:
        study = run.run_hpo(lambda: None, lambda trial: {}, lambda trial: {}, n_trials=1)
        assert closed
        assert (tmp_path / "file:ordinary-file").is_file()
        assert study.best_value == 2.0  # Persistent data survives reconnection.
    finally:
        _close_storage(storage)


def test_sqlite_memory_cleanup_failure_raises_and_disposes_pool(monkeypatch):
    run = importlib.import_module("resdag.hpo.run")
    storage = optuna.storages.RDBStorage("sqlite:///:memory:")
    closed = []
    sqlalchemy.event.listen(storage.engine, "close", lambda conn, record: closed.append(conn))
    monkeypatch.setattr(run, "resolve_storage", lambda *args, **kwargs: (storage, ":memory:"))
    monkeypatch.setattr(
        run,
        "run_trials",
        lambda study, runner, **kwargs: study.optimize(
            lambda trial: 2.0, n_trials=kwargs["n_trials"]
        ),
    )
    original_remove = storage.remove_session

    def broken_remove():
        original_remove()
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(storage, "remove_session", broken_remove)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        run.run_hpo(lambda: None, lambda trial: {}, lambda trial: {}, n_trials=1)
    assert closed


def test_sqlalchemy_url_wal_does_not_create_literal_query_file(tmp_path):
    path = tmp_path / "query.db"
    enable_sqlite_wal(f"sqlite:///{path}?timeout=2")
    assert path.is_file()
    assert not (tmp_path / "query.db?timeout=2").exists()
