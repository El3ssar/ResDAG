"""Exercise real loky workers, closures, cancellation and caller isolation."""

import builtins
import os
import pickle
import threading
import time
from itertools import islice

import numpy as np
import pytest

joblib = pytest.importorskip("joblib")

from resdag.hpo._parallel import iter_jobs, run_jobs  # noqa: E402


def test_notebook_closure_runs_in_child_processes_and_returns_input_order():
    offset = 13

    class LocalConfiguration:
        def __init__(self, value):
            self.value = value

    def notebook_cell(job: LocalConfiguration):
        time.sleep(0.02 * (3 - job.value))
        return job.value + offset, os.getpid()

    jobs = (LocalConfiguration(value) for value in range(4))
    results = run_jobs(notebook_cell, jobs, n_jobs=2)
    assert [value for value, pid in results] == [13, 14, 15, 16]
    assert all(pid != os.getpid() for value, pid in results)


def test_unordered_results_can_be_processed_before_an_earlier_job(tmp_path):
    release = tmp_path / "release"

    def work(index):
        if index == 0:
            deadline = time.monotonic() + 15
            while not release.exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError("parent never released earlier job")
                time.sleep(0.01)
        return index

    try:
        with iter_jobs(work, [0, 1], n_jobs=2) as results:
            assert next(results) == 1
            release.touch()
            assert list(results) == [0]
    finally:
        release.touch()


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_empty_iterable_is_supported(n_jobs):
    assert run_jobs(lambda value: value, [], n_jobs=n_jobs) == []


def test_single_worker_runs_in_caller_and_does_not_change_thread_environment(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    assert run_jobs(lambda value: (value, os.getpid()), [3], n_jobs=1) == [(3, os.getpid())]
    assert os.environ["OMP_NUM_THREADS"] == "7"


def test_worker_thread_limits_do_not_modify_parent_environment(monkeypatch):
    keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    for index, key in enumerate(keys):
        monkeypatch.setenv(key, str(index + 5))
    before = {key: os.environ[key] for key in keys}

    def worker_threads(_):
        import torch

        return {key: os.environ[key] for key in keys}, torch.get_num_threads()

    results = run_jobs(worker_threads, range(2), n_jobs=2, inner_max_num_threads=2)
    assert results == [({key: "2" for key in keys}, 2)] * 2
    assert {key: os.environ[key] for key in keys} == before


@pytest.mark.parametrize("error", [ValueError("worker evaluation failed"), KeyboardInterrupt()])
def test_worker_error_or_interrupt_propagates_and_next_call_still_works(error):
    def fail(_):
        raise error

    with pytest.raises(type(error)):
        run_jobs(fail, [0, 1], n_jobs=2)
    assert run_jobs(lambda value: value + 1, [2, 4], n_jobs=2) == [3, 5]


def test_input_iterator_failure_propagates_and_releases_execution():
    def jobs():
        yield 1
        raise ValueError("could not prepare next configuration")

    with pytest.raises(ValueError, match="could not prepare next configuration"):
        run_jobs(lambda value: value, jobs(), n_jobs=2)
    assert run_jobs(lambda value: value, [3], n_jobs=2) == [3]


def test_finite_window_bounds_consumption_and_generator_closes_on_early_exit():
    consumed = []

    def jobs():
        for value in range(100):
            consumed.append(value)
            yield value

    # A finite window, rather than pre_dispatch, enforces a strict trial budget.
    with iter_jobs(lambda value: value, islice(jobs(), 4), n_jobs=1) as results:
        assert next(results) == 0
    assert consumed == [0]
    assert list(results) == []


def test_parallel_window_never_consumes_jobs_from_the_next_budget_window():
    consumed = []

    def jobs():
        for value in range(100):
            consumed.append(value)
            yield value

    with iter_jobs(lambda value: value, islice(jobs(), 4), n_jobs=2) as results:
        assert sorted(results) == [0, 1, 2, 3]
    assert consumed == [0, 1, 2, 3]


def test_serialization_failure_propagates_and_next_call_still_works():
    lock = threading.Lock()
    with pytest.raises(pickle.PicklingError, match="Could not pickle"):
        run_jobs(lambda value: lock.locked(), [0, 1], n_jobs=2)
    assert run_jobs(lambda value: value, [3], n_jobs=2) == [3]


@pytest.mark.skipif(os.name != "posix", reason="uses POSIX pid-existence checks")
@pytest.mark.parametrize(
    "parent_error", [None, ValueError("stop after result"), KeyboardInterrupt()]
)
def test_early_exit_cancels_inflight_work_and_releases_workers(tmp_path, parent_error):
    started = tmp_path / "started"
    released = tmp_path / "released"

    def work(index):
        if index == 0:
            deadline = time.monotonic() + 15
            while not started.exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError("second job never started")
                time.sleep(0.01)
            return index
        started.write_text(str(os.getpid()))
        while not released.exists():
            time.sleep(0.01)
        return index

    try:
        with pytest.warns(UserWarning, match="cancelled"):
            if parent_error is not None:
                with pytest.raises(type(parent_error)) as caught:
                    with iter_jobs(work, [0, 1], n_jobs=2) as results:
                        assert next(results) == 0
                        raise parent_error
                assert caught.value is parent_error
            else:
                with iter_jobs(work, [0, 1], n_jobs=2) as results:
                    assert next(results) == 0
        worker_pid = int(started.read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(worker_pid, 0)
        assert list(results) == []
    finally:
        released.touch()
    assert run_jobs(lambda value: value, [5], n_jobs=2) == [5]


def test_joblib_memmapping_configuration_and_temporary_cleanup_are_preserved(tmp_path):
    array = np.ones(300_000, dtype=np.float64)
    with joblib.parallel_config(temp_folder=str(tmp_path), max_nbytes="1M"):
        results = run_jobs(
            lambda value: (isinstance(value, np.memmap), float(value.sum())),
            [array, array],
            n_jobs=2,
        )
    assert results == [(True, 300_000.0)] * 2
    assert not list(tmp_path.glob("joblib_memmapping_folder_*"))


@pytest.mark.parametrize("n_jobs", [0, True, 1.5])
def test_invalid_worker_count_fails_before_consuming_jobs(n_jobs):
    def jobs():
        pytest.fail("invalid configuration must not consume jobs")
        yield 1

    with pytest.raises((ValueError, TypeError), match="n_jobs"):
        run_jobs(lambda value: value, jobs(), n_jobs=n_jobs)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_inner_thread_limit_is_rejected(limit):
    with pytest.raises((ValueError, TypeError), match="inner_max_num_threads"):
        run_jobs(lambda value: value, [], n_jobs=1, inner_max_num_threads=limit)


def test_non_callable_worker_is_rejected():
    with pytest.raises(TypeError, match="function must be callable"):
        run_jobs(None, [], n_jobs=1)


def test_missing_joblib_has_an_actionable_optional_dependency_error(monkeypatch):
    original_import = builtins.__import__

    def without_joblib(name, *args, **kwargs):
        if name == "joblib":
            raise ModuleNotFoundError("no joblib")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_joblib)
    with pytest.raises(ImportError, match=r"pip install 'resdag\[hpo\]'"):
        run_jobs(lambda value: value, [], n_jobs=1)
