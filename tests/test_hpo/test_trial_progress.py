"""Real parent-owned pruning, process transport and failure-path checks."""

import multiprocessing as mp
import os
import queue
import threading
import time
from multiprocessing.managers import SyncManager

import optuna
import pytest

from resdag.hpo._parallel import run_jobs
from resdag.hpo._trial_progress import TrialProgress, _RemoteProgress


class _ParentPruner(optuna.pruners.BasePruner):
    def __init__(self):
        self.calls = []

    def prune(self, study, trial):
        self.calls.append((os.getpid(), trial.number, dict(trial.intermediate_values)))
        return trial.intermediate_values[max(trial.intermediate_values)] > 0.5


class _BrokenPruner(optuna.pruners.BasePruner):
    def prune(self, study, trial):
        raise LookupError("parent pruner failure")


def _evaluate(job):
    callback, values = job
    try:
        for step, value in enumerate(values, start=1):
            callback(value, step)
    except optuna.TrialPruned:
        return "pruned", os.getpid()
    return "complete", os.getpid()


@pytest.mark.parametrize("parallel", [False, True])
def test_parent_study_receives_real_reports_and_controls_pruning(parallel):
    pruner = _ParentPruner()
    study = optuna.create_study(direction="minimize", pruner=pruner)
    trials = [study.ask(), study.ask()]
    with TrialProgress(parallel=parallel) as progress:
        callbacks = [progress.callback(trial) for trial in trials]
        jobs = [(callbacks[0], [0.1, 0.2]), (callbacks[1], [1.0, 0.1])]
        results = run_jobs(_evaluate, jobs, n_jobs=2) if parallel else list(map(_evaluate, jobs))
        progress.check()
        for callback in callbacks:
            progress.release(callback)
        assert not progress._entries
    assert [result[0] for result in results] == ["complete", "pruned"]
    assert {call[0] for call in pruner.calls} == {os.getpid()}
    if parallel:
        assert all(result[1] != os.getpid() for result in results)
    assert study.trials[0].intermediate_values == {1: 0.1, 2: 0.2}
    assert study.trials[1].intermediate_values == {1: 1.0}
    # This channel never finalizes or replaces the actual caller-owned trials.
    assert all(trial.state == optuna.trial.TrialState.RUNNING for trial in study.trials)


@pytest.mark.parametrize("parallel", [False, True])
def test_parent_pruner_errors_are_not_hidden_by_worker_exception_handling(parallel):
    study = optuna.create_study(pruner=_BrokenPruner())
    with pytest.raises(LookupError, match="parent pruner failure"):
        with TrialProgress(parallel=parallel) as progress:
            callback = progress.callback(study.ask())
            if parallel:
                with pytest.raises(RuntimeError, match="Parent pruning failed"):
                    run_jobs(_evaluate, [(callback, [0.1])], n_jobs=2)
            else:
                with pytest.raises(LookupError):
                    callback(0.1, 1)
            progress.check()


@pytest.mark.parametrize(
    "value,step", [(float("nan"), 1), (float("inf"), 1), (0.1, -1), (0.1, True), (0.1, 1.5)]
)
def test_invalid_reports_are_rejected_without_touching_the_study(value, step):
    study = optuna.create_study()
    with TrialProgress() as progress:
        callback = progress.callback(study.ask())
        with pytest.raises(ValueError):
            callback(value, step)
    assert study.trials[0].intermediate_values == {}


def test_callbacks_cannot_escape_their_managed_lifetime():
    study = optuna.create_study()
    progress = TrialProgress()
    with pytest.raises(RuntimeError, match="Enter"):
        progress.callback(study.ask())
    with progress:
        callback = progress.callback(study.ask())
        progress.release(callback)
        with pytest.raises(RuntimeError, match="closed"):
            callback(1.0, 1)
    with pytest.raises(RuntimeError, match="twice"):
        with progress:
            pass


@pytest.mark.parametrize(
    "field,value",
    [
        ("response_timeout", 0),
        ("response_timeout", float("inf")),
        ("shutdown_timeout", -1),
        ("parallel", 2),
    ],
)
def test_channel_options_fail_before_starting_processes(field, value):
    with pytest.raises((ValueError, TypeError)):
        with TrialProgress(**{field: value}):
            pass


def test_remote_response_timeout_and_parent_cancellation_are_bounded():
    cancelled = threading.Event()
    callback = _RemoteProgress("token", queue.Queue(), queue.Queue(), cancelled, 0.02)
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="parent pruning decision"):
        callback(0.1, 1)
    assert time.monotonic() - started < 1
    cancelled.set()
    with pytest.raises(RuntimeError, match="closed"):
        callback(0.1, 2)


def test_out_of_sequence_response_is_not_treated_as_a_decision():
    replies = queue.Queue()
    replies.put((99, "continue", ""))
    callback = _RemoteProgress("token", queue.Queue(), replies, threading.Event(), 1)
    with pytest.raises(RuntimeError, match="sequence"):
        callback(0.1, 1)


def _wait_for_children_to_close(before):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        remaining = {child.pid for child in mp.active_children()} - before
        if not remaining:
            return
        time.sleep(0.02)
    assert not remaining


def test_listener_start_failure_closes_its_spawn_manager(monkeypatch):
    before = {child.pid for child in mp.active_children()}

    def fail_start(self):
        raise RuntimeError("thread start failed")

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    with pytest.raises(RuntimeError, match="thread start failed"):
        with TrialProgress(parallel=True):
            pass
    _wait_for_children_to_close(before)


def test_primary_interrupt_survives_channel_shutdown():
    before = {child.pid for child in mp.active_children()}
    with pytest.raises(KeyboardInterrupt, match="stop tuning"):
        with TrialProgress(parallel=True):
            raise KeyboardInterrupt("stop tuning")
    _wait_for_children_to_close(before)


def test_failed_manager_is_reported_without_a_listener_thread_warning():
    progress = TrialProgress(parallel=True)
    with pytest.raises((EOFError, BrokenPipeError, ConnectionError)):
        with progress:
            progress._manager.shutdown()
            deadline = time.monotonic() + 2
            while progress._error is None and time.monotonic() < deadline:
                time.sleep(0.01)
            progress.check()
    assert not progress._thread.is_alive()


def test_late_released_remote_callback_does_not_cancel_other_jobs():
    study = optuna.create_study(pruner=optuna.pruners.NopPruner())
    with TrialProgress(parallel=True, response_timeout=1) as progress:
        retired = progress.callback(study.ask())
        active = progress.callback(study.ask())
        progress.release(retired)
        with pytest.raises(RuntimeError, match="closed"):
            retired(0.1, 1)
        active(0.2, 1)
        progress.check()
        assert progress._thread.is_alive()
    assert study.trials[0].intermediate_values == {}
    assert study.trials[1].intermediate_values == {1: 0.2}


def test_release_ignores_a_report_already_queued_without_resurrecting_its_entry():
    study = optuna.create_study(pruner=optuna.pruners.NopPruner())
    with TrialProgress(parallel=True, response_timeout=1) as progress:
        retired = progress.callback(study.ask())
        active = progress.callback(study.ask())
        # Block listener admission until the queued job has been retired. The
        # live job behind it proves the listener consumed and ignored it.
        with progress._lock:
            progress._requests.put((retired.token, 0, 0.1, 1))
            progress.release(retired)
        active(0.2, 1)
        progress.check()
        assert retired.token not in progress._entries
    assert study.trials[0].intermediate_values == {}
    assert study.trials[1].intermediate_values == {1: 0.2}


def test_timeout_is_terminal_and_delayed_response_cannot_trigger_another_report():
    requests, replies = queue.Queue(), queue.Queue()
    callback = _RemoteProgress("token", requests, replies, threading.Event(), 0.01)
    with pytest.raises(TimeoutError):
        callback(0.1, 1)
    replies.put((0, "continue", ""))
    with pytest.raises(RuntimeError, match="closed"):
        callback(0.2, 2)
    assert requests.qsize() == 1


def test_stored_pruner_failure_takes_precedence_over_stack_cleanup_failure():
    study = optuna.create_study(pruner=_BrokenPruner())

    def cleanup_failure():
        raise OSError("secondary cleanup failure")

    with pytest.raises(LookupError, match="parent pruner failure") as captured:
        with TrialProgress() as progress:
            callback = progress.callback(study.ask())
            with pytest.raises(LookupError):
                callback(0.1, 1)
            progress._stack.callback(cleanup_failure)
    assert any("secondary cleanup failure" in note for note in captured.value.__notes__)


def test_body_interrupt_takes_precedence_over_pruner_and_cleanup_errors():
    study = optuna.create_study(pruner=_BrokenPruner())

    def cleanup_failure():
        raise OSError("secondary cleanup failure")

    with pytest.raises(KeyboardInterrupt, match="primary interruption") as captured:
        with TrialProgress() as progress:
            callback = progress.callback(study.ask())
            with pytest.raises(LookupError):
                callback(0.1, 1)
            progress._stack.callback(cleanup_failure)
            raise KeyboardInterrupt("primary interruption")
    assert any("parent pruner failure" in note for note in captured.value.__notes__)
    assert any("secondary cleanup failure" in note for note in captured.value.__notes__)


@pytest.mark.parametrize("partial", [False, True])
def test_manager_start_interrupted_after_spawn_reaps_process_and_preserves_interrupt(
    monkeypatch, partial
):
    before = {child.pid for child in mp.active_children()}
    original = SyncManager.start

    def interrupted(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if partial:
            # Emulate interruption before BaseManager.start installed its
            # shutdown finalizer, after the manager process was started.
            del self.shutdown
        raise KeyboardInterrupt("interrupted manager start")

    monkeypatch.setattr(SyncManager, "start", interrupted)
    with pytest.raises(KeyboardInterrupt, match="interrupted manager start"):
        with TrialProgress(parallel=True):
            pass
    _wait_for_children_to_close(before)


def test_dead_listener_wakes_remote_callback_and_is_reported_by_parent():
    study = optuna.create_study()
    with pytest.raises(RuntimeError, match="listener stopped unexpectedly"):
        with TrialProgress(parallel=True, response_timeout=1) as progress:
            callback = progress.callback(study.ask())
            progress._requests.put(None)
            progress._thread.join(timeout=1)
            assert not progress._thread.is_alive()
            started = time.monotonic()
            with pytest.raises(RuntimeError, match="closed"):
                callback(0.1, 1)
            assert time.monotonic() - started < 0.5
            progress.check()


def _crash_worker(_):
    os._exit(13)


def test_worker_crash_cleans_up_manager_and_listener_without_masking_worker_error():
    from joblib.externals.loky.process_executor import TerminatedWorkerError

    progress = TrialProgress(parallel=True)
    with pytest.raises(TerminatedWorkerError):
        with progress:
            run_jobs(_crash_worker, [None], n_jobs=2)
    assert not progress._thread.is_alive()
    assert not progress._manager._process.is_alive()
