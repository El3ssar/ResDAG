"""Parent-owned Optuna pruning with standard process-safe progress queues.

Joblib owns the workers. This small channel transports only scalar progress;
the real Trial and its storage always remain in the caller's process.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import operator
import queue
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from multiprocessing.managers import SyncManager
from typing import Any, Literal

import optuna

ProgressCallback = Callable[[float, int], None]


def _validate(value: float, step: int) -> tuple[float, int]:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Pruning reports must be finite")
    try:
        index = operator.index(step)
    except TypeError:
        raise ValueError("Pruning resource steps must be nonnegative integers") from None
    if isinstance(step, bool) or index < 0:
        raise ValueError("Pruning resource steps must be nonnegative integers")
    return value, index


def _report(trial: optuna.Trial, value: float, step: int) -> None:
    value, step = _validate(value, step)
    trial.report(value, step=step)
    if trial.should_prune():
        raise optuna.TrialPruned(f"Pruned at resource step {step}")


@dataclass
class _RemoteProgress:
    token: str
    requests: Any
    replies: Any
    cancelled: Any
    timeout: float
    sequence: int = 0
    released: Any = None
    _closed: bool = False

    def __call__(self, value: float, step: int) -> None:
        value, step = _validate(value, step)
        if self._closed:
            raise RuntimeError("The parent tuning progress channel has closed")
        try:
            self._request(value, step)
        except BaseException:
            # A timed-out/pruned job cannot safely reuse its queue: a delayed
            # response may still arrive for the previous request.
            self._closed = True
            raise

    def _request(self, value: float, step: int) -> None:
        if self.cancelled.is_set() or (self.released is not None and self.released.is_set()):
            raise RuntimeError("The parent tuning progress channel has closed")
        sequence = self.sequence
        self.sequence += 1
        self.requests.put((self.token, sequence, value, step))
        deadline = time.monotonic() + self.timeout
        while True:
            if self.cancelled.is_set() or (self.released is not None and self.released.is_set()):
                raise RuntimeError("The parent tuning progress channel has closed")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for the parent pruning decision")
            try:
                received, status, detail = self.replies.get(timeout=min(0.1, remaining))
            except queue.Empty:
                continue
            if received != sequence:
                raise RuntimeError("Unexpected pruning response sequence")
            if status == "continue":
                return
            if status == "prune":
                raise optuna.TrialPruned(detail)
            raise RuntimeError(detail)


@dataclass
class _LocalProgress:
    token: str
    owner: TrialProgress
    trial: optuna.Trial

    def __call__(self, value: float, step: int) -> None:
        with self.owner._lock:
            if self.owner._stack is None or self.token not in self.owner._entries:
                raise RuntimeError("The parent tuning progress channel has closed")
        value, step = _validate(value, step)
        self.owner.check()
        try:
            _report(self.trial, value, step)
        except optuna.TrialPruned:
            raise
        except BaseException as error:
            self.owner._remember_error(error)
            raise


@dataclass
class TrialProgress:
    """Own a pruning channel for one complete tuning call.

    Register a new callback for each job/stage, release it after that job has
    finished, and call `check` after consuming a joblib window. A callback
    must be called serially by its one job. Reports and custom pruners execute
    in the parent; parallel mode uses one listener thread and a standard spawn
    manager, without launching model workers itself. Pruners must terminate:
    Python cannot forcibly interrupt an arbitrary blocked user callback.
    `response_timeout` bounds waiting for a parent decision after submitting
    a report. It is not a hard operating-system deadline for manager RPCs or
    paused processes; standard manager IPC must remain responsive. Cleanup
    bounds its listener join and the manager's standard shutdown procedure.
    """

    parallel: bool = False
    response_timeout: float = 60.0
    shutdown_timeout: float = 5.0
    _entries: dict[str, tuple[optuna.Trial, Any, int, Any]] = field(
        default_factory=dict, init=False, repr=False
    )
    _lock: Any = field(default_factory=threading.RLock, init=False, repr=False)
    _error: BaseException | None = field(default=None, init=False, repr=False)
    _stack: ExitStack | None = field(default=None, init=False, repr=False)
    _manager: Any = field(default=None, init=False, repr=False)
    _requests: Any = field(default=None, init=False, repr=False)
    _cancelled: Any = field(default=None, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _entered: bool = field(default=False, init=False, repr=False)

    def __enter__(self) -> TrialProgress:
        if self._entered:
            raise RuntimeError("A pruning channel cannot be entered twice")
        if not isinstance(self.parallel, bool):
            raise TypeError("parallel must be a boolean")
        for name, value in (
            ("response_timeout", self.response_timeout),
            ("shutdown_timeout", self.shutdown_timeout),
        ):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        self._entered = True
        stack = self._stack = ExitStack()
        try:
            if self.parallel:
                manager = SyncManager(
                    ctx=mp.get_context("spawn"), shutdown_timeout=self.shutdown_timeout
                )
                self._manager = manager
                try:
                    stack.enter_context(manager)
                except BaseException as error:
                    try:
                        self._close_failed_manager_start(manager, error)
                    except BaseException as cleanup_error:
                        error.add_note(
                            f"Pruning manager startup cleanup also failed: {cleanup_error}"
                        )
                    raise
                self._requests = manager.Queue()
                self._cancelled = manager.Event()
                self._thread = threading.Thread(
                    target=self._listen, name="resdag-trial-progress", daemon=True
                )
                stack.callback(self._stop_listener)
                self._thread.start()
            return self
        except BaseException as error:
            try:
                stack.close()
            except BaseException as cleanup_error:
                error.add_note(f"Pruning startup cleanup also failed: {cleanup_error}")
            self._stack = None
            raise

    def _close_failed_manager_start(self, manager: Any, error: BaseException) -> None:
        """Reap a manager interrupted between spawning and context registration."""
        shutdown = getattr(manager, "shutdown", None)
        if shutdown is not None:
            try:
                shutdown()
                return
            except BaseException as cleanup_error:
                error.add_note(f"Pruning manager startup cleanup also failed: {cleanup_error}")
        process = getattr(manager, "_process", None)
        if process is None or process.pid is None:
            return
        for action in (
            process.terminate,
            lambda: process.join(timeout=self.shutdown_timeout),
            lambda: process.kill() if process.is_alive() else None,
            lambda: process.join(timeout=self.shutdown_timeout),
        ):
            try:
                action()
            except BaseException as cleanup_error:
                error.add_note(f"Pruning manager startup cleanup also failed: {cleanup_error}")

    def callback(self, trial: optuna.Trial) -> ProgressCallback:
        if self._stack is None:
            raise RuntimeError("Enter the pruning channel before registering a trial")
        self.check()
        token = uuid.uuid4().hex
        if not self.parallel:
            with self._lock:
                self._entries[token] = (trial, None, -1, None)
            return _LocalProgress(token, self, trial)
        replies = self._manager.Queue()
        released = self._manager.Event()
        with self._lock:
            self._entries[token] = (trial, replies, -1, released)
        return _RemoteProgress(
            token,
            self._requests,
            replies,
            self._cancelled,
            self.response_timeout,
            released=released,
        )

    def release(self, callback: ProgressCallback) -> None:
        token = getattr(callback, "token", None)
        if not isinstance(token, str):
            return
        with self._lock:
            entry = self._entries.pop(token, None)
        if entry is not None and entry[3] is not None:
            try:
                entry[3].set()
            except BaseException as error:
                self._remember_error(error)

    def check(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise error
        thread = self._thread
        if (
            self.parallel
            and self._stack is not None
            and thread is not None
            and not thread.is_alive()
        ):
            error = RuntimeError("The parent tuning progress listener stopped unexpectedly")
            self._remember_error(error)
            raise error

    def _remember_error(self, error: BaseException) -> None:
        with self._lock:
            if self._error is None:
                self._error = error

    def _listen(self) -> None:
        try:
            while not self._cancelled.is_set():
                try:
                    request = self._requests.get(timeout=0.1)
                except queue.Empty:
                    continue
                if request is None:
                    return
                token, sequence, value, step = request
                with self._lock:
                    entry = self._entries.get(token)
                if entry is None:
                    # Release can race an already queued final report. Its
                    # per-job signal wakes that worker without harming peers.
                    continue
                trial, replies, previous_sequence, released = entry
                if sequence != previous_sequence + 1:
                    raise RuntimeError("Unexpected pruning request sequence")
                with self._lock:
                    if token not in self._entries:
                        continue
                    self._entries[token] = (trial, replies, sequence, released)
                try:
                    self.check()
                    _report(trial, value, step)
                except optuna.TrialPruned as error:
                    replies.put((sequence, "prune", str(error)))
                except BaseException as error:
                    self._remember_error(error)
                    replies.put((sequence, "error", f"Parent pruning failed: {error}"))
                else:
                    replies.put((sequence, "continue", ""))
        except BaseException as error:
            self._remember_error(error)
        finally:
            try:
                self._cancelled.set()
            except BaseException as cleanup_error:
                self._remember_error(cleanup_error)

    def _stop_listener(self) -> None:
        for action in (self._cancelled.set, lambda: self._requests.put(None)):
            try:
                action()
            except BaseException as error:
                self._remember_error(error)
        thread = self._thread
        if thread is not None and thread.ident is not None:
            thread.join(timeout=self.shutdown_timeout)
            if thread.is_alive():
                raise RuntimeError("A parent pruning callback did not stop during cleanup")

    def __exit__(self, error_type: Any, error: Any, traceback: Any) -> Literal[False]:
        stack, self._stack = self._stack, None
        cleanup_error = None
        try:
            if stack is not None:
                stack.close()
        except BaseException as caught:
            cleanup_error = caught
        finally:
            with self._lock:
                self._entries.clear()
                recorded_error = self._error
        primary = error if error is not None else recorded_error
        if primary is None:
            primary = cleanup_error
        for secondary in (recorded_error, cleanup_error):
            if secondary is not None and primary is not None and secondary is not primary:
                primary.add_note(f"Pruning channel cleanup also failed: {secondary}")
        if error is None and primary is not None:
            raise primary
        return False
