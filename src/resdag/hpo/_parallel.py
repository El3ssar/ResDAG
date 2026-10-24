"""Small, notebook-friendly joblib execution primitives for tuning.

Sampling, budgets, study ownership and trial bookkeeping belong to the caller.
Workers receive ordinary job descriptions, so a parent can prepare reproducible
configurations before dispatch. A callable may instead open its own Optuna
study, provided it also owns that study's storage cleanup.
"""

from collections.abc import Callable, Generator, Iterable, Iterator
from contextlib import contextmanager
from typing import TypeVar, cast

_Job = TypeVar("_Job")
_Result = TypeVar("_Result")

__all__ = ["iter_jobs", "run_jobs"]


def run_jobs(
    function: Callable[[_Job], _Result],
    jobs: Iterable[_Job],
    *,
    n_jobs: int,
    inner_max_num_threads: int = 1,
) -> list[_Result]:
    """Run jobs with loky and return results in input order.

    Notebook functions and closures are serialized by joblib's cloudpickle
    support. `n_jobs` follows joblib conventions: one runs in the caller,
    negative values select a CPU-relative worker count, and zero is invalid.
    The thread limit applies to worker processes, leaving the caller's thread
    configuration unchanged. An outer `joblib.parallel_config` can configure
    options such as memory mapping and its temporary directory.

    For progress reporting or early stopping, consume `iter_jobs` in its
    context manager instead of accumulating the complete result list.
    """
    with iter_jobs(
        function,
        jobs,
        n_jobs=n_jobs,
        inner_max_num_threads=inner_max_num_threads,
        ordered=True,
    ) as results:
        return list(results)


@contextmanager
def iter_jobs(
    function: Callable[[_Job], _Result],
    jobs: Iterable[_Job],
    *,
    n_jobs: int,
    inner_max_num_threads: int = 1,
    ordered: bool = False,
) -> Iterator[Iterator[_Result]]:
    """Yield job results within an explicitly managed execution context.

    Results arrive in completion order by default, allowing a caller to process
    a fast evaluation while another is still running. `ordered=True` yields
    in input order. Exiting the context closes the result generator and lets
    joblib cancel outstanding work, including after an error or interrupt.
    Joblib may warn about completed results that were intentionally unused.
    Successfully completed pools follow joblib's standard reuse policy.

    `batch_size=1` and `pre_dispatch='n_jobs'` keep the initial queue small.
    Completion callbacks can continue drawing from the input independently of
    result consumption: pre-dispatch is not a trial or time budget. For strict
    budgets, pass a finite, parent-prepared window, consume it, then decide
    whether to submit another window. Closing stops outstanding work but cannot
    undo completed jobs or their external side effects.

    Examples
    --------
    >>> with iter_jobs(lambda x: x * x, range(4), n_jobs=2) as results:
    ...     for result in results:
    ...         update_progress(result)
    """
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int):
        raise TypeError("n_jobs must be an integer using joblib's worker-count conventions")
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be zero")
    if isinstance(inner_max_num_threads, bool) or not isinstance(inner_max_num_threads, int):
        raise TypeError("inner_max_num_threads must be a positive integer")
    if inner_max_num_threads < 1:
        raise ValueError("inner_max_num_threads must be a positive integer")
    if not callable(function):
        raise TypeError("function must be callable")

    try:
        from joblib import Parallel, delayed, parallel_config
    except ImportError as error:
        raise ImportError(
            "Parallel tuning requires joblib. Install it with: pip install 'resdag[hpo]'"
        ) from error

    with parallel_config(backend="loky", inner_max_num_threads=inner_max_num_threads):
        with Parallel(
            n_jobs=n_jobs,
            return_as="generator" if ordered else "generator_unordered",
            batch_size=1,
            pre_dispatch="n_jobs",
        ) as parallel:
            results = cast(
                Generator[_Result, None, None],
                parallel(delayed(function)(job) for job in jobs),
            )
            try:
                yield results
            finally:
                results.close()
