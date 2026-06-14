"""Timing utilities: repeated measurement with warmup discard and summary stats."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class RunResult:
    """Outcome of timing one operation across several repeats."""

    times: list[float] = field(default_factory=list)  # seconds, timed repeats only
    rmse: float | None = None  # optional accuracy sanity metric
    info: dict = field(default_factory=dict)

    @property
    def median(self) -> float:
        return statistics.median(self.times) if self.times else float("nan")

    @property
    def mean(self) -> float:
        return statistics.fmean(self.times) if self.times else float("nan")

    @property
    def std(self) -> float:
        return statistics.pstdev(self.times) if len(self.times) > 1 else 0.0

    @property
    def best(self) -> float:
        return min(self.times) if self.times else float("nan")

    def as_dict(self) -> dict:
        return {
            "times": self.times,
            "median": self.median,
            "mean": self.mean,
            "std": self.std,
            "best": self.best,
            "rmse": self.rmse,
            "info": self.info,
        }


def repeat_timed(
    setup: Callable[[], object],
    op: Callable[[object], object],
    repeats: int,
    warmups: int = 1,
    sync: Callable[[], None] | None = None,
) -> tuple[list[float], object]:
    """Time ``op`` ``repeats`` times.

    ``setup`` runs *before each* timed call and is not measured (e.g. rebuild a
    fresh model). ``warmups`` leading iterations are run and discarded (JIT /
    cache / GPU warmup). ``sync`` (e.g. ``torch.cuda.synchronize``) is called
    just before stopping the clock so asynchronous work is fully accounted for.

    Returns ``(times, last_result)`` where ``last_result`` is the return value
    of the final ``op`` call (handy for a correctness check).
    """
    times: list[float] = []
    last = None
    for i in range(warmups + repeats):
        ctx = setup()
        if sync is not None:
            sync()
        t0 = time.perf_counter()
        last = op(ctx)
        if sync is not None:
            sync()
        elapsed = time.perf_counter() - t0
        if i >= warmups:
            times.append(elapsed)
    return times, last
