"""rc_bench — a small, extensible harness for benchmarking reservoir-computing
libraries against each other on identical ESN architectures and data.

The design goal is *re-runnability and extensibility*: every library is wrapped
in an :class:`~rc_bench.adapters.base.Adapter` exposing a uniform
``time_train`` / ``time_forecast`` interface, so adding a new library to the
comparison is a single new adapter file. The runner sweeps a configuration
matrix, the reporter turns the resulting JSON into speedup tables.

See ``benchmarks/README.md`` for usage.
"""

from .config import CONTEXTS, HParams, Point

__all__ = ["HParams", "Point", "CONTEXTS"]
