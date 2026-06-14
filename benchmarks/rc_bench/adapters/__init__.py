"""Adapter registry.

``all_adapters()`` returns one instance per benchmarkable configuration
(resdag-CPU, resdag-GPU, reservoirpy, ReservoirComputing.jl, ...). Adding a
library is a new module here plus an entry in ``_REGISTRY``.
"""

from __future__ import annotations

from .base import Adapter
from .julia_adapter import ReservoirComputingJLAdapter
from .resdag_adapter import ResdagCPUAdapter, ResdagGPUAdapter
from .reservoirpy_adapter import ReservoirPyAdapter

_REGISTRY: list[type[Adapter]] = [
    ResdagCPUAdapter,
    ResdagGPUAdapter,
    ReservoirPyAdapter,
    ReservoirComputingJLAdapter,
]


def all_adapters(only: list[str] | None = None) -> list[Adapter]:
    """Instantiate every registered adapter, optionally filtered by ``key``."""
    out: list[Adapter] = []
    for cls in _REGISTRY:
        inst = cls()
        if only and inst.key not in only:
            continue
        out.append(inst)
    return out


__all__ = ["Adapter", "all_adapters"]
