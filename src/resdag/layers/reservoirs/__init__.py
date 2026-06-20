from .base_reservoir import BaseReservoirLayer
from .esn import ESNLayer
from .ngrc import NGReservoir

__all__ = ["ESNLayer", "BaseReservoirLayer", "NGReservoir"]


def __dir__() -> list[str]:
    """Restrict ``dir()`` / tab-completion to the public API (:pep:`562`)."""
    return list(__all__)
