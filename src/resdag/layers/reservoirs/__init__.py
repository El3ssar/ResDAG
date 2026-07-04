from .base_reservoir import BaseReservoirLayer
from .es2n import ES2NLayer
from .esn import ESNLayer
from .gated import GatedReservoir
from .ngrc import NGReservoir
from .resesn import ResESNLayer

__all__ = [
    "ESNLayer",
    "BaseReservoirLayer",
    "NGReservoir",
    "ES2NLayer",
    "ResESNLayer",
    "GatedReservoir",
]


def __dir__() -> list[str]:
    """Restrict ``dir()`` / tab-completion to the public API (:pep:`562`)."""
    return list(__all__)
