from .base_reservoir import BaseReservoirLayer
from .cornn import CoRNNLayer
from .es2n import ES2NLayer
from .esn import ESNLayer
from .eusn import EuSNLayer
from .gated import GatedReservoir, GatedReservoirLayer
from .legendre_delay import LegendreDelayLayer
from .lif import LIFReservoir, LIFReservoirLayer
from .ngrc import NGReservoir, NGReservoirLayer
from .resesn import ResESNLayer
from .torch_recurrent import TorchRecurrentLayer

__all__ = [
    "ESNLayer",
    "BaseReservoirLayer",
    "NGReservoirLayer",
    "NGReservoir",
    "ES2NLayer",
    "ResESNLayer",
    "GatedReservoirLayer",
    "GatedReservoir",
    "EuSNLayer",
    "LIFReservoirLayer",
    "LIFReservoir",
    "CoRNNLayer",
    "LegendreDelayLayer",
    "TorchRecurrentLayer",
]


def __dir__() -> list[str]:
    """Restrict `dir()` / tab-completion to the public API ([PEP 562](https://peps.python.org/pep-0562/))."""
    return list(__all__)
