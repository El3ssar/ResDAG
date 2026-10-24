"""Stable opt-in seeds for independently named parts of a model."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from numbers import Integral
from typing import ClassVar

__all__ = ["SeedStreams"]


@dataclass(frozen=True)
class SeedStreams:
    """Derive repeatable integer seeds without consuming any random stream.

    Explicit names keep existing branches unchanged when another branch is
    added or construction order changes. Reusing a name intentionally gives
    the same seed; it does not share parameter tensors or recurrent state.
    Names are exact UTF-8 strings and are independent of Python's hash seed.

    The ``blake2b-v1`` algorithm is fixed: BLAKE2b with an eight-byte digest,
    personalization ``b"ResDAG.seed.v1"``, decimal master seed, a NUL separator
    and the UTF-8 name, masked to an unsigned 63-bit integer. These seeds work
    with Torch generators and NumPy's ``default_rng``. This utility does not
    enable deterministic accelerator kernels or capture stochastic execution
    state; it controls initialization seeds only.

    Examples
    --------
    >>> streams = SeedStreams(42)
    >>> streams["fast"] == streams.seed_for("fast")
    True
    >>> streams["slow"] != streams["fast"]
    True
    """

    master_seed: int
    algorithm: ClassVar[str] = "blake2b-v1"

    def __post_init__(self) -> None:
        if isinstance(self.master_seed, bool) or not isinstance(self.master_seed, Integral):
            raise TypeError("SeedStreams master_seed must be an integer")
        object.__setattr__(self, "master_seed", int(self.master_seed))

    def seed_for(self, name: str) -> int:
        """Return the stable seed for a nonblank, explicitly named component."""
        if not isinstance(name, str):
            raise TypeError("SeedStreams component name must be a string")
        if not name.strip():
            raise ValueError("SeedStreams component name must not be blank")
        payload = str(self.master_seed).encode("ascii") + b"\0" + name.encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8, person=b"ResDAG.seed.v1").digest()
        return int.from_bytes(digest, "big") & ((1 << 63) - 1)

    def __getitem__(self, name: str) -> int:
        """Use ``streams[name]`` as a short form of :meth:`seed_for`."""
        return self.seed_for(name)
