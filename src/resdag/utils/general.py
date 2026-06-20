"""General utility functions for resdag."""

import numpy as np
import torch

# Anything accepted as a reproducibility seed throughout resdag.  A plain
# ``int`` pins a value; a ``torch.Generator`` lets callers (e.g. an HPO loop)
# carry a per-trial generator; ``None`` defers to the global RNG.
SeedLike = int | torch.Generator | None


def coerce_seed_to_int(seed: SeedLike) -> int | None:
    """Reduce a torch/int/None seed to a plain ``int`` seed (or ``None``).

    The NumPy-backed graph builders and named initializers thread an integer
    (or ``None``) seed down to :func:`create_rng`.  A :class:`torch.Generator`
    cannot be passed there directly, so this helper extracts a deterministic
    integer from its initial seed — two generators created with the same
    ``manual_seed`` therefore yield the same int, keeping the topology a pure
    function of the generator.

    Parameters
    ----------
    seed : int, torch.Generator, or None
        Seed in any of the accepted forms.

    Returns
    -------
    int or None
        ``seed`` itself if it was an ``int``; the generator's
        ``initial_seed()`` if it was a :class:`torch.Generator`; ``None`` if it
        was ``None``.
    """
    if seed is None or isinstance(seed, int):
        return seed
    if isinstance(seed, torch.Generator):
        # initial_seed() is the value the generator was seeded with; identical
        # generators (same manual_seed) therefore map to the same int.
        return int(seed.initial_seed())
    raise TypeError(
        f"Invalid seed type: {type(seed).__name__}. Expected int, torch.Generator, or None."
    )


def create_torch_generator(
    seed: SeedLike = None,
    device: torch.device | str | None = None,
) -> torch.Generator:
    """Create a torch :class:`~torch.Generator` for reproducible weight draws.

    Used for the default ``nn.init`` weight/bias draws inside reservoir cells so
    that a single ``seed`` deterministically fixes every parameter without
    touching (and without depending on the prior state of) torch's global RNG.

    Parameters
    ----------
    seed : int, torch.Generator, or None
        If ``int``, seeds a fresh generator on ``device``.
        If ``torch.Generator``, it is returned as-is (the caller's generator is
        threaded straight through, so successive draws advance one stream).
        If ``None``, the generator is seeded from torch's global RNG so that
        ``torch.manual_seed`` still propagates — matching :func:`create_rng`.

    device : torch.device, str, or None
        Device for the new generator.  Ignored when ``seed`` is already a
        :class:`torch.Generator`.

    Returns
    -------
    torch.Generator
        A generator suitable for passing to ``nn.init.*`` via ``generator=``.

    Notes
    -----
    When ``seed is None`` the integer seed is sampled from torch's global RNG
    via ``torch.randint``, advancing the global state but tying the draws to
    ``torch.manual_seed`` for end-to-end reproducibility.
    """
    if isinstance(seed, torch.Generator):
        return seed
    generator = torch.Generator(device=device if device is not None else "cpu")
    if seed is None:
        seed = int(torch.randint(0, 2**63 - 1, (1,)).item())
    generator.manual_seed(int(seed))
    return generator


def create_rng(seed: int | np.random.Generator | None = None) -> np.random.Generator:
    """Create a NumPy random number generator.

    Parameters
    ----------
    seed : int, np.random.Generator, or None
        If int, used as seed for a new Generator.
        If Generator, returned as-is.
        If None, the seed is derived from torch's global RNG so that
        ``torch.manual_seed`` propagates to the generator. This keeps
        graph-based topologies (which draw from NumPy) reproducible under
        the same torch global seed, matching the behaviour of matrix
        topologies that draw from torch directly.

    Returns
    -------
    np.random.Generator
        A NumPy random number generator.

    Notes
    -----
    When ``seed is None`` the numpy seed is sampled from torch's global RNG
    via ``torch.randint``. This advances the torch global RNG state, but ties
    every NumPy draw to ``torch.manual_seed`` for end-to-end reproducibility.
    """
    if isinstance(seed, np.random.Generator):
        return seed
    if seed is None:
        # Derive the numpy seed from torch's global RNG so that
        # torch.manual_seed(...) makes graph generators reproducible.
        seed = int(torch.randint(0, 2**63 - 1, (1,)).item())
    return np.random.default_rng(seed)
