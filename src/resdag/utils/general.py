"""General utility functions for resdag."""

import numpy as np
import torch


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
