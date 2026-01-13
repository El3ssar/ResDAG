"""General utility functions for resdag."""

import numpy as np


def create_rng(seed: int | np.random.Generator | None = None) -> np.random.Generator:
    """Create a NumPy random number generator.

    Parameters
    ----------
    seed : int, np.random.Generator, or None
        If int, used as seed for new Generator.
        If Generator, returned as-is.
        If None, creates unseeded Generator.

    Returns
    -------
    np.random.Generator
        A NumPy random number generator.
    """
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)
