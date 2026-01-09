"""General utility functions for torch_rc."""

from typing import Optional, Union

import numpy as np


def create_rng(seed: Optional[Union[int, np.random.Generator]] = None) -> np.random.Generator:
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
