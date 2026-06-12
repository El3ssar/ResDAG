"""Random orthogonal recurrent matrix.

Orthogonal reservoirs have all singular values equal to 1, giving
norm-preserving linear dynamics: long memory without the eigenvalue spread
of random matrices. A useful structured alternative to sparse random
topologies, especially for memory-capacity tasks.
"""

import torch

from resdag.init.topology import register_matrix_topology


@register_matrix_topology("orthogonal")
def orthogonal_matrix(n: int, gain: float = 1.0, seed: int | None = None) -> torch.Tensor:
    """Build a random orthogonal matrix via QR decomposition.

    Draws a standard Gaussian matrix, takes its QR decomposition, and fixes
    the signs so the result is drawn from the Haar (uniform) distribution
    over orthogonal matrices.

    Parameters
    ----------
    n : int
        Matrix size (number of reservoir units).
    gain : float, default=1.0
        Scaling factor applied to the orthogonal matrix. Note that
        ``spectral_radius`` on the layer rescales the matrix afterwards
        anyway; ``gain`` matters when no spectral radius is set.
    seed : int, optional
        Seed for the Gaussian draw. ``None`` uses the global torch RNG.

    Returns
    -------
    torch.Tensor
        Orthogonal matrix of shape ``(n, n)`` (times ``gain``).

    Examples
    --------
    >>> from resdag.layers import ESNLayer
    >>> reservoir = ESNLayer(500, feedback_size=3, topology="orthogonal")
    >>> reservoir = ESNLayer(
    ...     500, feedback_size=3, topology=("orthogonal", {"seed": 42})
    ... )
    """
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    gaussian = torch.randn(n, n, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(gaussian)
    # Sign correction: make the decomposition unique so Q is Haar-distributed.
    q = q * torch.sign(torch.diagonal(r))

    return (gain * q).to(torch.float32)
