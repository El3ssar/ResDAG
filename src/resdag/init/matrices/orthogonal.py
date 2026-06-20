"""Random orthogonal recurrent matrix.

Orthogonal reservoirs have all singular values equal to 1, giving
norm-preserving linear dynamics: long memory without the eigenvalue spread
of random matrices. A useful structured alternative to sparse random
topologies, especially for memory-capacity tasks.
"""

import torch

from resdag.init.topology import register_matrix_topology


@register_matrix_topology("orthogonal", prescaled=True)
def orthogonal_matrix(n: int, gain: float = 1.0, seed: int | None = None) -> torch.Tensor:
    """Build a random orthogonal matrix via QR decomposition.

    Draws a standard Gaussian matrix, takes its QR decomposition, and fixes
    the signs so the result is drawn from the Haar (uniform) distribution
    over orthogonal matrices.

    This topology is **pre-scaled**: an orthogonal matrix already has all its
    singular values equal to ``gain`` (norm-preserving for ``gain=1``), which is
    the structural property it exists to provide. Because the layer-level
    spectral-radius rescale would collapse every singular value to the target
    radius and destroy that property, it is suppressed for this topology. A
    layer ``spectral_radius`` passed alongside this topology is therefore
    **ignored** (with a warning); control the scale through ``gain`` instead. The
    singular-values-equal-1 / norm-preservation property holds only in this
    regime (no outer rescale applied).

    Parameters
    ----------
    n : int
        Matrix size (number of reservoir units).
    gain : float, default=1.0
        Scaling factor applied to the orthogonal matrix. With ``gain=1`` all
        singular values are 1 (norm-preserving). Since the layer-level
        ``spectral_radius`` rescale is suppressed (pre-scaled topology),
        ``gain`` is the only knob on the overall scale.
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
