"""Fast Spectral Initialization (FSI) recurrent matrix.

Builds a recurrent reservoir matrix whose spectral radius is set *analytically*
at sampling time — no eigendecomposition, no power iteration, no rescaling pass.
The cost is therefore dominated by drawing the random entries, which makes it
the recommended topology for very large reservoirs where the ``O(n^3)`` dense
``eigvals`` (or even an iterative estimate + rescale) is the bottleneck.

The construction follows Gallicchio (2020), *Fast Spectral Radius
Initialization for Recurrent Neural Networks* (in *Recent Advances in Big Data
and Deep Learning*, INNSBDDL 2019, pp. 380–390): draw the entries i.i.d. from a
symmetric uniform distribution :math:`\\mathcal{U}(-a, a)` and choose the
half-width :math:`a` so that the matrix's spectral radius matches the target.

For an :math:`N \\times N` matrix with i.i.d. zero-mean entries of variance
:math:`\\sigma^2`, Girko's circular law places the eigenvalues asymptotically
uniformly in a disk of radius :math:`\\sqrt{N}\\,\\sigma`, so the spectral
radius concentrates at :math:`\\rho \\approx \\sqrt{N}\\,\\sigma`.  For
:math:`\\mathcal{U}(-a, a)` the variance is :math:`\\sigma^2 = a^2 / 3`, hence

.. math::

    \\rho \\approx \\sqrt{N}\\,\\frac{a}{\\sqrt{3}}
    \\quad\\Longrightarrow\\quad
    a = \\rho \\, \\sqrt{\\frac{3}{N}}.

See Also
--------
resdag.init.matrices.orthogonal : Haar-random orthogonal recurrent matrix.
resdag.init.topology.scale_to_spectral_radius : Iterative rescaling (used by
    other topologies that do not know their spectral radius a priori).
"""

import math

import torch

from resdag.init.topology import register_matrix_topology


@register_matrix_topology("fast_spectral_initialization", spectral_radius=0.9)
def fast_spectral_initialization(
    n: int,
    spectral_radius: float = 0.9,
    seed: int | None = None,
) -> torch.Tensor:
    """Build a recurrent matrix at a target spectral radius without eigvals.

    Implements Gallicchio's Fast Spectral Initialization (FSI): entries are
    drawn i.i.d. from :math:`\\mathcal{U}(-a, a)` with
    :math:`a = \\rho \\sqrt{3 / n}`, so by the circular law the matrix's
    spectral radius is :math:`\\approx \\rho` *by construction*.  No
    eigendecomposition or power iteration is performed; the whole cost is the
    random draw, which is what makes it attractive for large ``n``.

    Because the spectral radius is fixed analytically here, the usual
    layer-level rescaling is unnecessary.  Passing this topology to an
    ``ESNLayer`` with a ``spectral_radius`` argument still works — the realized
    radius is already on target, so the subsequent rescale is a near no-op — but
    the target is best controlled through this builder's own ``spectral_radius``
    parameter (the registry default, ``0.9``).

    Parameters
    ----------
    n : int
        Matrix size (number of reservoir units).
    spectral_radius : float, default=0.9
        Target spectral radius :math:`\\rho`.  For finite ``n`` the realized
        radius fluctuates around this value (the circular law is asymptotic);
        the relative error shrinks as ``n`` grows.
    seed : int, optional
        Seed for the uniform draw.  ``None`` uses the global torch RNG, so the
        result is reproducible under :func:`torch.manual_seed`.

    Returns
    -------
    torch.Tensor
        Dense ``(n, n)`` ``float32`` recurrent matrix with spectral radius
        ``~= spectral_radius``.

    Examples
    --------
    >>> from resdag.layers import ESNLayer
    >>> reservoir = ESNLayer(
    ...     2000, feedback_size=3, topology="fast_spectral_initialization"
    ... )
    >>> reservoir = ESNLayer(
    ...     2000,
    ...     feedback_size=3,
    ...     topology=("fast_spectral_initialization", {"spectral_radius": 0.95}),
    ... )

    References
    ----------
    C. Gallicchio, "Fast Spectral Radius Initialization for Recurrent Neural
    Networks", in *Recent Advances in Big Data and Deep Learning* (INNSBDDL
    2019), Springer, 2020, pp. 380–390.
    """
    if n <= 0:
        raise ValueError(f"Matrix size n must be positive, got {n}")
    if spectral_radius < 0:
        raise ValueError(f"spectral_radius must be non-negative, got {spectral_radius}")

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    # Half-width of the uniform distribution that yields the target spectral
    # radius under the circular law (variance a^2/3, radius ~ sqrt(n) * a/sqrt(3)).
    half_width = spectral_radius * math.sqrt(3.0 / n)

    matrix = torch.empty(n, n, dtype=torch.float32)
    matrix.uniform_(-half_width, half_width, generator=generator)

    return matrix
