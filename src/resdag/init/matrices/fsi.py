"""Fast Spectral Initialization (FSI) recurrent matrix.

Builds a recurrent reservoir matrix whose spectral radius is targeted
*approximately* at sampling time, without spectral estimation. The direct
builder costs O(n²). Disable outer scaling explicitly to retain this advantage
when configuring a layer; the legacy layer default still measures and rescales.

The construction follows Gallicchio (2020), *Fast Spectral Radius
Initialization for Recurrent Neural Networks* (in *Recent Advances in Big Data
and Deep Learning*, INNSBDDL 2019, pp. 380–390): draw the entries i.i.d. from a
symmetric uniform distribution $\\mathcal{U}(-a, a)$ and choose the
half-width $a$ so that the matrix's spectral radius matches the target.

For an $N \\times N$ matrix with i.i.d. zero-mean entries of variance
$\\sigma^2$, Girko's circular law places the eigenvalues asymptotically
uniformly in a disk of radius $\\sqrt{N}\\,\\sigma$, so the spectral
radius concentrates at $\\rho \\approx \\sqrt{N}\\,\\sigma$.  For
$\\mathcal{U}(-a, a)$ the variance is $\\sigma^2 = a^2 / 3$, hence

$$
\\rho \\approx \\sqrt{N}\\,\\frac{a}{\\sqrt{3}}
\\quad\\Longrightarrow\\quad
a = \\rho \\, \\sqrt{\\frac{3}{N}}.
$$

See Also
--------
resdag.init.matrices.orthogonal : Haar-random orthogonal recurrent matrix.
resdag.init.topology.scale_to_spectral_radius : Iterative rescaling (used by
    other topologies that do not know their spectral radius a priori).
"""

import math

import torch

from resdag.init.topology import register_matrix_topology

from ._common import finite_scalar, matrix_device, seeded_generator


@register_matrix_topology("fast_spectral_initialization", spectral_radius=0.9)
def fast_spectral_initialization(
    n: int,
    spectral_radius: float = 0.9,
    seed: int | None = None,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build a recurrent matrix at a target spectral radius without eigvals.

    Implements Gallicchio's Fast Spectral Initialization (FSI): entries are
    drawn i.i.d. from $\\mathcal{U}(-a, a)$ with
    $a = \\rho \\sqrt{3 / n}$, so by the circular law the matrix's
    spectral radius is $\\approx \\rho$ *by construction*.  No
    eigendecomposition or power iteration is performed; the whole cost is the
    random draw, which is what makes it attractive for large `n`.

    The finite-size radius is approximate. The legacy registration retains
    the layer's outer spectral estimation and rescaling. To skip that cost,
    configure `get_recurrent_initializer("fast_spectral_initialization", spectral_radius=0.9, rescale=False)` and pass the resulting object to the
    layer. That explicitly accepts the finite-size approximation; the layer's
    `spectral_radius` is then ignored without a warning.

    Parameters
    ----------
    n : int
        Matrix size (number of reservoir units).
    spectral_radius : float, default=0.9
        Target spectral radius $\\rho$.  For finite `n` the realized
        radius fluctuates around this value (the circular law is asymptotic);
        the relative error shrinks as `n` grows.
    seed : int, optional
        Seed for the uniform draw.  `None` uses the global torch RNG, so the
        result is reproducible under `torch.manual_seed`.
    dtype : torch.dtype, default=torch.float32
        Real floating-point output and random-draw dtype.
    device : torch.device or str, optional
        Allocation and random-draw device (CPU by default).

    Returns
    -------
    torch.Tensor
        Dense `(n, n)` recurrent matrix on the requested dtype/device, with
        spectral radius approximately equal to `spectral_radius`.

    Examples
    --------
    >>> from resdag.init import get_recurrent_initializer
    >>> from resdag.layers import ESNLayer
    >>> reservoir = ESNLayer(
    ...     2000, feedback_size=3,
    ...     recurrent_initializer=get_recurrent_initializer(
    ...         "fast_spectral_initialization", spectral_radius=0.95, rescale=False
    ...     ),
    ... )

    References
    ----------
    C. Gallicchio, "Fast Spectral Radius Initialization for Recurrent Neural
    Networks", in *Recent Advances in Big Data and Deep Learning* (INNSBDDL
    2019), Springer, 2020, pp. 380–390.
    """
    target = matrix_device(n, dtype, device)
    spectral_radius = finite_scalar(spectral_radius, "spectral_radius")
    if spectral_radius < 0:
        raise ValueError(f"spectral_radius must be non-negative, got {spectral_radius}")

    generator = seeded_generator(seed, target)

    # Half-width of the uniform distribution that yields the target spectral
    # radius under the circular law (variance a^2/3, radius ~ sqrt(n) * a/sqrt(3)).
    half_width = spectral_radius * math.sqrt(3.0 / n)

    matrix = torch.empty(n, n, dtype=dtype, device=target)
    matrix.uniform_(-half_width, half_width, generator=generator)

    return matrix
