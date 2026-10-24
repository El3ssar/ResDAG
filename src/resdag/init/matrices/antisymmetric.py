r"""Antisymmetric (skew-symmetric) recurrent matrix.

An antisymmetric matrix satisfies $M = -M^\top$.  Its eigenvalues are
purely imaginary, so the linear map $x \mapsto M x$ is a pure *rotation
generator*: the associated flow $\dot x = M x$ conserves the state norm
exactly ($\frac{d}{dt}\|x\|^2 = 2\,x^\top M x = 0$ because
$x^\top M x = 0$ for skew-symmetric $M$).  A forward-Euler step of
that flow is *near*-norm-preserving for a small step size, which is precisely
the stable, non-dissipative long-memory regime the Euler State Network
(`resdag.layers.cells.eusn_cell`) is built on.

This builder draws a random square matrix $A$ and returns its
antisymmetric part $A - A^\top$.  It is registered as the `antisymmetric`
matrix topology, so it is available by name in `ESNLayer(topology=...)` for
general use; the EuSN cell applies its *own* on-the-fly antisymmetrization of a
frozen `W_r` (see `resdag.layers.cells.eusn_cell.EuSNCell`), so the two
paths are independent.

Spectral radius
---------------
Unlike `resdag.init.matrices.orthogonal`, this topology is **not**
pre-scaled: an antisymmetric matrix has a well-defined nonzero spectral radius
(the largest `|imaginary eigenvalue|`), and multiplying it by a scalar keeps
it antisymmetric, so a layer-level `spectral_radius` rescales it cleanly.
`resdag.init.topology.estimate_spectral_radius` uses
`torch.linalg.eigvals` (complex-eigenvalue aware), so it reads off the
imaginary-axis radius correctly.

See Also
--------
resdag.init.matrices.orthogonal : Haar-random orthogonal matrix (the
    *exp* of an antisymmetric generator; exactly norm-preserving).
resdag.layers.cells.eusn_cell.EuSNCell : Euler State Network, which uses an
    antisymmetric recurrent matrix minus a small diffusion term.
"""

import torch

from resdag.init.topology import register_matrix_topology

from ._common import finite_scalar, matrix_device, seeded_generator


@register_matrix_topology("antisymmetric", scaling=1.0)
def antisymmetric_matrix(
    n: int,
    scaling: float = 1.0,
    seed: int | None = None,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    r"""Build a random antisymmetric (skew-symmetric) matrix $A - A^\top$.

    Draws a standard-Gaussian matrix $A$ and returns
    $\text{scaling} \cdot (A - A^\top)$, which satisfies
    $M = -M^\top$ exactly (to floating-point).  Its eigenvalues are purely
    imaginary, so $x \mapsto M x$ is a rotation generator: the flow it
    induces conserves $\|x\|$, and a small forward-Euler step is
    near-norm-preserving — the stable long-memory regime exploited by the Euler
    State Network.

    Because the matrix is *not* pre-scaled, a layer-level `spectral_radius`
    rescales it cleanly (scaling preserves antisymmetry); control the raw scale
    through `scaling` when no `spectral_radius` is given.

    Parameters
    ----------
    n : int
        Matrix size (number of reservoir units).
    scaling : float, default=1.0
        Multiplicative gain applied to the antisymmetric part.  Only the raw
        scale — the antisymmetry is independent of it.
    seed : int, optional
        Seed for the Gaussian draw.  `None` uses the global torch RNG (so the
        result is reproducible under `torch.manual_seed`).
    dtype : torch.dtype, default=torch.float32
        Real floating-point output and random-draw dtype.
    device : torch.device or str, optional
        Allocation and random-draw device (CPU by default).

    Returns
    -------
    torch.Tensor
        Dense `(n, n)` antisymmetric matrix on the requested dtype/device.

    Raises
    ------
    ValueError
        If `n` is not a positive integer.

    Examples
    --------
    >>> import torch
    >>> from resdag.init.matrices.antisymmetric import antisymmetric_matrix
    >>> m = antisymmetric_matrix(64, seed=0)
    >>> torch.allclose(m, -m.t(), atol=1e-6)
    True

    By name in a layer (antisymmetric recurrent connectivity):

    >>> from resdag.layers import ESNLayer
    >>> reservoir = ESNLayer(
    ...     128, feedback_size=3, topology="antisymmetric", spectral_radius=0.9
    ... )
    """
    target = matrix_device(n, dtype, device)
    scaling = finite_scalar(scaling, "scaling")
    generator = seeded_generator(seed, target)
    a = torch.randn(n, n, generator=generator, dtype=dtype, device=target)
    return scaling * (a - a.t())
