"""Random orthogonal recurrent matrix.

Orthogonal reservoirs have all singular values equal to 1, giving
norm-preserving linear dynamics: long memory without the eigenvalue spread
of random matrices. A useful structured alternative to sparse random
topologies, especially for memory-capacity tasks.
"""

import torch

from resdag.init.topology import register_matrix_topology

from ._common import finite_scalar, haar_orthogonal, matrix_device, seeded_generator, working_dtype


@register_matrix_topology("orthogonal", prescaled=True)
def orthogonal_matrix(
    n: int,
    gain: float = 1.0,
    seed: int | None = None,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build a random orthogonal matrix via QR decomposition.

    Draws a standard Gaussian matrix, takes its QR decomposition, and fixes
    the signs so the result is drawn from the Haar (uniform) distribution
    over orthogonal matrices.

    This topology is **pre-scaled**: an orthogonal matrix already has all its
    singular values equal to `abs(gain)` (norm-preserving for `abs(gain)=1`), which is
    the structural property it exists to provide. Because the layer-level
    spectral-radius rescale would collapse every singular value to the target
    radius and destroy that property, it is suppressed for this topology. A
    layer `spectral_radius` passed alongside this topology is therefore
    **ignored** (with a warning); control the scale through `gain` instead.
    An explicit `get_recurrent_initializer("orthogonal", rescale=True)`
    configuration opts into the outer scale, changing the specified gain.

    Parameters
    ----------
    n : int
        Matrix size (number of reservoir units).
    gain : float, default=1.0
        Scaling factor applied to the orthogonal matrix. With `abs(gain)=1` all
        singular values are 1 (norm-preserving). Since the layer-level
        `spectral_radius` rescale is suppressed (pre-scaled topology),
        `gain` is the only knob on the overall scale.
    seed : int, optional
        Seed for the Gaussian draw. `None` uses the global torch RNG.
    dtype : torch.dtype, default=torch.float32
        Real floating-point output dtype. CPU QR retains float64 precision;
        accelerator QR uses the requested dtype, promoting half to float32.
    device : torch.device or str, optional
        Output device (CPU by default). CUDA uses native QR. MPS's unsupported
        QR operation runs on CPU and the resulting factor returns to MPS.

    Returns
    -------
    torch.Tensor
        Dense orthogonal matrix of shape `(n, n)` (times `gain`). Setup
        costs O(n³); storage and dense matrix-vector multiplication cost O(n²).

    Examples
    --------
    >>> from resdag.layers import ESNLayer
    >>> reservoir = ESNLayer(500, feedback_size=3, topology="orthogonal")
    >>> reservoir = ESNLayer(
    ...     500, feedback_size=3, topology=("orthogonal", {"seed": 42})
    ... )
    """
    target = matrix_device(n, dtype, device)
    gain = finite_scalar(gain, "gain")
    generator = seeded_generator(seed, target)
    # Retain the historical high-precision CPU draw. Accelerators compute in
    # the requested dtype (promoting half precision for QR).
    compute_dtype = torch.float64 if target.type == "cpu" else working_dtype(dtype)
    q = haar_orthogonal(n, dtype=compute_dtype, device=target, generator=generator)
    return (gain * q).to(dtype=dtype)
