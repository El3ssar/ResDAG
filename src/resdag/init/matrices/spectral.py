"""Recurrent matrices with prescribed singular values or eigenvalues.

These builders preserve the supplied full spectrum by default. They construct
dense matrices and use QR decompositions, so setup costs O(n³) and storage and
matrix-vector multiplication cost O(n²).
"""

import math
from collections.abc import Sequence

import torch

from resdag.init.topology import register_matrix_topology

from ._common import haar_orthogonal, matrix_device, seeded_generator, working_dtype


@register_matrix_topology("singular_spectrum", prescaled=True)
def singular_spectrum(
    n: int,
    singular_values: Sequence[float] | torch.Tensor | None = None,
    seed: int | None = None,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build `U @ diag(singular_values) @ V.T` from independent Haar factors.

    Supply exactly `n` finite nonnegative singular values, in any order.
    If omitted, they decrease linearly from 1 to 0.1 (`n=1` uses 1).
    Zero singular values allow rank-deficient reservoirs. This controls the
    one-step linear gains, not the eigenvalues or spectral radius. Nonlinear
    or leaky reservoir updates need their own stability analysis.

    `seed` uses a local RNG on `device`; `None` uses torch's global RNG.
    `dtype`/`device` control allocation and arithmetic; half-precision
    decompositions are computed in float32 then cast to the requested dtype.
    MPS QR uses a CPU fallback; CUDA QR stays on CUDA.
    An outer radius rescale would change all supplied singular values, so this
    builder is registered as prescaled. Explicit `rescale=True` opts in.

    Parameters
    ----------
    n : int
        Matrix size, a positive integer.
    singular_values : sequence of float or torch.Tensor, optional
        One-dimensional sequence of exactly n finite nonnegative values.
        Order is arbitrary. None uses a linear profile from 1 down to 0.1.
        Python sequences are converted directly into the computation dtype.
    seed : int, optional
        Local RNG seed. Repeated calls with the same seed, device, dtype,
        and parameters produce the same matrix without changing global RNG.
    dtype : torch.dtype, default=torch.float32
        Real floating-point output dtype. Float64 computations retain the
        supplied double precision; half/bfloat16 QR is promoted to float32.
    device : torch.device or str, optional
        Output device (CPU by default). MPS QR uses a CPU fallback.

    Returns
    -------
    torch.Tensor
        Dense `(n, n)` real matrix with the specified singular values, up
        to numerical precision. Setup is O(n³); storage is O(n²).

    Raises
    ------
    ValueError
        If size, dtype, spectrum shape, signs, or finiteness are invalid.
    TypeError
        If seed is neither an integer nor None.

    Examples
    --------
    >>> matrix = singular_spectrum(3, [1.0, 0.5, 0.0], seed=4, dtype=torch.float64)
    >>> from resdag.init import get_recurrent_initializer
    >>> from resdag.layers import ESNLayer
    >>> reservoir = ESNLayer(
    ...     3, feedback_size=1,
    ...     recurrent_initializer=get_recurrent_initializer(
    ...         "singular_spectrum", singular_values=[1.0, 0.5, 0.0],
    ...         seed=4, rescale=False,
    ...     ),
    ... )
    """
    target = matrix_device(n, dtype, device)
    compute_dtype = working_dtype(dtype)
    if singular_values is None:
        values = torch.linspace(1.0, 0.1, n, dtype=compute_dtype, device=target)
    else:
        raw = torch.as_tensor(singular_values)
        if raw.is_complex():
            raise ValueError("singular_values must be real, finite, and nonnegative")
        values = torch.as_tensor(singular_values, dtype=compute_dtype, device=target)
    if values.ndim != 1 or values.numel() != n:
        raise ValueError(f"singular_values must contain exactly n={n} values")
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise ValueError("singular_values must be real, finite, and nonnegative")

    generator = seeded_generator(seed, target)
    u = haar_orthogonal(n, dtype=compute_dtype, device=target, generator=generator)
    v = haar_orthogonal(n, dtype=compute_dtype, device=target, generator=generator)
    return ((u * values) @ v.T).to(dtype=dtype)


def _real_spectral_blocks(
    n: int, eigenvalues: Sequence[complex] | torch.Tensor | None
) -> list[tuple[float, float]]:
    """Pair conjugates; return real 1x1 and real 2x2 block parameters."""
    if eigenvalues is None:
        supplied = torch.linspace(-0.9, 0.9, n, dtype=torch.float64).tolist()
    else:
        raw = torch.as_tensor(eigenvalues, dtype=torch.complex128, device="cpu")
        if raw.ndim != 1 or raw.numel() != n:
            raise ValueError(f"eigenvalues must contain exactly n={n} values")
        supplied = raw.tolist()
    values = [complex(value) for value in supplied]
    if any(not (math.isfinite(v.real) and math.isfinite(v.imag)) for v in values):
        raise ValueError("eigenvalues must be finite")

    blocks: list[tuple[float, float]] = []
    while values:
        value = values.pop(0)
        if value.imag == 0:
            blocks.append((value.real, 0.0))
            continue
        partner = next(
            (
                i
                for i, candidate in enumerate(values)
                if math.isclose(candidate.real, value.real, rel_tol=1e-12, abs_tol=1e-14)
                and math.isclose(candidate.imag, -value.imag, rel_tol=1e-12, abs_tol=1e-14)
            ),
            None,
        )
        if partner is None:
            raise ValueError(
                "Nonreal eigenvalues must occur in complex-conjugate pairs; "
                f"missing conjugate of {value}"
            )
        values.pop(partner)
        blocks.append((value.real, abs(value.imag)))
    return blocks


@register_matrix_topology("eigen_spectrum", prescaled=True)
def eigen_spectrum(
    n: int,
    eigenvalues: Sequence[complex] | torch.Tensor | None = None,
    seed: int | None = None,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build a real normal matrix with prescribed real/complex-pair eigenvalues.

    Supply exactly `n` finite eigenvalues, with nonreal values in conjugate
    pairs (order is arbitrary). A pair `a ± bi` becomes a real block
    `[[a, -b], [b, a]]`. A Haar orthogonal similarity transforms the block
    diagonal matrix without changing its eigenvalues. If omitted, eigenvalues
    are real and spaced from -0.9 to 0.9 (`n=1` uses -0.9).

    The result is normal: its singular values are the eigenvalue magnitudes.
    This construction does not control nonnormal transient amplification.
    `dtype` is real floating-point; float64 retains double precision.
    `seed` is local and device-correct; half-precision QR uses float32.
    MPS QR runs on CPU; the returned matrix remains on the requested device.
    The complete spectrum is preserved unless outer rescaling is explicitly
    enabled with `get_recurrent_initializer(..., rescale=True)`.

    Parameters
    ----------
    n : int
        Matrix size, a positive integer.
    eigenvalues : sequence of complex or torch.Tensor, optional
        Exactly n finite real or complex values. Nonreal values must come in
        conjugate pairs, in any order; conjugates are matched within a relative
        tolerance of 1e-12 and absolute tolerance of 1e-14. A matched pair uses
        the first supplied value and its conjugate. None uses real values
        linearly spaced from -0.9 to 0.9.
    seed : int, optional
        Seed of the local RNG used for the orthogonal similarity factor.
        None draws from the global torch RNG.
    dtype : torch.dtype, default=torch.float32
        Real floating-point output dtype. Float64 retains double precision;
        half/bfloat16 QR is computed in float32 then cast.
    device : torch.device or str, optional
        Output device (CPU by default). MPS QR uses a CPU fallback.

    Returns
    -------
    torch.Tensor
        Dense real normal `(n, n)` matrix whose eigenvalues equal the
        supplied values up to numerical precision. Setup is O(n³), storage O(n²).

    Raises
    ------
    ValueError
        If size, dtype, number of eigenvalues, finiteness, or conjugate
        pairing is invalid. A real matrix cannot have an unpaired nonreal root.
    TypeError
        If seed is neither an integer nor None.

    Examples
    --------
    >>> matrix = eigen_spectrum(3, [0.5, 0.2 + 0.7j, 0.2 - 0.7j], seed=4)
    >>> from resdag.init import get_recurrent_initializer
    >>> from resdag.layers import ESNLayer
    >>> reservoir = ESNLayer(
    ...     3, feedback_size=1,
    ...     recurrent_initializer=get_recurrent_initializer(
    ...         "eigen_spectrum", eigenvalues=[0.5, 0.2 + 0.7j, 0.2 - 0.7j],
    ...         seed=4, rescale=False,
    ...     ),
    ... )
    """
    target = matrix_device(n, dtype, device)
    blocks = _real_spectral_blocks(n, eigenvalues)
    compute_dtype = working_dtype(dtype)
    block_matrix = torch.zeros(n, n, dtype=compute_dtype, device=target)
    position = 0
    for real, imaginary in blocks:
        block_matrix[position, position] = real
        if imaginary:
            block_matrix[position + 1, position + 1] = real
            block_matrix[position, position + 1] = -imaginary
            block_matrix[position + 1, position] = imaginary
            position += 2
        else:
            position += 1
    generator = seeded_generator(seed, target)
    q = haar_orthogonal(n, dtype=compute_dtype, device=target, generator=generator)
    return (q @ block_matrix @ q.T).to(dtype=dtype)
