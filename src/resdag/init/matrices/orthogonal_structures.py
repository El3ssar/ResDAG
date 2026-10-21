"""Signed permutations and products of Householder reflections.

Both return ordinary dense tensors for compatibility with reservoir cells.
Their mathematical structure does not imply sparse or implicit recurrent
execution: storage and dense matrix-vector multiplication remain O(n²).
"""

import torch

from resdag.init.topology import register_matrix_topology

from ._common import finite_scalar, matrix_device, seeded_generator, working_dtype


@register_matrix_topology("signed_permutation", prescaled=True)
def signed_permutation(
    n: int,
    gain: float = 1.0,
    seed: int | None = None,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build a random permutation with independent random signs, times ``gain``.

    Each row and column has exactly one nonzero when ``gain != 0``. All
    singular values and eigenvalue magnitudes equal ``abs(gain)``. These
    matrices preserve norm for ``abs(gain)=1`` while allowing signed cycles
    instead of mixing every state coordinate as a dense Haar draw does.
    Setup is O(n²) because the returned tensor is dense. The permutation and
    signs use a local device RNG when ``seed`` is given.

    Parameters
    ----------
    n : int
        Matrix size, a positive integer.
    gain : float, default=1.0
        Finite multiplier on the signed permutation. All singular values
        equal abs(gain). Zero gives an all-zero matrix; negative gains are valid.
    seed : int, optional
        Local seed for the random permutation and signs. None uses torch's
        global RNG; an explicit seed leaves the global RNG unchanged.
    dtype : torch.dtype, default=torch.float32
        Real floating-point output dtype.
    device : torch.device or str, optional
        Allocation and random-draw device (CPU by default).

    Returns
    -------
    torch.Tensor
        Dense ``(n, n)`` matrix. The output is not an implicit or sparse
        operator; recurrent application still costs O(n²).

    Raises
    ------
    ValueError
        If size or dtype is invalid, or gain is nonfinite.
    TypeError
        If seed is neither an integer nor None.

    Examples
    --------
    >>> matrix = signed_permutation(100, gain=0.9, seed=4)
    >>> from resdag.layers import ESNLayer
    >>> reservoir = ESNLayer(
    ...     100, feedback_size=3, spectral_radius=None,
    ...     recurrent_initializer=("signed_permutation", {"gain": 0.9, "seed": 4}),
    ... )
    """
    target = matrix_device(n, dtype, device)
    gain = finite_scalar(gain, "gain")
    generator = seeded_generator(seed, target)
    columns = torch.randperm(n, generator=generator, device=target)
    signs = torch.randint(0, 2, (n,), generator=generator, device=target).to(dtype)
    signs = signs.mul_(2).sub_(1).mul_(gain)
    matrix = torch.zeros(n, n, dtype=dtype, device=target)
    matrix[torch.arange(n, device=target), columns] = signs
    return matrix


@register_matrix_topology("householder_orthogonal", prescaled=True)
def householder_orthogonal(
    n: int,
    gain: float = 1.0,
    reflectors: int | None = None,
    seed: int | None = None,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    r"""Build ``gain * H_k ... H_1``, with ``H_i = I - 2 v_i v_i.T``.

    Each ``v_i`` is an independently sampled Gaussian direction, normalized
    to unit norm. ``reflectors=None`` uses ``n`` reflections; zero gives a
    scaled identity. All singular values equal ``abs(gain)``. For ``k < n``
    at least ``n-k`` directions are unchanged before applying ``gain``; choose
    the number of reflections to balance mixing against initialization cost.

    Dense rank-one updates cost O(k n²), with O(n²) storage. Thus the default
    ``k=n`` costs O(n³); a fixed small ``k`` costs O(n²). The returned matrix
    still uses dense recurrent multiplication. This fixed-reflection product
    is not a Haar-uniform draw. Half-precision arithmetic is promoted to
    float32 until the result is cast to the requested dtype.

    Parameters
    ----------
    n : int
        Matrix size, a positive integer.
    gain : float, default=1.0
        Finite multiplier. All singular values equal abs(gain); zero and
        negative gains are allowed.
    reflectors : int or None, default=None
        Nonnegative number of reflections. None uses n; zero gives gain * I.
        A fixed small value reduces setup work but leaves more directions
        unchanged. The output is still a dense matrix.
    seed : int, optional
        Local seed for Gaussian reflection directions on the requested
        device. None uses torch's global RNG.
    dtype : torch.dtype, default=torch.float32
        Real floating-point output dtype. Half/bfloat16 arithmetic is
        promoted to float32 before the final cast.
    device : torch.device or str, optional
        Allocation, random-draw, and computation device (CPU by default).

    Returns
    -------
    torch.Tensor
        Dense scaled orthogonal ``(n, n)`` matrix. Construction costs
        O(reflectors * n²), and dense storage/application cost O(n²).

    Raises
    ------
    ValueError
        If size, dtype, reflector count, or gain is invalid.
    TypeError
        If seed is neither an integer nor None.

    Examples
    --------
    >>> matrix = householder_orthogonal(100, gain=0.9, reflectors=8, seed=4)
    >>> from resdag.layers import ESNLayer
    >>> reservoir = ESNLayer(
    ...     100, feedback_size=3, spectral_radius=None,
    ...     recurrent_initializer=("householder_orthogonal", {"reflectors": 8, "gain": 0.9}),
    ... )
    """
    target = matrix_device(n, dtype, device)
    gain = finite_scalar(gain, "gain")
    count = n if reflectors is None else reflectors
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(f"reflectors must be a nonnegative integer or None, got {reflectors}")
    compute_dtype = working_dtype(dtype)
    generator = seeded_generator(seed, target)
    matrix = torch.eye(n, dtype=compute_dtype, device=target)
    directions = torch.randn(count, n, generator=generator, dtype=compute_dtype, device=target)
    directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(
        torch.finfo(compute_dtype).tiny
    )
    for direction in directions:
        matrix = matrix - 2 * torch.outer(direction, direction @ matrix)
    return (gain * matrix).to(dtype=dtype)
