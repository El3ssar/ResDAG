"""Sparse-native construction for exact, identity-checked built-in matrix cases.

This is an internal optimization of existing constructors, not another registry.
Unrecognized functions/settings return None without executing user callbacks.
Outer spectral-radius and prescaled policies remain MatrixTopology's concern.
"""

import inspect
from collections.abc import Callable
from typing import Any

import torch


def _csr_edges(
    n: int, rows: torch.Tensor, columns: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    # Dense builders' zero entries disappear during an ordinary CSR conversion.
    # Retain that convention, including a zero gain or zero connection weight.
    keep = values != 0
    indices = torch.stack((rows[keep], columns[keep]))
    return (
        torch.sparse_coo_tensor(
            indices, values[keep], size=(n, n), device=values.device, dtype=values.dtype
        )
        .coalesce()
        .to_sparse_csr()
    )


def known_matrix_csr(
    matrix_func: Callable,
    n: int,
    matrix_kwargs: dict[str, Any],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    """Build a supported existing matrix exactly using only linear-size storage.

    Supports simple cycles, forward/backward delay lines, signed permutations,
    and diagonal cases of band/Toeplitz/block-diagonal/zero-reflector builders.
    Explicit builder placement overrides are respected before conversion to the
    destination, matching MatrixTopology.initialize. Random draws and caller RNG
    state match the original signed-permutation builder.
    """
    from resdag.init.matrices._common import (
        finite_scalar,
        matrix_device,
        seeded_generator,
        working_dtype,
    )
    from resdag.init.matrices.minimal import delay_line, delay_line_backward, simple_cycle
    from resdag.init.matrices.orthogonal_structures import (
        householder_orthogonal,
        signed_permutation,
    )
    from resdag.init.matrices.structured import band, block_diagonal, toeplitz

    known = (
        simple_cycle,
        delay_line,
        delay_line_backward,
        signed_permutation,
        band,
        block_diagonal,
        toeplitz,
        householder_orthogonal,
    )
    if not any(matrix_func is function for function in known):
        return None
    options = dict(matrix_kwargs)
    options.setdefault("dtype", dtype)
    options.setdefault("device", device)
    bound = inspect.signature(matrix_func).bind(n, **options)
    bound.apply_defaults()
    config = bound.arguments
    if matrix_func is band and config["bandwidth"] != 0:
        return None
    if matrix_func is toeplitz and config["bandwidth"] != 0:
        return None
    if matrix_func is toeplitz:
        float(config["decay"])  # Preserve validation even when only the diagonal remains.
    if matrix_func is block_diagonal and (
        config["block_size"] != 1 or not isinstance(config["block_size"], int)
    ):
        return None
    if matrix_func is householder_orthogonal and config["reflectors"] != 0:
        return None
    work_dtype = config["dtype"]
    target = matrix_device(n, work_dtype, config["device"])
    rows = torch.arange(n, device=target)
    if matrix_func is signed_permutation:
        gain = finite_scalar(config["gain"], "gain")
        generator = seeded_generator(config["seed"], target)
        columns = torch.randperm(n, generator=generator, device=target)
        values = torch.randint(0, 2, (n,), generator=generator, device=target).to(work_dtype)
        values = values.mul_(2).sub_(1).mul_(gain)
    elif matrix_func is simple_cycle:
        columns = (rows - 1).remainder(n)
        values = torch.full((n,), float(config["weight"]), dtype=work_dtype, device=target)
    elif matrix_func in (delay_line, delay_line_backward):
        shift = config["shift"]
        if not isinstance(shift, int) or not 1 <= shift < n:
            raise ValueError(f"shift must satisfy 1 <= shift < n={n}, got {shift}")
        columns = torch.arange(n - shift, device=target)
        rows = columns + shift
        values = torch.full((n - shift,), float(config["weight"]), dtype=work_dtype, device=target)
        if matrix_func is delay_line_backward:
            backward = config["backward_shift"]
            if not isinstance(backward, int) or not 1 <= backward < n:
                raise ValueError(
                    f"backward_shift must satisfy 1 <= backward_shift < n={n}, got {backward}"
                )
            back_rows = torch.arange(n - backward, device=target)
            rows = torch.cat((rows, back_rows))
            columns = torch.cat((columns, back_rows + backward))
            back_values = torch.full(
                (n - backward,), float(config["backward_weight"]), dtype=work_dtype, device=target
            )
            values = torch.cat((values, back_values))
    else:
        columns = rows
        if matrix_func is householder_orthogonal:
            gain = finite_scalar(config["gain"], "gain")
            if isinstance(config["reflectors"], bool) or not isinstance(config["reflectors"], int):
                raise ValueError("reflectors must be a nonnegative integer or None")
            seeded_generator(config["seed"], target)  # validation; zero reflectors consume no draws
            values = (torch.ones(n, dtype=working_dtype(work_dtype), device=target) * gain).to(
                work_dtype
            )
        else:
            values = torch.full((n,), float(config["weight"]), dtype=work_dtype, device=target)
    result = _csr_edges(n, rows, columns, values)
    # Convert before movement so MPS never receives a transient float64 tensor.
    return result.to(dtype=dtype).to(device=device)
