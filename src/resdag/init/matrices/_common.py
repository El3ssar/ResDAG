"""Shared validation and random draws for direct recurrent matrix builders."""

import math
import operator
from typing import cast

import torch


def validate_size(n: int) -> None:
    """Require a positive integer matrix size."""
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError(f"Matrix size n must be a positive integer, got {n!r}")


def matrix_device(n: int, dtype: torch.dtype, device: torch.device | str | None) -> torch.device:
    """Validate size/dtype and resolve the requested allocation device."""
    validate_size(n)
    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError(f"dtype must be a real floating-point dtype, got {dtype}")
    return torch.device("cpu" if device is None else device)


def finite_scalar(value: float, name: str) -> float:
    """Convert a real scalar and reject NaN/infinity before allocating."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value}")
    return result


def seeded_generator(seed: int | None, device: torch.device) -> torch.Generator | None:
    """Use a local device-correct RNG for explicit seeds, global RNG otherwise."""
    if seed is None:
        return None
    if isinstance(seed, bool):
        raise TypeError("seed must be an integer or None")
    try:
        seed_value = operator.index(seed)
    except TypeError as exc:
        raise TypeError("seed must be an integer or None") from exc
    return torch.Generator(device=device).manual_seed(seed_value)


def working_dtype(dtype: torch.dtype) -> torch.dtype:
    """Promote low-precision decompositions and reflector arithmetic."""
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def haar_orthogonal(
    n: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Draw a Haar factor by signed QR; MPS's unsupported QR runs on CPU."""
    gaussian = torch.randn(n, n, generator=generator, dtype=dtype, device=device)
    qr_input = gaussian.cpu() if device.type == "mps" else gaussian
    q, r = torch.linalg.qr(qr_input)
    diagonal = torch.diagonal(r)
    signs = torch.where(diagonal < 0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
    return cast(torch.Tensor, (q * signs).to(device=device))
