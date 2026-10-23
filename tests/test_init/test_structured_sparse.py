"""Sparse-native built-ins match existing dense constructors and exact RNG draws."""

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from resdag.init.matrices.minimal import delay_line, delay_line_backward, simple_cycle
from resdag.init.matrices.orthogonal_structures import householder_orthogonal, signed_permutation
from resdag.init.matrices.structured import band, block_diagonal, toeplitz
from resdag.init.topology._structured import known_matrix_csr

CASES = [
    (simple_cycle, {"weight": -0.3}),
    (simple_cycle, {"weight": 0.0}),
    (delay_line, {"weight": -0.3, "shift": 2}),
    (
        delay_line_backward,
        {"weight": -0.3, "backward_weight": 0.7, "shift": 2, "backward_shift": 1},
    ),
    (signed_permutation, {"gain": -0.3, "seed": 17}),
    (signed_permutation, {"gain": 0.0, "seed": 17}),
    (band, {"weight": -0.3, "bandwidth": 0}),
    (block_diagonal, {"weight": -0.3, "block_size": 1}),
    (toeplitz, {"weight": -0.3, "decay": -0.4, "bandwidth": 0}),
    (householder_orthogonal, {"gain": -0.3, "reflectors": 0, "seed": 17}),
]


@pytest.mark.parametrize(
    "builder,options", CASES, ids=[f.__name__ + str(i) for i, (f, _) in enumerate(CASES)]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_known_builders_are_exact_dense_equivalents(builder, options, dtype):
    expected = builder(7, **options, dtype=dtype)
    result = known_matrix_csr(builder, 7, options, dtype=dtype, device=torch.device("cpu"))
    assert result.layout == torch.sparse_csr and result.dtype == dtype
    torch.testing.assert_close(result.to_dense(), expected, atol=0, rtol=0)
    assert result.values().numel() <= 14


@pytest.mark.parametrize("seed", [None, 17])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_signed_permutation_matches_global_rng_consumption_and_explicit_seed(seed, dtype):
    torch.manual_seed(958)
    expected = signed_permutation(71, seed=seed, dtype=dtype)
    expected_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(958)
    actual = known_matrix_csr(
        signed_permutation, 71, {"seed": seed}, dtype=dtype, device=torch.device("cpu")
    )
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    torch.testing.assert_close(actual.to_dense(), expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "builder,options", CASES, ids=[f.__name__ + str(i) for i, (f, _) in enumerate(CASES)]
)
def test_explicit_builder_dtype_precedes_destination_conversion(builder, options):
    options = {**options, "dtype": torch.float32, "device": "cpu"}
    expected = builder(7, **options).double()
    actual = known_matrix_csr(builder, 7, options, dtype=torch.float64, device=torch.device("cpu"))
    torch.testing.assert_close(actual.to_dense(), expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "builder,options", CASES, ids=[f.__name__ + str(i) for i, (f, _) in enumerate(CASES)]
)
def test_large_known_builds_never_materialize_a_dense_recurrent_matrix(builder, options):
    n = 20000

    class RejectDenseSquare(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            result = function(*args, **(kwargs or {}))
            for value in tree_leaves(result):
                if isinstance(value, torch.Tensor) and value.layout == torch.strided:
                    assert value.shape != (n, n), f"Dense allocation by {function}"
            return result

    with RejectDenseSquare():
        result = known_matrix_csr(
            builder, n, options, dtype=torch.float64, device=torch.device("cpu")
        )
    assert result.layout == torch.sparse_csr and result.values().numel() <= 2 * n


@pytest.mark.parametrize(
    "builder,options",
    [
        (band, {"bandwidth": 2}),
        (block_diagonal, {"block_size": 2}),
        (toeplitz, {"bandwidth": None}),
        (householder_orthogonal, {"reflectors": 2}),
    ],
)
def test_other_known_cases_keep_existing_fallback(builder, options):
    assert (
        known_matrix_csr(builder, 7, options, dtype=torch.float64, device=torch.device("cpu"))
        is None
    )


def test_unknown_and_same_named_callbacks_are_never_executed_or_specialized():
    def simple_cycle(n, **kwargs):
        raise AssertionError("The sparse helper executed a custom callback")

    assert (
        known_matrix_csr(simple_cycle, 7, {}, dtype=torch.float64, device=torch.device("cpu"))
        is None
    )


@pytest.mark.parametrize(
    "builder,options",
    [
        (delay_line, {"shift": 0}),
        (delay_line_backward, {"backward_shift": 7}),
        (signed_permutation, {"gain": float("nan")}),
        (signed_permutation, {"seed": True}),
        (householder_orthogonal, {"reflectors": False}),
        (toeplitz, {"bandwidth": 0, "decay": "invalid"}),
    ],
)
def test_invalid_supported_options_fail_without_silent_math_changes(builder, options):
    with pytest.raises((ValueError, TypeError)):
        known_matrix_csr(builder, 7, options, dtype=torch.float64, device=torch.device("cpu"))
