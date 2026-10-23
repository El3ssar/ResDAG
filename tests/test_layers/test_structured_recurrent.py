"""Independent CSR and explicit-coordinate oracles for the gather recurrence."""

import pytest
import torch

from resdag.layers.cells._structured_recurrent import (
    one_entry_per_row_columns,
    one_entry_recurrent,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("n", [1, 11, 47])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("repeated_columns", [False, True])
def test_gather_matches_csr_outputs_and_state_gradients(dtype, n, batch, repeated_columns):
    generator = torch.Generator().manual_seed(83)
    columns = (
        torch.randint(0, n, (n,), generator=generator)
        if repeated_columns
        else torch.randperm(n, generator=generator)
    )
    values = torch.randn(n, dtype=dtype, generator=generator)
    matrix = torch.sparse_csr_tensor(torch.arange(n + 1), columns, values, size=(n, n))
    eligible = one_entry_per_row_columns(matrix)
    assert eligible is not None and eligible.data_ptr() == matrix.col_indices().data_ptr()
    # A noncontiguous state exercises both possible gather layouts.
    state = torch.randn(batch, n * 2, dtype=dtype, generator=generator)[:, ::2].requires_grad_()
    expected = torch.sparse.mm(matrix, state.T.contiguous()).T
    actual = one_entry_recurrent(state, eligible, matrix.values())
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    objective = torch.randn(batch, n, dtype=dtype, generator=generator)
    gradient_expected = torch.autograd.grad((expected * objective).sum(), state)[0]
    gradient_actual = torch.autograd.grad((actual * objective).sum(), state)[0]
    torch.testing.assert_close(
        gradient_actual,
        gradient_expected,
        atol=2e-6 if dtype == torch.float32 else 1e-14,
        rtol=2e-6 if dtype == torch.float32 else 1e-14,
    )
    # Independent coordinate accumulation, including duplicate selected columns.
    oracle = torch.zeros_like(state)
    for row, column in enumerate(columns.tolist()):
        oracle[:, column] += objective[:, row] * values[row]
    torch.testing.assert_close(
        gradient_actual,
        oracle,
        atol=2e-6 if dtype == torch.float32 else 1e-14,
        rtol=2e-6 if dtype == torch.float32 else 1e-14,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_values_remain_live_without_dense_or_padded_weight_storage(dtype):
    matrix = torch.eye(7, dtype=dtype).to_sparse_csr()
    columns = one_entry_per_row_columns(matrix)
    state = torch.arange(14, dtype=dtype).reshape(2, 7)
    before = one_entry_recurrent(state, columns, matrix.values())
    matrix.values().mul_(0.25)
    torch.testing.assert_close(
        one_entry_recurrent(state, columns, matrix.values()), before * 0.25, atol=0, rtol=0
    )
    assert columns.numel() == 7 and matrix.values().numel() == 7


@pytest.mark.parametrize("kind", ["dense", "rectangular", "empty", "uneven", "extra_entries"])
def test_eligibility_rejects_nonmatching_structure(kind):
    if kind == "dense":
        value = torch.eye(4)
    elif kind == "rectangular":
        value = torch.ones(4, 1).to_sparse_csr()
    elif kind == "empty":
        value = torch.empty(0, 0).to_sparse_csr()
    elif kind == "uneven":
        value = torch.tensor([[1.0, 1.0], [0.0, 0.0]]).to_sparse_csr()
    else:
        value = torch.ones(4, 4).to_sparse_csr()
    assert one_entry_per_row_columns(value) is None
