"""Exact gather specialization for validated CSR recurrence with one entry per row."""

import torch


def one_entry_per_row_columns(matrix: torch.Tensor) -> torch.Tensor | None:
    """Return aliased column indices when CSR rows each contain one stored entry.

    Call when structure is installed, not for every timestep. The caller owns
    CSR invariant validation and invalidates cached eligibility if indices are
    replaced or edited. Value edits do not change this structural property.
    Repeated columns are valid: a permutation is not required.
    """
    if matrix.layout != torch.sparse_csr or matrix.ndim != 2:
        return None
    n, f = matrix.shape
    if n != f or n == 0 or matrix.values().numel() != n:
        return None
    if not bool(torch.all(matrix.crow_indices().diff() == 1)):
        return None
    return matrix.col_indices()


def one_entry_recurrent(
    state: torch.Tensor, columns: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    """Compute ``state @ W.T`` with one selected weighted coordinate per row.

    Inputs follow the cell's already-validated batch/state/device/dtype contract.
    Every result has one product per entry; upstream state gradients accumulate
    correctly when several rows select the same input coordinate. Values remain
    live rather than copied into a persistent dense or padded representation.

    CPU float64 batches use the transposed contiguous RHS layout also used by
    CSR multiplication. Paired CPU measurements found this avoids slow strided
    double-precision gathers. Other cases use the direct last-axis selection.
    """
    if state.device.type == "cpu" and state.dtype == torch.float64 and state.shape[0] > 1:
        selected = state.T.contiguous().index_select(0, columns)
        return (selected * values[:, None]).T
    return state.index_select(-1, columns) * values
