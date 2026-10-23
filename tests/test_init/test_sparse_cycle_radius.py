"""Analytic functional-CSR spectra, independently checked against dense eigvals."""

import math
import random

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from resdag.init.topology import base, get_topology


def _csr(columns, weights, *, dtype=torch.float64):
    pointers, indices, values = [0], [], []
    for column, weight in zip(columns, weights, strict=True):
        if column is not None:
            indices.append(column)
            values.append(weight)
        pointers.append(len(indices))
    return torch.sparse_csr_tensor(
        torch.tensor(pointers),
        torch.tensor(indices, dtype=torch.int64),
        torch.tensor(values, dtype=dtype),
        size=(len(columns), len(columns)),
        check_invariants=True,
    )


def _forbid_fallbacks(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A functional CSR matrix must not need an eigensolve or dense fallback")

    monkeypatch.setattr(base, "_sparse_eigs_radius", forbidden)
    monkeypatch.setattr(base, "_dense_eigvals_radius", forbidden)
    monkeypatch.setattr(torch.Tensor, "to_dense", forbidden)


@pytest.mark.parametrize("n", [400, 20000])
@pytest.mark.parametrize("weight", [0.7, -0.3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_uniform_cycle_normalizes_at_scale_without_dense_storage_or_eigensolves(
    monkeypatch, n, weight, dtype
):
    _forbid_fallbacks(monkeypatch)

    class RejectDenseSquare(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            result = function(*args, **(kwargs or {}))
            for value in tree_leaves(result):
                if isinstance(value, torch.Tensor) and value.layout == torch.strided:
                    assert value.shape != (n, n), f"Dense square allocation by {function}"
            return result

    topology = get_topology("simple_cycle", weight=weight)
    with RejectDenseSquare():
        raw = topology.build_sparse(n, dtype=dtype, device=torch.device("cpu"))
        assert base.estimate_spectral_radius(raw) == abs(float(torch.tensor(weight, dtype=dtype)))
        scaled = topology.build_sparse(
            n, dtype=dtype, device=torch.device("cpu"), spectral_radius=0.9
        )
        assert base.estimate_spectral_radius(scaled) == float(torch.tensor(0.9, dtype=dtype))
    assert scaled.layout == torch.sparse_csr and scaled.values().numel() == n
    torch.testing.assert_close(
        scaled.values(),
        torch.full((n,), math.copysign(0.9, weight), dtype=dtype),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_small_functional_graph_matches_independent_dense_eigenvalues(monkeypatch, seed, dtype):
    generator = random.Random(seed)
    columns = [generator.randrange(9) if row % 4 else None for row in range(9)]
    weights = [generator.uniform(-2, 2) for _ in range(9)]
    if dtype == torch.complex128:
        weights = [complex(value, generator.uniform(-2, 2)) for value in weights]
    # Guarantee a nontriangular component; the rest can be trees, self-loops,
    # disconnected cycles, terminal rows or explicitly stored zero edges.
    columns[:2], weights[:2] = [1, 0], [0.7, -1.3]
    weights[5] = 0
    matrix = _csr(columns, weights, dtype=dtype)
    expected = float(torch.linalg.eigvals(matrix.to_dense()).abs().max())
    before = [
        value.clone() for value in (matrix.crow_indices(), matrix.col_indices(), matrix.values())
    ]
    _forbid_fallbacks(monkeypatch)

    actual = base.estimate_spectral_radius(matrix)

    assert actual == pytest.approx(expected, rel=2e-12, abs=2e-14)
    for value, original in zip(
        (matrix.crow_indices(), matrix.col_indices(), matrix.values()), before, strict=True
    ):
        torch.testing.assert_close(value, original, rtol=0, atol=0)


def test_multiple_weighted_cycles_ignore_large_transient_branch_weights(monkeypatch):
    # Three cycles have radii 2, 3 and 0.25. Rows 5--6 feed the first cycle;
    # their large weights affect transient amplification, not the spectrum.
    matrix = _csr([1, 2, 0, 4, 3, 0, 5, 7], [-2, 4, 1, 3, -3, 1e200, -1e200, 0.25])
    _forbid_fallbacks(monkeypatch)
    assert base.estimate_spectral_radius(matrix) == pytest.approx(3.0, rel=1e-15)
    scaled = base.scale_to_spectral_radius(matrix, 0.9)
    assert base.estimate_spectral_radius(scaled) == pytest.approx(0.9, rel=1e-15)


@pytest.mark.parametrize("magnitude", [1e-300, 1e300])
def test_cycle_geometric_mean_avoids_product_underflow_and_overflow(monkeypatch, magnitude):
    matrix = _csr([1, 2, 0], [magnitude, magnitude, 1 / magnitude])
    expected = math.exp(math.log(magnitude) / 3)
    _forbid_fallbacks(monkeypatch)
    assert base.estimate_spectral_radius(matrix) == pytest.approx(expected, rel=2e-13)


def test_complex_cycle_handles_individual_magnitudes_beyond_float_range(monkeypatch):
    weights = [complex(1.7e308, 1.7e308), complex(1e-308, 1e-308), 1j]
    # Multiplying the reciprocal-scale pair first keeps this independent
    # three-edge product representable, although abs(weights[0]) overflows.
    expected = abs(weights[0] * weights[1] * weights[2]) ** (1 / 3)
    matrix = _csr([1, 2, 0], weights, dtype=torch.complex128)
    _forbid_fallbacks(monkeypatch)
    assert base.estimate_spectral_radius(matrix) == pytest.approx(expected, rel=2e-13)


@pytest.mark.parametrize("bad", [complex(math.inf, 0), complex(0, math.inf), complex(math.nan, 1)])
def test_complex_csr_nonfinite_values_are_rejected_before_spectral_work(monkeypatch, bad):
    matrix = _csr([1, 2, 0], [1j, bad, 0.7], dtype=torch.complex128)
    _forbid_fallbacks(monkeypatch)
    with pytest.raises(ValueError, match="only finite values"):
        base.estimate_spectral_radius(matrix)


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_general_csr_with_zeroed_stored_entries_is_zero_without_spectral_work(monkeypatch, dtype):
    matrix = torch.sparse_csr_tensor(
        torch.tensor([0, 2, 3, 5]),
        torch.tensor([1, 2, 0, 0, 1]),
        torch.arange(1, 6, dtype=torch.float64).to(dtype),
        size=(3, 3),
        check_invariants=True,
    )
    matrix.values().zero_()
    before = [
        value.clone() for value in (matrix.crow_indices(), matrix.col_indices(), matrix.values())
    ]
    _forbid_fallbacks(monkeypatch)

    assert matrix._nnz() == 5
    assert base.estimate_spectral_radius(matrix) == 0.0
    with pytest.warns(RuntimeWarning, match="zero spectral radius"):
        scaled = base.scale_to_spectral_radius(matrix, 0.9)
    assert scaled is matrix
    for value, original in zip(
        (matrix.crow_indices(), matrix.col_indices(), matrix.values()), before, strict=True
    ):
        torch.testing.assert_close(value, original, rtol=0, atol=0)


@pytest.mark.parametrize(
    "columns,weights", [([2, 0, 1], [2.0, 0.0, 4.0]), ([2, None, 1], [2.0, 0.0, 4.0])]
)
def test_broken_cycle_or_terminal_graph_is_nilpotent_and_warns_on_rescale(
    monkeypatch, columns, weights
):
    matrix = _csr(columns, weights)
    _forbid_fallbacks(monkeypatch)
    assert base.estimate_spectral_radius(matrix) == 0.0
    with pytest.warns(RuntimeWarning, match="zero spectral radius"):
        scaled = base.scale_to_spectral_radius(matrix, 0.9)
    assert scaled is matrix


@pytest.mark.parametrize("returned_radius", [2.5, None])
def test_nonfunctional_csr_keeps_arnoldi_and_its_fail_closed_policy(monkeypatch, returned_radius):
    matrix = torch.tensor([[0.0, 1.0, 2.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    sparse = matrix.to_sparse_csr()
    calls = []

    def arnoldi(value):
        assert value is sparse
        calls.append(value)
        return returned_radius

    def forbidden(*args, **kwargs):
        pytest.fail("General CSR failure must not allocate a dense fallback")

    monkeypatch.setattr(base, "_sparse_eigs_radius", arnoldi)
    monkeypatch.setattr(base, "_dense_eigvals_radius", forbidden)
    monkeypatch.setattr(torch.Tensor, "to_dense", forbidden)
    if returned_radius is None:
        with pytest.raises(RuntimeError, match="no dense fallback"):
            base.estimate_spectral_radius(sparse)
    else:
        assert base.estimate_spectral_radius(sparse) == returned_radius
    assert len(calls) == 1
