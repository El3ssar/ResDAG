"""Independent mathematical, ownership and lifecycle checks for compact weights."""

import copy

import numpy as np
import pytest
import torch
from torch import nn

import resdag as rd
from resdag.init.graphs._sparse import _erdos_renyi_csr
from resdag.init.graphs.erdos_renyi import erdos_renyi_adjacency
from resdag.init.topology import TopologyInitializer, estimate_spectral_radius


@pytest.mark.parametrize("n", [1, 2, 31, 32, 33, 67])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("loops", [False, True])
def test_bounded_er_draws_exactly_match_dense_definition(n, directed, loops):
    expected = erdos_renyi_adjacency(n, 0.2, directed, loops, 17)
    actual = _erdos_renyi_csr(n, 0.2, directed, loops, 17)
    np.testing.assert_array_equal(actual.toarray(), expected)


def _cell(**kwargs):
    options = dict(
        reservoir_size=48,
        feedback_size=2,
        seed=17,
        sparse=True,
        dtype=torch.float64,
        topology=("erdos_renyi", {"p": 0.1}),
    )
    options.update(kwargs)
    return rd.ESNLayer(**options).cell


def test_er_scaled_construction_never_materializes_dense_square(monkeypatch):
    from resdag.init.graphs import _sparse

    original_rng = _sparse.create_rng
    draws = []

    class BoundedRNG:
        def __init__(self, seed):
            self.generator = original_rng(seed)

        def random(self, shape):
            draws.append(shape)
            assert shape[0] <= 32
            return self.generator.random(shape)

    def forbidden_dense(tensor, *args, **kwargs):
        raise AssertionError("Sparse initialization must not materialize a dense square")

    monkeypatch.setattr(_sparse, "create_rng", BoundedRNG)
    monkeypatch.setattr(torch.Tensor, "to_dense", forbidden_dense)
    cell = _cell(reservoir_size=96, spectral_radius=0.9)
    assert draws and cell.weight_hh.layout == torch.sparse_csr
    assert cell._parameters["weight_hh"] is None
    assert not any(
        tuple(t.shape) == (96, 96) and t.layout == torch.strided
        for t in (*cell.parameters(), *cell.buffers())
    )


@pytest.mark.parametrize("n", [3, 17, 67])
@pytest.mark.parametrize("directed", [False, True])
def test_compact_spectral_scaling_has_requested_radius(n, directed):
    cell = _cell(
        reservoir_size=n,
        spectral_radius=0.8,
        topology=("erdos_renyi", {"p": 0.6, "directed": directed}),
    )
    actual = torch.linalg.eigvals(cell.weight_hh.to_dense()).abs().max()
    torch.testing.assert_close(actual, actual.new_tensor(0.8), rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("kind", ["zero", "upper", "lower"])
def test_sparse_spectral_analytic_zero_and_triangular_cases(kind, monkeypatch):
    from resdag.init.topology import base

    matrix = torch.zeros(7, 7, dtype=torch.float64)
    if kind == "upper":
        matrix[1, 5] = 2
        matrix[2, 2] = -0.7
    elif kind == "lower":
        matrix[5, 1] = 2
        matrix[2, 2] = -0.7
    monkeypatch.setattr(base, "_sparse_eigs_radius", lambda _: pytest.fail("analytic case"))
    expected = float(torch.linalg.eigvals(matrix).abs().max())
    assert estimate_spectral_radius(matrix.to_sparse_csr()) == expected


def test_three_dimensional_complex_dominant_mode_stays_sparse(monkeypatch):
    matrix = torch.tensor([[0.0, -0.8, 0.1], [0.8, 0.0, 0.2], [0.0, 0.0, 0.1]], dtype=torch.float64)
    sparse = matrix.to_sparse_csr()
    expected = torch.linalg.eigvals(matrix).abs().max().item()

    def forbidden_dense(*args, **kwargs):
        pytest.fail("A three-dimensional complex mode must not need a dense fallback")

    monkeypatch.setattr(torch.Tensor, "to_dense", forbidden_dense)
    assert estimate_spectral_radius(sparse) == pytest.approx(expected, rel=1e-12)


class _OwnedCSR(TopologyInitializer):
    def __init__(self, matrix):
        self.matrix = matrix

    def initialize(self, weight, spectral_radius=None):
        raise AssertionError("A sparse-native initializer must not use dense scratch")

    def build_sparse(self, n, *, dtype, device, spectral_radius=None):
        return self.matrix


def test_custom_sparse_builder_owns_storage_and_honors_requested_precision():
    source = torch.eye(8, dtype=torch.float32).to_sparse_csr()
    cell = _cell(
        reservoir_size=8,
        topology=_OwnedCSR(source),
        spectral_radius=None,
        initial_state_initializer="normal",
    )
    expected = cell.weight_hh.to_dense().clone()
    source.values().zero_()
    torch.testing.assert_close(cell.weight_hh.to_dense(), expected, rtol=0, atol=0)
    assert cell.weight_hh.dtype == torch.float64
    assert cell._initial_state.layout == torch.strided
    assert cell._hh_values.data_ptr() == cell.weight_hh.values().data_ptr()


def test_custom_sparse_builder_rejects_invalid_indices_before_recurrence():
    broken = torch.sparse_csr_tensor(
        torch.tensor([0, 1, 1]),
        torch.tensor([9]),
        torch.ones(1),
        size=(2, 2),
        check_invariants=False,
    )
    with pytest.raises(RuntimeError):
        _cell(reservoir_size=2, topology=_OwnedCSR(broken))


def test_copy_share_memory_to_empty_and_reload_keep_pattern_and_independence():
    cell = _cell()
    pattern = (cell.weight_hh.crow_indices().clone(), cell.weight_hh.col_indices().clone())
    state = copy.deepcopy(cell.state_dict())
    clone = copy.deepcopy(cell).share_memory()
    assert clone.weight_hh.values().is_shared()
    assert clone._hh_values.data_ptr() == clone.weight_hh.values().data_ptr()
    assert clone._hh_values.data_ptr() != cell._hh_values.data_ptr()
    clone.to_empty(device="cpu")
    assert torch.equal(clone.weight_hh.crow_indices(), pattern[0])
    assert torch.equal(clone.weight_hh.col_indices(), pattern[1])
    clone.load_state_dict(state)
    torch.testing.assert_close(
        clone.weight_hh.to_dense(), cell.weight_hh.to_dense(), rtol=0, atol=0
    )


def test_inference_created_weights_do_not_rebuild_on_every_step(monkeypatch):
    with torch.inference_mode():
        cell = _cell()
        state = torch.zeros(2, 48, dtype=torch.float64)
        before = cell.weight_hh
        monkeypatch.setattr(cell, "_maybe_build_sparse_weight", lambda: pytest.fail("rebuild"))
        for _ in range(5):
            _, state = cell([torch.ones(2, 2, dtype=torch.float64)], state)
        assert cell.weight_hh is before


def test_dense_replacement_and_cross_storage_load_preserve_parameter_identity_and_gradients():
    cell = _cell()
    parameter = nn.Parameter(cell.weight_hh.to_dense(), requires_grad=True)
    cell.weight_hh = parameter
    inputs = torch.randn(2, 48, dtype=torch.float64, requires_grad=True)
    cell._recurrent(inputs).sum().backward()
    assert cell.weight_hh is parameter and parameter.requires_grad
    assert parameter.grad is not None and inputs.grad is not None
    cell.load_state_dict(_cell(seed=3).state_dict())
    assert cell.weight_hh is parameter and parameter.requires_grad


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_explicit_compact_storage_reactivates_after_mps_dense_bridge():
    cell = _cell(dtype=torch.float32)
    expected = cell.weight_hh.to_dense().clone()
    cell.to("mps")
    assert not cell.sparse and cell.weight_hh.layout == torch.strided
    cell.cpu()
    assert cell.sparse and cell.weight_hh.layout == torch.sparse_csr
    torch.testing.assert_close(cell.weight_hh.to_dense(), expected, rtol=0, atol=0)


def test_storage_scales_with_edge_count_not_dense_square():
    cell = _cell(reservoir_size=2048, topology=("simple_cycle_jumps", {"jump_length": 16}))
    weights = cell.weight_hh
    assert weights._nnz() < 4 * 2048
    actual_bytes = sum(
        t.untyped_storage().nbytes()
        for t in (cell._hh_crow_indices, cell._hh_col_indices, cell._hh_values)
    )
    assert actual_bytes < 2048**2 * 8 / 100
    assert not any(tuple(p.shape) == (2048, 2048) for p in cell.parameters())


@pytest.mark.parametrize("source_kind", ["torch", "scipy"])
def test_registered_sparse_matrix_builder_needs_no_dense_destination(source_kind, monkeypatch):
    import scipy.sparse as sp

    from resdag.init import register_recurrent_initializer
    from resdag.init.topology import registry

    monkeypatch.setattr(registry, "_TOPOLOGY_REGISTRY", dict(registry._TOPOLOGY_REGISTRY))
    calls = []

    @register_recurrent_initializer("test_sparse_builder", prescaled=True)
    def builder(n, *, dtype=None, device=None):
        calls.append(n)
        if source_kind == "scipy":
            return sp.eye(n, format="csr")
        return torch.sparse_csr_tensor(
            torch.arange(n + 1, device=device),
            torch.arange(n, device=device),
            torch.ones(n, dtype=dtype, device=device),
            size=(n, n),
        )

    empty = torch.empty

    def bounded_empty(*args, **kwargs):
        assert args[:2] != (96, 96)
        assert not (args and args[0] == (96, 96))
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", bounded_empty)
    cell = _cell(reservoir_size=96, topology="test_sparse_builder")
    assert calls == [96]
    assert cell.weight_hh._nnz() == 96


def test_structured_specialization_tracks_live_value_and_valid_pattern_edits():
    cell = _cell(reservoir_size=8, topology=("band", {"bandwidth": 0}))
    assert cell._hh_single_columns is not None
    state = torch.randn(3, 8, dtype=torch.float64, requires_grad=True)
    weights = cell.weight_hh
    with torch.no_grad():
        weights.values().mul_(0.7)
    expected = torch.sparse.mm(weights, state.T.contiguous()).T
    actual = cell._recurrent(state)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Preserve a valid sorted CSR pattern while changing two rows' counts.
    with torch.no_grad():
        weights.crow_indices()[1] = 0
    expected = torch.sparse.mm(weights, state.T.contiguous()).T
    actual = cell._recurrent(state)
    assert cell._hh_single_columns is None
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "name", ["simple_cycle", "delay_line", "delay_line_backward", "signed_permutation"]
)
def test_named_structured_initializers_preserve_unscaled_dense_definition(name):
    dense = _cell(topology=name, sparse=False)
    compact = _cell(topology=name)
    torch.testing.assert_close(compact.weight_hh.to_dense(), dense.weight_hh, rtol=0, atol=0)
