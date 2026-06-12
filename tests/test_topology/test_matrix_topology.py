"""Tests for the generalized init system: matrix builders + bare callables.

Covers MatrixTopology, register_matrix_topology, FunctionInitializer,
callable specs in the resolvers, and end-to-end use through ESNLayer.
"""

import networkx as nx
import numpy as np
import pytest
import torch

from resdag.init.input_feedback import (
    FunctionInitializer,
    get_input_feedback,
    register_input_feedback,
)
from resdag.init.topology import (
    GraphTopology,
    MatrixTopology,
    get_topology,
    register_matrix_topology,
)
from resdag.init.utils import resolve_initializer, resolve_topology
from resdag.layers import ESNLayer


def block_diagonal(n: int, blocks: int = 4) -> torch.Tensor:
    w = torch.zeros(n, n)
    size = n // blocks
    for b in range(blocks):
        s = b * size
        w[s : s + size, s : s + size] = torch.randn(size, size)
    return w


class TestMatrixTopology:
    def test_build_style_tensor(self) -> None:
        topology = MatrixTopology(block_diagonal, {"blocks": 2})
        weight = torch.empty(20, 20)
        topology.initialize(weight)

        # Off-diagonal blocks must be exactly zero
        assert torch.all(weight[:10, 10:] == 0)
        assert torch.all(weight[10:, :10] == 0)
        assert not torch.all(weight[:10, :10] == 0)

    def test_build_style_numpy(self) -> None:
        topology = MatrixTopology(lambda n: np.eye(n, dtype=np.float64))
        weight = torch.empty(8, 8)
        topology.initialize(weight)

        assert torch.allclose(weight, torch.eye(8))
        assert weight.dtype == torch.float32

    def test_build_style_graph_return(self) -> None:
        """A callable returning a NetworkX graph is coerced to its adjacency."""
        topology = MatrixTopology(lambda n: nx.cycle_graph(n))
        weight = torch.empty(6, 6)
        topology.initialize(weight)

        # Cycle graph adjacency: each node connects to two neighbours
        assert torch.allclose(weight.sum(dim=0), torch.full((6,), 2.0))

    def test_inplace_style_torch_nn_init(self) -> None:
        topology = MatrixTopology(torch.nn.init.orthogonal_)
        weight = torch.empty(16, 16)
        topology.initialize(weight)

        assert torch.allclose(weight @ weight.T, torch.eye(16), atol=1e-5)

    def test_spectral_radius_scaling(self) -> None:
        topology = MatrixTopology(lambda n: torch.randn(n, n))
        weight = torch.empty(32, 32)
        topology.initialize(weight, spectral_radius=0.7)

        radius = torch.max(torch.abs(torch.linalg.eigvals(weight))).item()
        assert radius == pytest.approx(0.7, abs=1e-4)

    def test_wrong_shape_raises(self) -> None:
        topology = MatrixTopology(lambda n: torch.randn(n, n + 1))
        with pytest.raises(ValueError, match="shape"):
            topology.initialize(torch.empty(10, 10))

    def test_invalid_return_type_raises(self) -> None:
        topology = MatrixTopology(lambda n: "not a matrix")
        with pytest.raises(ValueError, match="cannot"):
            topology.initialize(torch.empty(4, 4))

    def test_unusable_callable_raises_informative(self) -> None:
        def needs_three_positional(a, b, c):
            return a + b + c

        topology = MatrixTopology(needs_three_positional)
        with pytest.raises(ValueError, match="neither initializer convention"):
            topology.initialize(torch.empty(4, 4))

    def test_non_square_weight_raises(self) -> None:
        topology = MatrixTopology(block_diagonal)
        with pytest.raises(ValueError, match="square"):
            topology.initialize(torch.empty(4, 6))


class TestMatrixRegistry:
    def test_register_and_get(self) -> None:
        @register_matrix_topology("_test_blockdiag", blocks=2)
        def _blockdiag(n: int, blocks: int = 2) -> torch.Tensor:
            return block_diagonal(n, blocks)

        topology = get_topology("_test_blockdiag")
        assert isinstance(topology, MatrixTopology)

        weight = torch.empty(10, 10)
        topology.initialize(weight)
        assert torch.all(weight[:5, 5:] == 0)

    def test_builtin_orthogonal(self) -> None:
        topology = get_topology("orthogonal", seed=42)
        assert isinstance(topology, MatrixTopology)

        weight = torch.empty(24, 24)
        topology.initialize(weight)
        assert torch.allclose(weight @ weight.T, torch.eye(24), atol=1e-5)

    def test_orthogonal_seed_reproducible(self) -> None:
        w1, w2 = torch.empty(12, 12), torch.empty(12, 12)
        get_topology("orthogonal", seed=7).initialize(w1)
        get_topology("orthogonal", seed=7).initialize(w2)
        assert torch.equal(w1, w2)

    def test_duplicate_name_raises(self) -> None:
        with pytest.raises(ValueError, match="already registered"):
            register_matrix_topology("orthogonal")(lambda n: torch.eye(n))

    def test_graph_topologies_unaffected(self) -> None:
        topology = get_topology("erdos_renyi", p=0.2, seed=1)
        assert isinstance(topology, GraphTopology)


class TestResolverCallables:
    def test_bare_callable_topology(self) -> None:
        resolved = resolve_topology(block_diagonal)
        assert isinstance(resolved, MatrixTopology)

    def test_callable_tuple_topology(self) -> None:
        resolved = resolve_topology((block_diagonal, {"blocks": 5}))
        assert isinstance(resolved, MatrixTopology)
        assert resolved.matrix_kwargs == {"blocks": 5}

    def test_bare_callable_initializer(self) -> None:
        resolved = resolve_initializer(torch.nn.init.xavier_uniform_)
        assert isinstance(resolved, FunctionInitializer)

    def test_callable_tuple_initializer(self) -> None:
        def scaled(rows, cols, scale=1.0):
            return torch.full((rows, cols), scale)

        resolved = resolve_initializer((scaled, {"scale": 0.25}))
        weight = torch.empty(6, 3)
        resolved.initialize(weight)
        assert torch.all(weight == 0.25)


class TestFunctionInitializer:
    def test_build_style(self) -> None:
        def first_neuron_only(rows, cols, scale=1.0):
            w = torch.zeros(rows, cols)
            w[0, :] = scale
            return w

        init = FunctionInitializer(first_neuron_only, scale=0.5)
        weight = torch.empty(10, 3)
        init.initialize(weight)

        assert torch.all(weight[0] == 0.5)
        assert torch.all(weight[1:] == 0)

    def test_inplace_style(self) -> None:
        init = FunctionInitializer(torch.nn.init.xavier_uniform_)
        weight = torch.empty(50, 5)
        init.initialize(weight)

        assert not torch.all(weight == 0)
        assert weight.abs().max() < 1.0  # xavier bound for this shape

    def test_registered_function(self) -> None:
        @register_input_feedback("_test_constant", value=2.0)
        def constant(rows, cols, value=1.0):
            return torch.full((rows, cols), value)

        init = get_input_feedback("_test_constant")
        weight = torch.empty(4, 2)
        init.initialize(weight)
        assert torch.all(weight == 2.0)

        init = get_input_feedback("_test_constant", value=-1.0)
        init.initialize(weight)
        assert torch.all(weight == -1.0)


class TestESNLayerIntegration:
    def test_layer_with_callable_topology(self) -> None:
        layer = ESNLayer(
            reservoir_size=20,
            feedback_size=3,
            topology=(block_diagonal, {"blocks": 2}),
            spectral_radius=0.9,
        )

        assert torch.all(layer.weight_hh[:10, 10:] == 0)
        radius = torch.max(torch.abs(torch.linalg.eigvals(layer.weight_hh))).item()
        assert radius == pytest.approx(0.9, abs=1e-4)

    def test_layer_with_torch_init_everywhere(self) -> None:
        layer = ESNLayer(
            reservoir_size=16,
            feedback_size=3,
            topology=torch.nn.init.orthogonal_,
            feedback_initializer=torch.nn.init.xavier_uniform_,
        )

        out = layer(torch.randn(2, 10, 3))
        assert out.shape == (2, 10, 16)

    def test_layer_with_named_matrix_topology(self) -> None:
        layer = ESNLayer(
            reservoir_size=16,
            feedback_size=3,
            topology=("orthogonal", {"seed": 3}),
        )
        w = layer.weight_hh.data
        assert torch.allclose(w @ w.T, torch.eye(16), atol=1e-5)
