"""Tests for graph topology initialization."""

import pytest
import torch

from torch_rc.init.graphs import erdos_renyi_graph, ring_chord_graph
from torch_rc.init.topology import GraphTopology, get_topology, list_topologies


class TestGraphTopology:
    """Tests for GraphTopology class."""

    def test_initialization_basic(self):
        """Test basic graph topology initialization."""
        topology = GraphTopology(erdos_renyi_graph, {"p": 0.1, "directed": True, "seed": 42})
        weight = torch.empty(50, 50)

        result = topology.initialize(weight)

        assert result is weight  # Should return the same tensor
        assert weight.shape == (50, 50)
        assert not torch.all(weight == 0)  # Should have been initialized

    def test_initialization_with_spectral_radius(self):
        """Test initialization with spectral radius scaling."""
        topology = GraphTopology(erdos_renyi_graph, {"p": 0.2, "directed": True, "seed": 42})
        weight = torch.empty(50, 50)
        target_radius = 0.9

        topology.initialize(weight, spectral_radius=target_radius)

        # Verify spectral radius is close to target
        eigenvalues = torch.linalg.eigvals(weight)
        actual_radius = torch.max(torch.abs(eigenvalues)).item()

        assert abs(actual_radius - target_radius) < 0.01

    def test_non_square_weight_raises_error(self):
        """Test that non-square weights raise ValueError."""
        topology = GraphTopology(erdos_renyi_graph, {"p": 0.1})
        weight = torch.empty(50, 100)

        with pytest.raises(ValueError, match="must be square"):
            topology.initialize(weight)

    def test_different_graph_functions(self):
        """Test with different graph functions."""
        topologies = [
            GraphTopology(erdos_renyi_graph, {"p": 0.1, "directed": True, "seed": 42}),
            GraphTopology(ring_chord_graph, {"L": 1, "w": 0.5, "alpha": 1.0}),
        ]

        for topology in topologies:
            weight = torch.empty(30, 30)
            result = topology.initialize(weight, spectral_radius=0.9)

            assert result.shape == (30, 30)
            assert not torch.all(result == 0)


class TestTopologyRegistry:
    """Tests for topology registry."""

    def test_list_topologies(self):
        """Test listing available topologies."""
        topologies = list_topologies()

        assert isinstance(topologies, list)
        assert len(topologies) > 0
        assert "erdos_renyi" in topologies
        assert "watts_strogatz" in topologies

    def test_get_topology_by_name(self):
        """Test getting topology by name."""
        topology = get_topology("erdos_renyi", p=0.15, seed=42)

        assert isinstance(topology, GraphTopology)
        assert topology.graph_kwargs["p"] == 0.15
        assert topology.graph_kwargs["seed"] == 42

    def test_get_topology_unknown_raises_error(self):
        """Test that unknown topology name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown topology"):
            get_topology("nonexistent_topology")

    def test_get_topology_with_defaults(self):
        """Test getting topology with default parameters."""
        topology = get_topology("erdos_renyi")
        weight = torch.empty(40, 40)

        result = topology.initialize(weight, spectral_radius=0.95)

        assert result.shape == (40, 40)

    def test_get_topology_override_defaults(self):
        """Test overriding default parameters."""
        topology = get_topology("erdos_renyi", p=0.5, directed=False)

        assert topology.graph_kwargs["p"] == 0.5
        assert topology.graph_kwargs["directed"] is False


class TestGraphTopologyEdgeCases:
    """Edge case tests for graph topology."""

    def test_very_small_graph(self):
        """Test with very small graphs."""
        topology = GraphTopology(erdos_renyi_graph, {"p": 0.5, "directed": True, "seed": 42})
        weight = torch.empty(3, 3)

        topology.initialize(weight)

        assert weight.shape == (3, 3)

    def test_gpu_tensor(self):
        """Test initialization on GPU tensor if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        topology = GraphTopology(erdos_renyi_graph, {"p": 0.1, "directed": True, "seed": 42})
        weight = torch.empty(50, 50, device="cuda")

        result = topology.initialize(weight, spectral_radius=0.9)

        assert result.device.type == "cuda"
        assert result.shape == (50, 50)

    def test_different_dtypes(self):
        """Test with different tensor dtypes."""
        topology = GraphTopology(erdos_renyi_graph, {"p": 0.1, "directed": True, "seed": 42})

        for dtype in [torch.float32, torch.float64]:
            weight = torch.empty(30, 30, dtype=dtype)
            result = topology.initialize(weight)

            assert result.dtype == dtype
