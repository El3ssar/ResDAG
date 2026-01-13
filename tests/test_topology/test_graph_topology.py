"""Tests for graph topology initialization."""

import pytest
import torch

from resdag.init.graphs import dendrocycle_graph, erdos_renyi_graph, ring_chord_graph
from resdag.init.topology import GraphTopology, get_topology, show_topologies


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

    def test_show_topologies_list(self):
        """Test listing available topologies."""
        topologies = show_topologies()

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


class TestDendrocycleRounding:
    """Tests for dendrocycle graph with rounding edge cases."""

    def test_dendrocycle_rounding_edge_case(self):
        """Test that dendrocycle handles rounding correctly when c+d≈1."""
        # Cases where rounding could cause C + D to exceed n
        # e.g., c=0.503, d=0.497 → C=201, D=199 → C+D=400 ✓
        # but c=0.5025, d=0.4975 could round differently
        topology = GraphTopology(dendrocycle_graph, {"c": 0.5025, "d": 0.4975, "seed": 42})
        weight = torch.empty(400, 400)

        # Should not raise ValueError about graph size mismatch
        result = topology.initialize(weight, spectral_radius=0.9)

        assert result.shape == (400, 400)

    def test_dendrocycle_various_parameter_combinations(self):
        """Test dendrocycle with various c,d combinations that could cause rounding issues."""
        n = 400
        test_cases = [
            (0.499, 0.499),  # Close to equal split
            (0.503, 0.495),  # Slightly uneven
            (0.5025, 0.4975),  # Very close to 1.0
            (0.501, 0.498),  # Another close case
            (0.333, 0.332),  # Thirds (rounding issues)
            (0.666, 0.333),  # Two thirds
        ]

        for c, d in test_cases:
            if c + d > 1.0:
                continue  # Skip invalid combinations

            topology = GraphTopology(dendrocycle_graph, {"c": c, "d": d, "seed": 42})
            weight = torch.empty(n, n)

            # Should create exactly n nodes
            result = topology.initialize(weight, spectral_radius=0.9)
            assert result.shape == (n, n), f"Failed for c={c}, d={d}"

    def test_dendrocycle_systematic_scan(self):
        """Systematically test dendrocycle across parameter space."""
        n = 400
        c_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        d_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

        for c in c_values:
            for d in d_values:
                if c + d > 1.0 or c + d < 0.0:
                    continue  # Skip invalid combinations

                topology = GraphTopology(dendrocycle_graph, {"c": c, "d": d, "seed": 42})
                weight = torch.empty(n, n)

                # Should create exactly n nodes regardless of rounding
                result = topology.initialize(weight, spectral_radius=0.9)
                assert result.shape == (n, n), f"Failed for c={c}, d={d}"
