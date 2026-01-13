"""Tests for ReservoirLayer with graph topology initialization."""

import pytest
import torch

from resdag.init.graphs import erdos_renyi_graph
from resdag.init.topology import GraphTopology, get_topology
from resdag.layers import ReservoirLayer


class TestReservoirLayerTopology:
    """Tests for ReservoirLayer topology initialization."""

    def test_reservoir_with_string_topology(self):
        """Test reservoir initialization with string topology name."""
        reservoir = ReservoirLayer(
            reservoir_size=50,
            feedback_size=10,
            topology="erdos_renyi",
            spectral_radius=0.9,
        )

        # Should have initialized weight matrices
        assert reservoir.weight_feedback.shape == (50, 10)
        assert reservoir.weight_hh.shape == (50, 50)
        assert reservoir.weight_input is None  # No driving inputs

        # Check spectral radius is approximately correct
        eigenvalues = torch.linalg.eigvals(reservoir.weight_hh.data)
        actual_radius = torch.max(torch.abs(eigenvalues)).item()
        assert abs(actual_radius - 0.9) < 0.05

    def test_reservoir_with_topology_object(self):
        """Test reservoir with TopologyInitializer object."""
        topology = get_topology("watts_strogatz", k=4, p=0.1, seed=42)

        reservoir = ReservoirLayer(
            reservoir_size=40,
            feedback_size=5,
            topology=topology,
            spectral_radius=0.95,
        )

        assert reservoir.weight_hh.shape == (40, 40)

        # Verify spectral radius
        eigenvalues = torch.linalg.eigvals(reservoir.weight_hh.data)
        actual_radius = torch.max(torch.abs(eigenvalues)).item()
        assert abs(actual_radius - 0.95) < 0.05

    def test_reservoir_with_custom_graph_topology(self):
        """Test reservoir with custom GraphTopology."""
        topology = GraphTopology(erdos_renyi_graph, {"p": 0.15, "directed": True, "seed": 42})

        reservoir = ReservoirLayer(
            reservoir_size=30,
            feedback_size=8,
            topology=topology,
        )

        assert reservoir.weight_hh.shape == (30, 30)
        assert not torch.all(reservoir.weight_hh == 0)

    def test_reservoir_topology_forward_pass(self):
        """Test that reservoir with topology works in forward pass."""
        reservoir = ReservoirLayer(
            reservoir_size=50,
            feedback_size=10,
            topology="erdos_renyi",
            spectral_radius=0.9,
        )

        # Create input
        feedback = torch.randn(4, 20, 10)  # (batch=4, time=20, features=10)

        # Forward pass
        output = reservoir(feedback)

        assert output.shape == (4, 20, 50)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_reservoir_topology_with_driving_inputs(self):
        """Test reservoir with topology and driving inputs."""
        reservoir = ReservoirLayer(
            reservoir_size=60,
            feedback_size=5,
            input_size=3,
            topology="ring_chord",
            spectral_radius=0.85,
        )

        feedback = torch.randn(2, 15, 5)
        driving = torch.randn(2, 15, 3)

        output = reservoir(feedback, driving)

        assert output.shape == (2, 15, 60)
        assert reservoir.weight_input.shape == (60, 3)

    def test_different_topologies_produce_different_weights(self):
        """Test that different topologies produce different weight matrices."""
        reservoir1 = ReservoirLayer(
            reservoir_size=30,
            feedback_size=5,
            topology="erdos_renyi",
        )

        reservoir2 = ReservoirLayer(
            reservoir_size=30,
            feedback_size=5,
            topology="watts_strogatz",
        )

        # Weights should be different (very unlikely to be identical by chance)
        assert not torch.allclose(reservoir1.weight_hh, reservoir2.weight_hh)

    def test_reservoir_topology_invalid_type(self):
        """Test that invalid topology type raises error."""
        with pytest.raises(TypeError, match="topology must be a string or TopologyInitializer"):
            ReservoirLayer(
                reservoir_size=30,
                feedback_size=5,
                topology=123,  # Invalid type
            )

    def test_reservoir_topology_state_persistence(self):
        """Test that topology-based reservoir maintains state correctly."""
        reservoir = ReservoirLayer(
            reservoir_size=40,
            feedback_size=10,
            topology="erdos_renyi",
        )

        feedback1 = torch.randn(2, 10, 10)
        feedback2 = torch.randn(2, 10, 10)

        # First forward pass
        out1 = reservoir(feedback1)

        # Second forward pass (state should carry over)
        out2 = reservoir(feedback2)

        # Reset and run again
        reservoir.reset_state()
        out3 = reservoir(feedback1)

        # out3 should match out1 (same input, fresh state)
        # but out2 should be different (carried state)
        assert torch.allclose(out1, out3, rtol=1e-5)
        assert not torch.allclose(out2, out3, rtol=1e-5)

    def test_reservoir_topology_reproducibility(self):
        """Test that topology with seed produces reproducible results."""
        topology1 = GraphTopology(erdos_renyi_graph, {"p": 0.1, "directed": True, "seed": 42})
        topology2 = GraphTopology(erdos_renyi_graph, {"p": 0.1, "directed": True, "seed": 42})

        reservoir1 = ReservoirLayer(
            reservoir_size=30,
            feedback_size=5,
            topology=topology1,
        )

        reservoir2 = ReservoirLayer(
            reservoir_size=30,
            feedback_size=5,
            topology=topology2,
        )

        # Same seed should produce identical weights
        assert torch.allclose(reservoir1.weight_hh, reservoir2.weight_hh)

    def test_reservoir_topology_various_sizes(self):
        """Test topology initialization with various reservoir sizes."""
        sizes = [50, 100, 200]

        for size in sizes:
            reservoir = ReservoirLayer(
                reservoir_size=size,
                feedback_size=5,
                topology="erdos_renyi",
                spectral_radius=0.9,
            )

            assert reservoir.weight_hh.shape == (size, size)

            # Verify spectral radius
            eigenvalues = torch.linalg.eigvals(reservoir.weight_hh.data)
            actual_radius = torch.max(torch.abs(eigenvalues)).item()
            assert abs(actual_radius - 0.9) < 0.1
