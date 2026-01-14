"""Example: Using ReservoirLayer with graph topologies.

This example demonstrates how to use different graph topologies
to initialize the recurrent weight matrix of a ReservoirLayer.
"""

import torch

from resdag.init.graphs import erdos_renyi_graph
from resdag.init.topology import GraphTopology, show_topologies
from resdag.layers import ReservoirLayer

# Example 1: Using a pre-registered topology by name
print("=" * 60)
print("Example 1: Using a named topology")
print("=" * 60)

reservoir = ReservoirLayer(
    reservoir_size=100,
    feedback_size=10,
    topology="erdos_renyi",  # Use Erdős-Rényi random graph
    spectral_radius=0.9,
)

# Create some dummy input
feedback = torch.randn(4, 50, 10)  # (batch=4, time=50, features=10)

# Forward pass
output = reservoir(feedback)
print(f"Input shape:  {feedback.shape}")
print(f"Output shape: {output.shape}")
print(f"Recurrent weight matrix shape: {reservoir.weight_hh.shape}")

# Verify spectral radius
eigenvalues = torch.linalg.eigvals(reservoir.weight_hh.data)
actual_radius = torch.max(torch.abs(eigenvalues)).item()
print(f"Target spectral radius: {reservoir.spectral_radius}")
print(f"Actual spectral radius: {actual_radius:.4f}")

# Example 2: Using a topology with custom parameters
print("\n" + "=" * 60)
print("Example 2: Custom topology parameters")
print("=" * 60)

# Get available topologies
print(f"Available topologies: {show_topologies()}")

# Use Watts-Strogatz small-world topology with custom parameters
reservoir2 = ReservoirLayer(
    reservoir_size=200,
    feedback_size=5,
    input_size=3,  # Also have driving inputs
    topology=("watts_strogatz", {"k": 6, "p": 0.1, "seed": 42}),
    spectral_radius=0.95,
)

feedback2 = torch.randn(2, 30, 5)
driving2 = torch.randn(2, 30, 3)

output2 = reservoir2(feedback2, driving2)
print("Reservoir with topology='watts_strogatz' created")
print(f"Output shape: {output2.shape}")

# Example 3: Creating a custom topology
print("\n" + "=" * 60)
print("Example 3: Custom GraphTopology")
print("=" * 60)

# Create a custom topology with direct control
custom_topology = GraphTopology(
    erdos_renyi_graph, {"p": 0.15, "directed": True, "self_loops": True, "seed": 123}
)

reservoir3 = ReservoirLayer(
    reservoir_size=150,
    feedback_size=8,
    topology=custom_topology,
    spectral_radius=0.85,
)

feedback3 = torch.randn(3, 40, 8)
output3 = reservoir3(feedback3)
print("Custom topology reservoir created")
print(f"Output shape: {output3.shape}")

# Example 4: Comparing different topologies
print("\n" + "=" * 60)
print("Example 4: Comparing topology effects")
print("=" * 60)

topologies_to_compare = ["erdos_renyi", "watts_strogatz", "ring_chord", "regular"]

for topo_name in topologies_to_compare:
    res = ReservoirLayer(
        reservoir_size=50,
        feedback_size=5,
        topology=topo_name,
        spectral_radius=0.9,
    )

    # Count non-zero edges
    num_edges = (res.weight_hh.abs() > 1e-8).sum().item()
    density = num_edges / (50 * 50)

    eigenvalues = torch.linalg.eigvals(res.weight_hh.data)
    actual_sr = torch.max(torch.abs(eigenvalues)).item()

    print(f"{topo_name:20s} | Edges: {num_edges:4d} | Density: {density:.3f} | SR: {actual_sr:.4f}")

print("\n" + "=" * 60)
print("Example 5: Stateful processing with topology")
print("=" * 60)

# Topology-based reservoir maintains state across calls
reservoir4 = ReservoirLayer(
    reservoir_size=80,
    feedback_size=10,
    topology="erdos_renyi",
    spectral_radius=0.9,
)

# Process multiple sequences (state carries over)
seq1 = torch.randn(1, 20, 10)
seq2 = torch.randn(1, 20, 10)

out1 = reservoir4(seq1)
print(f"After sequence 1, state shape: {reservoir4.state.shape}")

out2 = reservoir4(seq2)
print(f"After sequence 2, state shape: {reservoir4.state.shape}")

# Reset and process again
reservoir4.reset_state()
out3 = reservoir4(seq1)
print(
    f"After reset + sequence 1, output matches first run: {torch.allclose(out1, out3, rtol=1e-5)}"
)

print("\nAll examples completed successfully!")
