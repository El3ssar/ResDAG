"""Example: Using the Registry System for Topologies and Initializers.

This example demonstrates:
1. Using pre-registered graph topologies
2. Using pre-registered input/feedback initializers
3. Creating and registering custom graph topologies
4. Creating and registering custom input/feedback initializers
5. Listing available options
"""

import networkx as nx
import torch

from resdag.init.input_feedback import (
    InputFeedbackInitializer,
    get_input_feedback,
    register_input_feedback,
    show_input_initializers,
)
from resdag.init.topology import (
    get_topology,
    register_graph_topology,
    show_topologies,
)


def main():
    print("=" * 80)
    print("REGISTRY SYSTEM EXAMPLES")
    print("=" * 80)

    # =========================================================================
    # Part 1: Using Pre-Registered Graph Topologies
    # =========================================================================
    print("\n[1] PRE-REGISTERED GRAPH TOPOLOGIES")
    print("-" * 80)

    # List all available topologies
    print(f"Available topologies: {', '.join(show_topologies())}")

    # Get a topology by name
    topology = get_topology("erdos_renyi", p=0.15, seed=42)
    print(f"\nCreated topology: {topology}")

    # Initialize a weight matrix
    weight = torch.empty(100, 100)
    topology.initialize(weight, spectral_radius=0.9)
    print(
        f"Initialized weight matrix: shape={weight.shape}, "
        f"nonzero={weight.count_nonzero().item()}/{weight.numel()}"
    )

    # =========================================================================
    # Part 2: Using Pre-Registered Input/Feedback Initializers
    # =========================================================================
    print("\n[2] PRE-REGISTERED INPUT/FEEDBACK INITIALIZERS")
    print("-" * 80)

    # List all available initializers
    print(f"Available initializers: {', '.join(show_input_initializers())}")

    # Get an initializer by name
    input_init = get_input_feedback("binary_balanced", input_scaling=0.5)
    print(f"\nCreated initializer: {input_init}")

    # Initialize an input weight matrix (rectangular)
    input_weight = torch.empty(100, 10)  # (reservoir_size, input_dim)
    input_init.initialize(input_weight)
    print(
        f"Initialized input weight: shape={input_weight.shape}, "
        f"mean={input_weight.mean():.4f}, std={input_weight.std():.4f}"
    )

    # =========================================================================
    # Part 3: Registering Custom Graph Topologies (Decorator Style)
    # =========================================================================
    print("\n[3] CUSTOM GRAPH TOPOLOGY (DECORATOR)")
    print("-" * 80)

    # Define and register a custom graph topology using decorator
    @register_graph_topology("custom_grid", k=2, p_rewire=0.1)
    def custom_grid_graph(n, k=2, p_rewire=0.1, seed=None):
        """Create a grid graph with optional rewiring.

        Parameters
        ----------
        n : int
            Number of nodes (will be rounded to nearest perfect square)
        k : int
            Connection distance in grid
        p_rewire : float
            Probability of rewiring an edge
        seed : int, optional
            Random seed
        """
        import math

        # Round to nearest square
        side = int(math.sqrt(n))
        actual_n = side * side

        # Create grid graph
        G = nx.grid_2d_graph(side, side, create_using=nx.DiGraph)

        # Map 2D coordinates to sequential node IDs
        mapping = {(i, j): i * side + j for i in range(side) for j in range(side)}
        G = nx.relabel_nodes(G, mapping)

        # Add weights to edges
        for u, v in G.edges():
            G[u][v]["weight"] = 1.0

        # Optionally rewire some edges
        if p_rewire > 0 and seed is not None:
            rng = torch.Generator().manual_seed(seed)
            edges = list(G.edges())
            for u, v in edges:
                if torch.rand(1, generator=rng).item() < p_rewire:
                    # Remove old edge and add new random edge
                    G.remove_edge(u, v)
                    new_target = torch.randint(0, actual_n, (1,), generator=rng).item()
                    if not G.has_edge(u, new_target):
                        G.add_edge(u, new_target, weight=1.0)

        # Ensure we have exactly n nodes (pad if needed)
        if actual_n < n:
            for i in range(actual_n, n):
                G.add_node(i)

        return G

    # Use the custom topology
    custom_topology = get_topology("custom_grid", k=3, p_rewire=0.2, seed=123)
    print(f"Created custom topology: {custom_topology}")

    weight = torch.empty(100, 100)
    custom_topology.initialize(weight, spectral_radius=0.95)
    print(
        f"Initialized with custom topology: shape={weight.shape}, "
        f"nonzero={weight.count_nonzero().item()}/{weight.numel()}"
    )

    # =========================================================================
    # Part 4: Custom Input/Feedback Initializers (Decorator Style)
    # =========================================================================
    print("\n[4] CUSTOM INPUT/FEEDBACK INITIALIZER (DECORATOR)")
    print("-" * 80)

    # Define and register a custom initializer using decorator
    @register_input_feedback("gaussian_sparse", sparsity=0.9, std=0.5)
    class GaussianSparseInitializer(InputFeedbackInitializer):
        """Initialize weights with sparse Gaussian values.

        Parameters
        ----------
        sparsity : float
            Fraction of weights to set to zero (0 = dense, 1 = all zeros)
        std : float
            Standard deviation of Gaussian distribution
        seed : int, optional
            Random seed
        """

        def __init__(self, sparsity=0.9, std=0.5, seed=None):
            self.sparsity = sparsity
            self.std = std
            self.seed = seed

        def initialize(self, weight, **kwargs):
            """Initialize weight tensor with sparse Gaussian values."""
            if self.seed is not None:
                torch.manual_seed(self.seed)

            # Generate Gaussian values
            weight.normal_(mean=0.0, std=self.std)

            # Apply sparsity mask
            if self.sparsity > 0:
                mask = torch.rand(weight.shape, device=weight.device) > self.sparsity
                weight.mul_(mask)

            return weight

        def __repr__(self):
            return f"GaussianSparseInitializer(sparsity={self.sparsity}, std={self.std})"

    # Use the custom initializer
    custom_init = get_input_feedback("gaussian_sparse", sparsity=0.8, std=0.3, seed=42)
    print(f"Created custom initializer: {custom_init}")

    input_weight = torch.empty(100, 5)
    custom_init.initialize(input_weight)
    print(
        f"Initialized with custom initializer: shape={input_weight.shape}, "
        f"nonzero={input_weight.count_nonzero().item()}/{input_weight.numel()}, "
        f"mean={input_weight.mean():.4f}, std={input_weight.std():.4f}"
    )

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n[7] SUMMARY")
    print("-" * 80)
    print(f"Total registered topologies: {len(show_topologies())}")
    print(f"Total registered initializers: {len(show_input_initializers())}")
    print("\nNew topologies:", [t for t in show_topologies() if t in ["custom_grid", "star"]])
    print(
        "New initializers:",
        [i for i in show_input_initializers() if i in ["gaussian_sparse", "triangular"]],
    )

    print("\n" + "=" * 80)
    print("REGISTRY SYSTEM EXAMPLES COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
