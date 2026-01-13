"""Example: Using input/feedback weight initializers.

This example demonstrates the various input/feedback initializers available
for customizing how inputs connect to reservoir neurons.
"""

import torch

from resdag.init.input_feedback import (
    BinaryBalancedInitializer,
    ChebyshevInitializer,
    ChessboardInitializer,
    PseudoDiagonalInitializer,
    RandomBinaryInitializer,
    RandomInputInitializer,
)
from resdag.layers import ReservoirLayer

print("=" * 70)
print("Example 1: Random Uniform Initializer (Baseline)")
print("=" * 70)

feedback_init = RandomInputInitializer(input_scaling=1.0, seed=42)
reservoir = ReservoirLayer(reservoir_size=100, feedback_size=10, feedback_initializer=feedback_init)

print(f"Feedback weight shape: {reservoir.weight_feedback.shape}")
print(
    f"Value range: [{reservoir.weight_feedback.min():.3f}, {reservoir.weight_feedback.max():.3f}]"
)
print(f"Mean: {reservoir.weight_feedback.mean():.3f}")
print(f"Std: {reservoir.weight_feedback.std():.3f}")

# Test forward pass
feedback = torch.randn(4, 20, 10)
output = reservoir(feedback)
print(f"Output shape: {output.shape}")

print("\n" + "=" * 70)
print("Example 2: Binary Initializer")
print("=" * 70)

binary_init = RandomBinaryInitializer(input_scaling=0.5, seed=42)
reservoir2 = ReservoirLayer(reservoir_size=100, feedback_size=10, feedback_initializer=binary_init)

print(f"Unique values: {torch.unique(reservoir2.weight_feedback).tolist()}")
print(f"Number of +0.5: {(reservoir2.weight_feedback == 0.5).sum().item()}")
print(f"Number of -0.5: {(reservoir2.weight_feedback == -0.5).sum().item()}")

print("\n" + "=" * 70)
print("Example 3: Chebyshev (Deterministic Chaotic)")
print("=" * 70)

chebyshev_init = ChebyshevInitializer(
    p=0.3,
    q=5.9,
    k=3.5,  # Chaotic regime
    input_scaling=0.8,
)
reservoir3 = ReservoirLayer(
    reservoir_size=100, feedback_size=10, feedback_initializer=chebyshev_init
)

print(
    f"Value range: [{reservoir3.weight_feedback.min():.3f}, {reservoir3.weight_feedback.max():.3f}]"
)
print(f"First column (sinusoidal): {reservoir3.weight_feedback[:5, 0].tolist()}")
print("Deterministic: No randomness, fully reproducible")

print("\n" + "=" * 70)
print("Example 4: Binary Balanced (Hadamard-based)")
print("=" * 70)

balanced_init = BinaryBalancedInitializer(input_scaling=1.0, balance_global=True)
reservoir4 = ReservoirLayer(
    reservoir_size=100, feedback_size=10, feedback_initializer=balanced_init
)

col_sums = reservoir4.weight_feedback.sum(dim=0)
print(f"Column sums (should be near 0): {col_sums.tolist()}")
print(f"Column sums range: [{col_sums.min():.3f}, {col_sums.max():.3f}]")
print(f"Unique values: {torch.unique(reservoir4.weight_feedback).tolist()}")

print("\n" + "=" * 70)
print("Example 5: Pseudo-Diagonal (Structured)")
print("=" * 70)

pseudo_init = PseudoDiagonalInitializer(input_scaling=1.0, binarize=False, seed=42)
reservoir5 = ReservoirLayer(reservoir_size=100, feedback_size=5, feedback_initializer=pseudo_init)

nonzero = (reservoir5.weight_feedback.abs() > 1e-8).sum().item()
total = reservoir5.weight_feedback.numel()
print(f"Non-zero elements: {nonzero} / {total} ({100 * nonzero / total:.1f}%)")
print("Structure: Each input connects to contiguous block of neurons")

# Visualize structure (show which neurons each input connects to)
for i in range(5):
    connected = (reservoir5.weight_feedback[:, i].abs() > 1e-8).nonzero().squeeze()
    print(f"  Input {i} → neurons {connected[0].item()} to {connected[-1].item()}")

print("\n" + "=" * 70)
print("Example 6: Chessboard (Deterministic Pattern)")
print("=" * 70)

chess_init = ChessboardInitializer(input_scaling=0.5)
reservoir6 = ReservoirLayer(reservoir_size=10, feedback_size=10, feedback_initializer=chess_init)

print("First 5x5 block of weight matrix:")
print(reservoir6.weight_feedback[:5, :5])
print("Pattern: Alternating +/-0.5 in checkerboard pattern")

print("\n" + "=" * 70)
print("Example 7: Multiple Initializers (Feedback + Input)")
print("=" * 70)

feedback_init = ChebyshevInitializer(p=0.3, k=3.5, input_scaling=0.5)
input_init = BinaryBalancedInitializer(input_scaling=1.0)

reservoir7 = ReservoirLayer(
    reservoir_size=200,
    feedback_size=10,
    input_size=5,
    feedback_initializer=feedback_init,
    input_initializer=input_init,
    topology="erdos_renyi",
    spectral_radius=0.9,
)

print("Feedback weights (Chebyshev):")
print(f"  Shape: {reservoir7.weight_feedback.shape}")
print(f"  Range: [{reservoir7.weight_feedback.min():.3f}, {reservoir7.weight_feedback.max():.3f}]")

print("\nInput weights (Binary Balanced):")
print(f"  Shape: {reservoir7.weight_input.shape}")
print(f"  Unique values: {torch.unique(reservoir7.weight_input).tolist()}")
print(f"  Column sums: {reservoir7.weight_input.sum(dim=0).tolist()}")

print("\nRecurrent weights (Erdős-Rényi topology):")
print(f"  Shape: {reservoir7.weight_hh.shape}")
print(f"  Density: {(reservoir7.weight_hh.abs() > 1e-8).float().mean():.3f}")

# Forward pass with both inputs
feedback = torch.randn(2, 30, 10)
driving = torch.randn(2, 30, 5)
output = reservoir7(feedback, driving)
print(f"\nForward pass successful: {output.shape}")

print("\n" + "=" * 70)
print("Example 8: Comparison of Initializers")
print("=" * 70)

initializers = {
    "RandomInput": RandomInputInitializer(input_scaling=1.0, seed=42),
    "RandomBinary": RandomBinaryInitializer(input_scaling=1.0, seed=42),
    "Chebyshev": ChebyshevInitializer(p=0.3, k=3.5, input_scaling=1.0),
    "BinaryBalanced": BinaryBalancedInitializer(input_scaling=1.0),
    "PseudoDiagonal": PseudoDiagonalInitializer(input_scaling=1.0, seed=42),
    "Chessboard": ChessboardInitializer(input_scaling=1.0),
}

print(
    f"{'Initializer':<20} | {'Mean':>8} | {'Std':>8} | {'Min':>8} | {'Max':>8} | {'Sparsity':>10}"
)
print("-" * 80)

for name, init in initializers.items():
    weight = torch.empty(100, 10)
    init.initialize(weight)

    mean = weight.mean().item()
    std = weight.std().item()
    min_val = weight.min().item()
    max_val = weight.max().item()
    sparsity = (weight.abs() < 1e-8).float().mean().item()

    print(
        f"{name:<20} | {mean:8.3f} | {std:8.3f} | {min_val:8.3f} | {max_val:8.3f} | {sparsity:10.3f}"
    )

print("\n✅ All examples completed successfully!")
