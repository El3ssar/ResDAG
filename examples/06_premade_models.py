"""Example usage of premade model architectures.

This script demonstrates the four premade ESN architectures:
- classic_esn: Traditional ESN with input concatenation
- ott_esn: Ott's ESN with state augmentation
- headless_esn: Reservoir only (no readout)
- linear_esn: Linear reservoir for baseline comparison

All models now use pytorch_symbolic for cleaner, more Pythonic API.
"""

import torch

from resdag.models import classic_esn, headless_esn, linear_esn, ott_esn


def example_classic_esn():
    """Example: Classic ESN for time series prediction."""
    print("\n" + "=" * 60)
    print("Example 1: Classic ESN")
    print("=" * 60)

    # Simple usage with defaults
    model = classic_esn(reservoir_size=100, feedback_size=1, output_size=1)

    # Generate some dummy data
    x = torch.randn(4, 50, 1)  # (batch, time, features)

    # Forward pass (direct, no dict needed!)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # With custom configuration using flat params
    model_custom = classic_esn(
        reservoir_size=100,
        feedback_size=1,
        output_size=1,
        # Topology as tuple: (name, params)
        topology=("erdos_renyi", {"p": 0.15}),
        spectral_radius=0.9,
        leak_rate=0.3,
        # Initializer as string (uses defaults)
        feedback_initializer="pseudo_diagonal",
        # Readout params
        readout_alpha=1e-5,
    )

    output_custom = model_custom(x)
    print(f"Custom model output shape: {output_custom.shape}")

    # Show model summary
    print("\nModel summary:")
    model.summary()


def example_ott_esn():
    """Example: Ott's ESN with state augmentation."""
    print("\n" + "=" * 60)
    print("Example 2: Ott's ESN")
    print("=" * 60)

    # Ott's ESN squares even-indexed reservoir units
    model = ott_esn(reservoir_size=200, feedback_size=2, output_size=3)

    x = torch.randn(4, 50, 2)
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Show model layers
    print("\nModel layers:")
    for name, module in model.named_modules():
        if len(name) > 0:  # Skip the top-level module
            print(f"  {name}: {module.__class__.__name__}")


def example_headless_esn():
    """Example: Headless ESN for reservoir analysis."""
    print("\n" + "=" * 60)
    print("Example 3: Headless ESN (Reservoir Analysis)")
    print("=" * 60)

    # No readout layer - useful for analyzing reservoir dynamics
    model = headless_esn(reservoir_size=100, feedback_size=2)

    x = torch.randn(4, 50, 2)
    states = model(x)  # Returns reservoir states directly

    print(f"Input shape: {x.shape}")
    print(f"Reservoir states shape: {states.shape}")

    # Analyze reservoir dynamics
    print("State statistics:")
    print(f"  Mean: {states.mean():.4f}")
    print(f"  Std: {states.std():.4f}")
    print(f"  Min: {states.min():.4f}")
    print(f"  Max: {states.max():.4f}")


def example_linear_esn():
    """Example: Linear ESN for baseline comparison."""
    print("\n" + "=" * 60)
    print("Example 4: Linear ESN (Baseline)")
    print("=" * 60)

    # Linear activation for comparison with nonlinear reservoirs
    model_linear = linear_esn(reservoir_size=100, feedback_size=1)
    model_nonlinear = headless_esn(reservoir_size=100, feedback_size=1)

    x = torch.randn(4, 50, 1)

    # Reset states for fair comparison
    model_linear.reset_reservoirs()
    model_nonlinear.reset_reservoirs()

    states_linear = model_linear(x)
    states_nonlinear = model_nonlinear(x)

    print(f"Input shape: {x.shape}")
    print(f"Linear states shape: {states_linear.shape}")
    print(f"Nonlinear states shape: {states_nonlinear.shape}")

    # Compare dynamics
    print("\nLinear reservoir statistics:")
    print(f"  Mean: {states_linear.mean():.4f}")
    print(f"  Std: {states_linear.std():.4f}")

    print("\nNonlinear reservoir statistics:")
    print(f"  Mean: {states_nonlinear.mean():.4f}")
    print(f"  Std: {states_nonlinear.std():.4f}")


def example_model_comparison():
    """Example: Compare different architectures on the same task."""
    print("\n" + "=" * 60)
    print("Example 5: Architecture Comparison")
    print("=" * 60)

    # Create models with same reservoir size
    reservoir_size = 100
    input_size = 2
    output_size = 1

    classic = classic_esn(reservoir_size, input_size, output_size)
    ott = ott_esn(reservoir_size, input_size, output_size)

    # Same input
    x = torch.randn(4, 50, input_size)

    # Compare outputs
    out_classic = classic(x)
    out_ott = ott(x)

    print(f"Input shape: {x.shape}")
    print(f"Classic ESN output: {out_classic.shape}")
    print(f"Ott ESN output: {out_ott.shape}")

    # Compare parameter counts
    classic_params = sum(p.numel() for p in classic.parameters())
    ott_params = sum(p.numel() for p in ott.parameters())

    print("\nParameter counts:")
    print(f"  Classic ESN: {classic_params:,}")
    print(f"  Ott ESN: {ott_params:,}")


def example_flexible_specs():
    """Example: Showing the three ways to specify topology/initializers."""
    print("\n" + "=" * 60)
    print("Example 6: Flexible Spec Formats")
    print("=" * 60)

    # Method 1: String (uses registry defaults)
    model1 = classic_esn(100, 1, 1, topology="erdos_renyi")
    print("1. String spec: topology='erdos_renyi'")

    # Method 2: Tuple (name + custom params)
    model2 = classic_esn(100, 1, 1, topology=("watts_strogatz", {"k": 6, "p": 0.1}))
    print("2. Tuple spec: topology=('watts_strogatz', {'k': 6, 'p': 0.1})")

    # Method 3: Pre-configured object

    model3 = classic_esn(100, 1, 1, topology=("ring_chord", {"L": 2, "w": 0.5}))
    print("3. Object spec: topology=('ring_chord', {'L': 2, 'w': 0.5})")

    # Same for initializers
    from resdag.init.input_feedback import get_input_feedback

    model4 = classic_esn(100, 1, 1, feedback_initializer="pseudo_diagonal")
    model5 = classic_esn(100, 1, 1, feedback_initializer=("chebyshev", {"p": 0.5}))
    model6 = classic_esn(100, 1, 1, feedback_initializer=get_input_feedback("random_binary"))

    print("\nAll models created successfully!")


def example_gpu_usage():
    """Example: Using premade models on GPU."""
    if not torch.cuda.is_available():
        print("\n" + "=" * 60)
        print("Example 7: GPU Usage (CUDA not available)")
        print("=" * 60)
        return

    print("\n" + "=" * 60)
    print("Example 7: GPU Usage")
    print("=" * 60)

    # Create model and move to GPU
    model = classic_esn(reservoir_size=100, feedback_size=1, output_size=1).cuda()

    # Create GPU tensors
    x = torch.randn(4, 50, 1).cuda()

    # Forward pass on GPU
    output = model(x)

    print(f"Input device: {x.device}")
    print(f"Output device: {output.device}")
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Run all examples
    example_classic_esn()
    example_ott_esn()
    example_headless_esn()
    example_linear_esn()
    example_model_comparison()
    example_flexible_specs()
    example_gpu_usage()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
