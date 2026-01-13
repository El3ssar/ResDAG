"""Example: Model Visualization with pytorch_symbolic and torchvista.

This example demonstrates how to visualize ESN models using the new
pytorch_symbolic-based API with torchvista integration.
"""

import torch

from resdag.models import classic_esn, headless_esn, ott_esn

print("=" * 70)
print("MODEL VISUALIZATION WITH PYTORCH_SYMBOLIC")
print("=" * 70)

# ============================================================================
# Example 1: Classic ESN Visualization
# ============================================================================
print("\n1. Classic ESN")
print("-" * 70)

model = classic_esn(100, 1, 1)

# Show text summary
print("\nModel Summary:")
model.summary()

# Visualize with torchvista (displays in notebooks)
print("\nGenerating visualization...")
result = model.plot_model()
print("✓ Visualization generated (displayed in notebook)")

# ============================================================================
# Example 2: Ott's ESN with State Augmentation
# ============================================================================
print("\n\n2. Ott's ESN (with State Augmentation)")
print("-" * 70)

model_ott = ott_esn(200, 2, 3)

print("\nModel Summary:")
model_ott.summary()

print("\nGenerating visualization...")
result_ott = model_ott.plot_model()
print("✓ Visualization shows SelectiveExponentiation layer")

# ============================================================================
# Example 3: Headless ESN (No Readout)
# ============================================================================
print("\n\n3. Headless ESN (Reservoir Only)")
print("-" * 70)

model_headless = headless_esn(100, 2)

print("\nModel Summary:")
model_headless.summary()

print("\nGenerating visualization...")
result_headless = model_headless.plot_model()
print("✓ Visualization shows reservoir without readout")

# ============================================================================
# Example 4: Custom Input Dimensions
# ============================================================================
print("\n\n4. Custom Visualization Parameters")
print("-" * 70)

model_custom = classic_esn(50, 3, 2)

# Use custom batch size and sequence length for visualization
print("\nVisualizing with batch_size=4, seq_len=20:")
result_custom = model_custom.plot_model(batch_size=4, seq_len=20)
print("✓ Custom parameters applied")

# ============================================================================
# Example 5: Visualization with Custom Input Data
# ============================================================================
print("\n\n5. Visualization with Specific Input")
print("-" * 70)

model_data = classic_esn(80, 2, 1)

# Provide specific input tensor for tracing
input_tensor = torch.randn(2, 30, 2)  # (batch=2, seq_len=30, features=2)
print(f"\nUsing input tensor of shape: {input_tensor.shape}")

result_data = model_data.plot_model(input_data=input_tensor)
print("✓ Traced with specific input data")

# ============================================================================
# Example 6: Comparing Different Architectures
# ============================================================================
print("\n\n6. Comparing Architectures")
print("-" * 70)

print("\nArchitecture Comparison:")
print(f"{'Model':<20} {'Layers':<30} {'Parameters':<15}")
print("-" * 65)

models = {
    "classic_esn": classic_esn(100, 1, 1),
    "ott_esn": ott_esn(100, 1, 1),
    "headless_esn": headless_esn(100, 1),
}

for name, m in models.items():
    layer_count = len(list(m.named_modules())) - 1  # Exclude root
    param_count = sum(p.numel() for p in m.parameters())
    layer_names = [type(mod).__name__ for _, mod in m.named_modules() if _ != ""]
    layer_str = ", ".join(set(layer_names))[:28] + "..."
    print(f"{name:<20} {layer_str:<30} {param_count:<15,}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("VISUALIZATION FEATURES")
print("=" * 70)

print("""
1. model.summary()
   - Keras-style text summary
   - Shows layer names, output shapes, parameters
   - Works everywhere (no dependencies)

2. model.plot_model()
   - Interactive visualization with torchvista
   - Best for Jupyter notebooks
   - Shows model graph structure
   - Supports custom input dimensions

Usage in Notebooks:
   >>> model = classic_esn(100, 1, 1)
   >>> model.plot_model()  # Interactive graph
   
Usage in Scripts:
   >>> model.summary()  # Text-based summary
""")

print("\n✅ All visualization examples completed!")
