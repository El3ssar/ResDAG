"""Example demonstrating model visualization with pytorch_symbolic and torchvista.

This example shows how to visualize ESN models using:
1. model.summary() - Text-based summary (like Keras)
2. model.plot_model() - Interactive graph visualization
"""

import pytorch_symbolic as ps

from torch_rc.composition import ESNModel
from torch_rc.layers import ReservoirLayer
from torch_rc.layers.readouts import CGReadoutLayer
from torch_rc.models import classic_esn, ott_esn

print("=" * 70)
print("MODEL VISUALIZATION EXAMPLES")
print("=" * 70)

# ============================================================================
# Example 1: Simple Feedback-Only Model
# ============================================================================
print("\n1. Simple Feedback-Only Model")
print("-" * 70)

# Build model
feedback_input = ps.Input((50, 1))
reservoir = ReservoirLayer(reservoir_size=100, feedback_size=1)(feedback_input)
readout = CGReadoutLayer(in_features=100, out_features=1)(reservoir)
model = ESNModel(feedback_input, readout)

print("\nText Summary:")
model.summary()

print("\nGenerating visualization...")
model.plot_model()
print("✓ Visualization generated (clean view with only layers)")

# ============================================================================
# Example 2: Input-Driven Model
# ============================================================================
print("\n\n2. Input-Driven Model (Feedback + Driving)")
print("-" * 70)

# Build model with two inputs
feedback_input = ps.Input((50, 1))
driving_input = ps.Input((50, 3))
reservoir = ReservoirLayer(reservoir_size=150, feedback_size=1, input_size=3)(
    feedback_input, driving_input
)
readout = CGReadoutLayer(in_features=150, out_features=1)(reservoir)
model = ESNModel([feedback_input, driving_input], readout)

print("\nText Summary:")
model.summary()

print("\nNote: Multi-input model visualization skipped")
print("(torchvista has limitations with multi-input models)")
print("Use model.summary() for text-based visualization")
print("✓ Multi-input model created successfully")

# ============================================================================
# Example 3: Deep ESN (Stacked Reservoirs)
# ============================================================================
print("\n\n3. Deep ESN (Stacked Reservoirs)")
print("-" * 70)

feedback_input = ps.Input((50, 1))
res1 = ReservoirLayer(reservoir_size=150, feedback_size=1)(feedback_input)
res2 = ReservoirLayer(reservoir_size=100, feedback_size=150)(res1)
res3 = ReservoirLayer(reservoir_size=50, feedback_size=100)(res2)
readout = CGReadoutLayer(in_features=50, out_features=1)(res3)
model = ESNModel(feedback_input, readout)

print("\nText Summary:")
model.summary()

print("\nGenerating visualization...")
model.plot_model()
print("✓ Visualization generated")

# ============================================================================
# Example 4: Premade Models
# ============================================================================
print("\n\n4. Premade Models (Classic ESN)")
print("-" * 70)

model = classic_esn(reservoir_size=100, input_size=1, output_size=1)

print("\nText Summary:")
model.summary()

print("\nGenerating visualization...")
model.plot_model()
print("✓ Visualization generated")

# ============================================================================
# Example 5: Visualization Options
# ============================================================================
print("\n\n5. Visualization Options")
print("-" * 70)

model = ott_esn(reservoir_size=100, input_size=1, output_size=1)

print("\nOption 1: Clean view (default) - Only shows layers")
model.plot_model()
print("✓ Clean view generated")

print("\nOption 2: Detailed view - Shows all operations")
model.plot_model(show_non_gradient_nodes=True)
print("✓ Detailed view generated")

print("\nOption 3: Module internals - Shows what's inside each layer")
model.plot_model(collapse_modules_after_depth=1)
print("✓ Module internals view generated")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("\nVisualization methods:")
print("  1. model.summary()       - Text-based summary (pytorch_symbolic)")
print("  2. model.plot_model()    - Interactive graph with torchvista")
print("\nKey features:")
print("  - pytorch_symbolic: Keras-like model building")
print("  - torchvista: Interactive visualization for notebooks")
print("  - Works with premade models (classic_esn, ott_esn, etc.)")
print("  - Clean view by default (only shows layers)")
print("\nVisualization options:")
print("  - plot_model()                              # Clean: only layers")
print("  - plot_model(show_non_gradient_nodes=True)  # Detailed: all operations")
print("  - plot_model(collapse_modules_after_depth=1) # Show module internals")
print("\n✅ All examples completed!")
