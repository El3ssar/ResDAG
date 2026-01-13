"""Example demonstrating model visualization with pytorch_symbolic and graphviz.

This example shows how to visualize ESN models using:
1. model.summary() - Text-based summary (like Keras)
2. model.plot_model() - Graph visualization using graphviz
"""

import pytorch_symbolic as ps

from resdag.composition import ESNModel
from resdag.layers import ReservoirLayer
from resdag.layers.readouts import CGReadoutLayer
from resdag.layers.custom import Concatenate
from resdag.models import classic_esn, ott_esn

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
print("✓ Visualization generated (feedback-only model)")

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

print("\nGenerating visualization...")
model.plot_model()
print("✓ Visualization generated (multi-input model)")

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
# Example 4: Complex Branching Architecture
# ============================================================================
print("\n\n4. Complex Branching Architecture")
print("-" * 70)

# Build a complex model with branching paths
feedback = ps.Input(shape=(1, 1))
inputs = ps.Input(shape=(1, 1))
reservoir = ReservoirLayer(200, 1, input_size=1)(feedback, inputs)
readout = CGReadoutLayer(reservoir.shape[-1], 2)(reservoir)

inputs1 = ps.Input(shape=(1, 2))
reservoir1 = ReservoirLayer(100, 2, input_size=2)(readout, inputs1)
readout1 = CGReadoutLayer(reservoir1.shape[-1], 1)(reservoir1)

reservoir2 = ReservoirLayer(150, 1, input_size=0)(readout1)
readout2 = CGReadoutLayer(reservoir2.shape[-1], 1)(reservoir2)

concat = Concatenate()(readout, readout2)

model = ESNModel(inputs=(feedback, inputs, inputs1), outputs=concat)

print("\nText Summary:")
model.summary()

print("\nGenerating visualization...")
model.plot_model()
print("✓ Visualization generated (complex branching model)")

# ============================================================================
# Example 5: Premade Models
# ============================================================================
print("\n\n5. Premade Models (Classic ESN)")
print("-" * 70)

model = classic_esn(reservoir_size=100, feedback_size=1, output_size=1)

print("\nText Summary:")
model.summary()

print("\nGenerating visualization...")
model.plot_model()
print("✓ Visualization generated")

# ============================================================================
# Example 6: Visualization Options
# ============================================================================
print("\n\n6. Visualization Options")
print("-" * 70)

model = ott_esn(reservoir_size=100, feedback_size=1, output_size=1)

print("\nOption 1: Default view (top-to-bottom)")
model.plot_model()
print("✓ Default view generated")

print("\nOption 2: Left-to-right layout")
model.plot_model(rankdir="LR")
print("✓ Left-to-right view generated")

print("\nOption 3: Without shapes")
model.plot_model(show_shapes=False)
print("✓ View without shapes generated")

print("\nOption 4: Save to file")
model.plot_model(save_path="/tmp/model_output.svg")
print("✓ Saved to /tmp/model_output.svg")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("\nVisualization methods:")
print("  1. model.summary()       - Text-based summary (pytorch_symbolic)")
print("  2. model.plot_model()    - Graph visualization (graphviz)")
print("\nKey features:")
print("  - Uses symbolic graph structure for accurate visualization")
print("  - Correctly shows complex branching architectures")
print("  - Works with multi-input models")
print("  - SVG/PNG/PDF export support")
print("\nVisualization options:")
print("  - plot_model()                     # Default: top-to-bottom with shapes")
print("  - plot_model(rankdir='LR')         # Left-to-right layout")
print("  - plot_model(show_shapes=False)    # Hide tensor shapes")
print("  - plot_model(save_path='x.svg')    # Save to file")
print("\n✅ All examples completed!")
