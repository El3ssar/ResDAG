"""Example demonstrating model composition with pytorch_symbolic.

This example shows how to build ESN models with different input configurations:
1. Feedback-only models (output fed back as input)
2. Input-driven models (external driving signal + feedback)
"""

import pytorch_symbolic as ps
import torch

from resdag.composition import ESNModel
from resdag.layers import ReservoirLayer
from resdag.layers.readouts import CGReadoutLayer

print("=" * 70)
print("TORCH_RC MODEL COMPOSITION EXAMPLES")
print("=" * 70)

# ============================================================================
# Example 1: Feedback-Only Model
# ============================================================================
print("\n1. Feedback-Only Model (Classic ESN)")
print("-" * 70)
print("Architecture: Input (feedback) -> Reservoir -> Readout")
print("The reservoir receives only the previous output as feedback.")

# Define model architecture
feedback_input = ps.Input((50, 1))  # (seq_len, features)
reservoir = ReservoirLayer(
    reservoir_size=100,
    feedback_size=1,  # Receives 1D feedback from readout
    input_size=0,  # No driving input
    spectral_radius=0.9,
    leak_rate=0.3,
)(feedback_input)
readout = CGReadoutLayer(in_features=100, out_features=1)(reservoir)

model = ESNModel(feedback_input, readout)

print("\nModel Summary:")
model.summary()

# Forward pass
x = torch.randn(4, 50, 1)  # (batch, time, features)
output = model(x)
print(f"\nInput shape:  {x.shape}")
print(f"Output shape: {output.shape}")
print("✓ Feedback-only model working")

# ============================================================================
# Example 2: Input-Driven Model (Single Driving Input)
# ============================================================================
print("\n\n2. Input-Driven Model (Single Driving Input)")
print("-" * 70)
print("Architecture: Feedback + Driving -> Reservoir -> Readout")
print("The reservoir receives both feedback and an external driving signal.")

# Define model with two inputs
feedback_input = ps.Input((50, 1))  # Feedback from readout
driving_input = ps.Input((50, 3))  # External driving signal

reservoir = ReservoirLayer(
    reservoir_size=150,
    feedback_size=1,  # Receives 1D feedback
    input_size=3,  # Receives 3D driving input
    spectral_radius=0.95,
    leak_rate=0.2,
)(feedback_input, driving_input)  # Two inputs!

readout = CGReadoutLayer(in_features=150, out_features=1)(reservoir)

model = ESNModel([feedback_input, driving_input], readout)

print("\nModel Summary:")
model.summary()

# Forward pass with two inputs
feedback = torch.randn(4, 50, 1)  # (batch, time, feedback_features)
driving = torch.randn(4, 50, 3)  # (batch, time, driving_features)
output = model(feedback, driving)

print(f"\nFeedback shape: {feedback.shape}")
print(f"Driving shape:  {driving.shape}")
print(f"Output shape:   {output.shape}")
print("✓ Input-driven model working")

# ============================================================================
# Example 3: Multi-Dimensional Feedback + Driving
# ============================================================================
print("\n\n3. Multi-Dimensional Feedback + Driving")
print("-" * 70)
print("Architecture: Multi-dim feedback + Multi-dim driving -> Reservoir -> Readout")
print("Both feedback and driving can have multiple dimensions.")

# Define model with multi-dimensional inputs
feedback_input = ps.Input((50, 5))  # 5D feedback
driving_input = ps.Input((50, 10))  # 10D driving signal

reservoir = ReservoirLayer(
    reservoir_size=200,
    feedback_size=5,  # 5D feedback
    input_size=10,  # 10D driving
    spectral_radius=0.9,
    leak_rate=0.25,
)(feedback_input, driving_input)

readout = CGReadoutLayer(in_features=200, out_features=5)(reservoir)

model = ESNModel([feedback_input, driving_input], readout)

print("\nModel Summary:")
model.summary()

# Forward pass
feedback = torch.randn(2, 50, 5)
driving = torch.randn(2, 50, 10)
output = model(feedback, driving)

print(f"\nFeedback shape: {feedback.shape}")
print(f"Driving shape:  {driving.shape}")
print(f"Output shape:   {output.shape}")
print("✓ Multi-dimensional model working")

# ============================================================================
# Example 4: Stacked Reservoirs (Deep ESN)
# ============================================================================
print("\n\n4. Stacked Reservoirs (Deep ESN)")
print("-" * 70)
print("Architecture: Input -> Reservoir1 -> Reservoir2 -> Reservoir3 -> Readout")
print("Each reservoir feeds into the next, creating a deep hierarchy.")

# Define deep model
feedback_input = ps.Input((50, 1))

# Stack of reservoirs
res1 = ReservoirLayer(reservoir_size=150, feedback_size=1, input_size=0)(feedback_input)
res2 = ReservoirLayer(reservoir_size=100, feedback_size=150, input_size=0)(res1)
res3 = ReservoirLayer(reservoir_size=50, feedback_size=100, input_size=0)(res2)

readout = CGReadoutLayer(in_features=50, out_features=1)(res3)

model = ESNModel(feedback_input, readout)

print("\nModel Summary:")
model.summary()

# Forward pass
x = torch.randn(3, 50, 1)
output = model(x)

print(f"\nInput shape:  {x.shape}")
print(f"Output shape: {output.shape}")
print("✓ Deep ESN working")

# ============================================================================
# Example 5: State Management
# ============================================================================
print("\n\n5. State Management")
print("-" * 70)

# Create a simple model
feedback_input = ps.Input((20, 1))
reservoir = ReservoirLayer(reservoir_size=100, feedback_size=1)(feedback_input)
readout = CGReadoutLayer(in_features=100, out_features=1)(reservoir)
model = ESNModel(feedback_input, readout)

# Run forward pass
x = torch.randn(2, 20, 1)
output = model(x)

# Get reservoir states
states = model.get_reservoir_states()
print("Reservoir states:")
for name, state in states.items():
    print(f"  {name}: shape={state.shape}, mean={state.mean().item():.4f}")

# Reset reservoirs
model.reset_reservoirs()
states_reset = model.get_reservoir_states()
print("\nAfter reset:")
for name, state in states_reset.items():
    print(f"  {name}: shape={state.shape}, mean={state.mean().item():.4f}")
print("✓ State management working")

# ============================================================================
# Example 6: GPU Support
# ============================================================================
if torch.cuda.is_available():
    print("\n\n6. GPU Support")
    print("-" * 70)

    feedback_input = ps.Input((50, 1))
    reservoir = ReservoirLayer(reservoir_size=200, feedback_size=1)(feedback_input)
    readout = CGReadoutLayer(in_features=200, out_features=1)(reservoir)
    model = ESNModel(feedback_input, readout).cuda()

    x = torch.randn(4, 50, 1, device="cuda")
    output = model(x)

    print(f"Model on GPU: {next(model.parameters()).device}")
    print(f"Output on GPU: {output.device}")
    print(f"Output shape: {output.shape}")
    print("✓ GPU support working")
else:
    print("\n\n6. GPU Support")
    print("-" * 70)
    print("CUDA not available, skipping GPU example")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("\nKey Concepts:")
print("  1. Feedback-only: ReservoirLayer(feedback_size=N, input_size=0)")
print("  2. Input-driven:  ReservoirLayer(feedback_size=N, input_size=M)")
print("  3. Multi-input:   Pass multiple tensors to reservoir layer")
print("  4. Deep models:   Stack reservoir layers sequentially")
print("\nModel Building:")
print("  - Use pytorch_symbolic.Input() to define inputs")
print("  - Call layers with symbolic inputs to build graph")
print("  - Create ESNModel(inputs, outputs)")
print("\nForward Pass:")
print("  - Single input:  model(x)")
print("  - Multi-input:   model(feedback, driving)")
print("  - Direct tensor inputs (no dicts!)")
print("\n✅ All examples completed successfully!")
