"""Example usage of functional-style model_scope API.

This demonstrates the functional context manager API for building ESN models,
which provides a more concise syntax compared to ModelBuilder.
"""

import torch
from pytorch_symbolic import Input, SymbolicModel

from resdag.layers import ReadoutLayer, ReservoirLayer

print("=" * 80)
print("Functional API Examples")
print("=" * 80)

# ============================================================================
# Example 1: Simple Sequential Model
# ============================================================================
print("\n1. Simple Sequential Model")
print("-" * 80)

feedback = Input((10, 1))
reservoir = ReservoirLayer(100, feedback_size=1)(feedback)
readout = ReadoutLayer(in_features=100, out_features=5)(reservoir)

model = SymbolicModel(inputs=feedback, outputs=readout)

print(f"Model: {model}")
print(f"Inputs: {model.inputs}")
print(f"Outputs: {model.outputs}")

# Forward pass
inputs = torch.randn(2, 10, 1)
output = model(inputs)
print(f"Output shape: {output.shape}")

# ============================================================================
# Example 2: Deep Sequential Model (Multiple Reservoirs)
# ============================================================================
print("\n2. Deep Sequential Model")
print("-" * 80)

feedback = Input((10, 1))
reservoir1 = ReservoirLayer(100, feedback_size=1)(feedback)
reservoir2 = ReservoirLayer(80, feedback_size=100)(reservoir1)
reservoir3 = ReservoirLayer(60, feedback_size=80)(reservoir2)
readout = ReadoutLayer(in_features=60, out_features=5)(reservoir3)

model = SymbolicModel(inputs=feedback, outputs=readout)

print(model.summary())

# Forward pass
inputs = torch.randn(2, 10, 1)
output = model(inputs)
print(f"Input shape: {inputs.shape}")
print(f"Output shape: {output.shape}")

# ============================================================================
# Example 3: Branching Model (Parallel Paths)
# ============================================================================
print("\n3. Branching Model")
print("-" * 80)

feedback = Input((10, 1))
reservoir1 = ReservoirLayer(100, feedback_size=1)(feedback)
reservoir2 = ReservoirLayer(80, feedback_size=100)(reservoir1)
reservoir3 = ReservoirLayer(60, feedback_size=80)(reservoir2)
readout1 = ReadoutLayer(in_features=60, out_features=5)(reservoir3)
readout2 = ReadoutLayer(in_features=60, out_features=3)(reservoir3)
model = SymbolicModel(inputs=feedback, outputs=[readout1, readout2])

# Two parallel branches
reservoir1 = ReservoirLayer(100, feedback_size=1)(feedback)
reservoir2 = ReservoirLayer(80, feedback_size=1)(feedback)

# Two outputs
readout1 = ReadoutLayer(in_features=100, out_features=5)(reservoir1)
readout2 = ReadoutLayer(in_features=80, out_features=3)(reservoir2)

model = SymbolicModel(inputs=feedback, outputs=[readout1, readout2])

print("model.summary()")

# Forward pass
inputs = torch.randn(2, 10, 1)
outputs = model(inputs)
print(f"Input shape: {inputs.shape}")
print(f"Output1 shape: {outputs[0].shape}")
print(f"Output2 shape: {outputs[1].shape}")

# ============================================================================
# Example 4: Multi-Input Model
# ============================================================================
print("\n5. Multi-Input Model")
print("-" * 80)

feedback = Input((10, 1))
driving = Input((10, 5))
reservoir = ReservoirLayer(100, feedback_size=1, input_size=5)(feedback, driving)
readout = ReadoutLayer(in_features=100, out_features=1)(reservoir)
model = SymbolicModel(inputs=[feedback, driving], outputs=readout)

print(model.summary())

# Forward pass
inputs = (torch.randn(2, 10, 1), torch.randn(2, 10, 5))
output = model(*inputs)
print(f"Feedback shape: {inputs[0].shape}")
print(f"Driving shape: {inputs[1].shape}")
print(f"Output shape: {output.shape}")

# ============================================================================
# Example 6: Complex DAG
# ============================================================================
print("\n6. Complex DAG")
print("-" * 80)

feedback = Input((10, 1))
driving = Input((10, 5))

reservoir1 = ReservoirLayer(100, feedback_size=1, input_size=5)(feedback, driving)

# Branch into two reservoirs
reservoir2 = ReservoirLayer(80, feedback_size=100)(reservoir1)
reservoir3 = ReservoirLayer(60, feedback_size=80)(reservoir2)

readout1 = ReadoutLayer(in_features=80, out_features=5)(reservoir2)
readout2 = ReadoutLayer(in_features=60, out_features=3)(reservoir3)

model = SymbolicModel(inputs=[feedback, driving], outputs=[readout1, readout2])

print(model.summary())

# Forward pass
inputs = (torch.randn(2, 10, 1), torch.randn(2, 10, 5))
outputs = model(*inputs)
print(f"Feedback shape: {inputs[0].shape}")
print(f"Driving shape: {inputs[1].shape}")
print(f"Output1 shape: {outputs[0].shape}")
print(f"Output2 shape: {outputs[1].shape}")
