"""Example usage of functional-style model_scope API.

This demonstrates the functional context manager API for building ESN models,
which provides a more concise syntax compared to ModelBuilder.
"""

import torch
import torch.nn as nn

from torch_rc.composition import model_scope
from torch_rc.layers import ReadoutLayer, ReservoirLayer

print("=" * 80)
print("Functional API Examples")
print("=" * 80)

# ============================================================================
# Example 1: Simple Sequential Model
# ============================================================================
print("\n1. Simple Sequential Model")
print("-" * 80)

with model_scope() as m:
    feedback = m.input("feedback")
    reservoir = m(ReservoirLayer(100, feedback_size=10), inputs=feedback)
    readout = m(ReadoutLayer(in_features=100, out_features=5, name="output"), inputs=reservoir)

    model = m.build(outputs=readout)

print(f"Model: {model}")
print(f"Inputs: {model.input_names}")
print(f"Outputs: {model.output_names}")

# Forward pass
inputs = {"feedback": torch.randn(2, 10, 10)}
output = model({"input": x})
print(f"Input shape: {inputs['feedback'].shape}")
print(f"Output shape: {output.shape}")

# ============================================================================
# Example 2: Deep Sequential Model (Multiple Reservoirs)
# ============================================================================
print("\n2. Deep Sequential Model")
print("-" * 80)

with model_scope() as m:
    feedback = m.input("feedback")

    # Chain of reservoirs
    res1 = m(ReservoirLayer(100, feedback_size=10), inputs=feedback, name="res1")
    res2 = m(ReservoirLayer(80, feedback_size=100), inputs=res1, name="res2")
    res3 = m(ReservoirLayer(60, feedback_size=80), inputs=res2, name="res3")

    readout = m(ReadoutLayer(in_features=60, out_features=5, name="output"), inputs=res3)

    model = m.build(outputs=readout)

print(f"Model: {model}")

# Forward pass
inputs = {"feedback": torch.randn(2, 15, 10)}
output = model({"input": x})
print(f"Input shape: {inputs['feedback'].shape}")
print(f"Output shape: {output.shape}")

# ============================================================================
# Example 3: Branching Model (Parallel Paths)
# ============================================================================
print("\n3. Branching Model")
print("-" * 80)

with model_scope() as m:
    feedback = m.input("feedback")

    # Two parallel branches
    res1 = m(ReservoirLayer(100, feedback_size=10), inputs=feedback, name="branch1")
    res2 = m(ReservoirLayer(80, feedback_size=10), inputs=feedback, name="branch2")

    # Two outputs
    out1 = m(ReadoutLayer(in_features=100, out_features=5, name="output1"), inputs=res1)
    out2 = m(ReadoutLayer(in_features=80, out_features=3, name="output2"), inputs=res2)

    model = m.build(outputs=[out1, out2])

print(f"Model: {model}")
print(f"Outputs: {model.output_names}")

# Forward pass
inputs = {"feedback": torch.randn(2, 10, 10)}
outputs = model({"input": x})
print(f"Input shape: {inputs['feedback'].shape}")
print(f"Output1 shape: {outputs['output1'].shape}")
print(f"Output2 shape: {outputs['output2'].shape}")

# ============================================================================
# Example 4: Merging Branches
# ============================================================================
print("\n4. Merging Branches")
print("-" * 80)


# Custom merge layer
class ConcatLayer(nn.Module):
    """Concatenates inputs along feature dimension."""

    def forward(self, x1, x2):
        return torch.cat([x1, x2], dim=-1)


with model_scope() as m:
    feedback = m.input("feedback")

    # Two parallel branches
    res1 = m(ReservoirLayer(100, feedback_size=10), inputs=feedback, name="branch1")
    res2 = m(ReservoirLayer(80, feedback_size=10), inputs=feedback, name="branch2")

    # Merge and readout
    merged = m(ConcatLayer(), inputs=[res1, res2], name="merge")
    readout = m(ReadoutLayer(in_features=180, out_features=5, name="output"), inputs=merged)

    model = m.build(outputs=readout)

print(f"Model: {model}")

# Forward pass
inputs = {"feedback": torch.randn(2, 10, 10)}
output = model({"input": x})
print(f"Input shape: {inputs['feedback'].shape}")
print(f"Output shape: {output.shape}")

# ============================================================================
# Example 5: Multi-Input Model
# ============================================================================
print("\n5. Multi-Input Model")
print("-" * 80)

with model_scope() as m:
    feedback = m.input("feedback")
    driving = m.input("driving")

    # Reservoir with both feedback and driving input
    reservoir = m(
        ReservoirLayer(100, feedback_size=10, input_size=5),
        inputs=[feedback, driving],
        name="reservoir",
    )

    readout = m(ReadoutLayer(in_features=100, out_features=3, name="output"), inputs=reservoir)

    model = m.build(outputs=readout)

print(f"Model: {model}")
print(f"Inputs: {model.input_names}")

# Forward pass
inputs = {"feedback": torch.randn(2, 10, 10), "driving": torch.randn(2, 10, 5)}
output = model({"input": x})
print(f"Feedback shape: {inputs['feedback'].shape}")
print(f"Driving shape: {inputs['driving'].shape}")
print(f"Output shape: {output.shape}")

# ============================================================================
# Example 6: Complex DAG
# ============================================================================
print("\n6. Complex DAG")
print("-" * 80)

with model_scope() as m:
    # Multiple inputs
    feedback = m.input("feedback")
    driving = m.input("driving")

    # First reservoir with both inputs
    res1 = m(
        ReservoirLayer(100, feedback_size=10, input_size=5),
        inputs=[feedback, driving],
        name="res1",
    )

    # Branch into two reservoirs
    res2 = m(ReservoirLayer(80, feedback_size=100), inputs=res1, name="res2")
    res3 = m(ReservoirLayer(60, feedback_size=100), inputs=res1, name="res3")

    # Multiple outputs
    out1 = m(ReadoutLayer(in_features=80, out_features=5, name="output1"), inputs=res2)
    out2 = m(ReadoutLayer(in_features=60, out_features=3, name="output2"), inputs=res3)

    model = m.build(outputs=[out1, out2])

print(f"Model: {model}")
print(f"Inputs: {model.input_names}")
print(f"Outputs: {model.output_names}")

# Forward pass
inputs = {"feedback": torch.randn(2, 10, 10), "driving": torch.randn(2, 10, 5)}
outputs = model({"input": x})
print(f"Output1 shape: {outputs['output1'].shape}")
print(f"Output2 shape: {outputs['output2'].shape}")

# ============================================================================
# Example 7: Comparison with ModelBuilder
# ============================================================================
print("\n7. Comparison: ModelBuilder vs model_scope")
print("-" * 80)

print("\nModelBuilder syntax:")
print("```python")
print("from torch_rc.composition import ModelBuilder")
print("")
print("builder = ModelBuilder()")
print("feedback = builder.input('feedback')")
print("reservoir = builder.add(ReservoirLayer(100, feedback_size=10), inputs=feedback)")
print("readout = builder.add(ReadoutLayer(100, 5, name='output'), inputs=reservoir)")
print("model = builder.build(outputs=readout)")
print("```")

print("\nmodel_scope syntax:")
print("```python")
print("from torch_rc.composition import model_scope")
print("")
print("with model_scope() as m:")
print("    feedback = m.input('feedback')")
print("    reservoir = m(ReservoirLayer(100, feedback_size=10), inputs=feedback)")
print("    readout = m(ReadoutLayer(100, 5, name='output'), inputs=reservoir)")
print("    model = m.build(outputs=readout)")
print("```")

print("\n" + "=" * 80)
print("Key differences:")
print("  - model_scope uses context manager (with statement)")
print("  - m(module, inputs=...) instead of m.add(module, inputs=...)")
print("  - Slightly more concise while maintaining explicit graph construction")
print("  - Both APIs produce identical ESNModel objects")
print("=" * 80)
