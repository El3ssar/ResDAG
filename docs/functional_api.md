# Functional API for Model Composition

The `model_scope` provides a functional-style context manager API for building ESN models. It offers a more concise syntax compared to `ModelBuilder` while maintaining explicit graph construction.

## Overview

The functional API wraps `ModelBuilder` in a clean context manager interface, providing:

- **Cleaner syntax**: `m(module, inputs=...)` instead of `m.add(module, inputs=...)`
- **Context management**: Automatic scope handling with `with` statement
- **Same semantics**: Produces identical `ESNModel` objects as `ModelBuilder`
- **No monkey-patching**: Pure wrapper approach with no magic

## Basic Usage

### Simple Sequential Model

```python
from torch_rc.composition import model_scope
from torch_rc.layers import ReservoirLayer, ReadoutLayer

with model_scope() as m:
    feedback = m.input("feedback")
    reservoir = m(ReservoirLayer(100, feedback_size=10), inputs=feedback)
    readout = m(ReadoutLayer(100, 5, name="output"), inputs=reservoir)

    model = m.build(outputs=readout)
```

### Deep Sequential Model

```python
with model_scope() as m:
    feedback = m.input("feedback")

    # Chain of reservoirs
    res1 = m(ReservoirLayer(100, feedback_size=10), inputs=feedback, name="res1")
    res2 = m(ReservoirLayer(80, feedback_size=100), inputs=res1, name="res2")
    res3 = m(ReservoirLayer(60, feedback_size=80), inputs=res2, name="res3")

    readout = m(ReadoutLayer(60, 5, name="output"), inputs=res3)

    model = m.build(outputs=readout)
```

## Advanced Patterns

### Branching (Parallel Paths)

```python
with model_scope() as m:
    feedback = m.input("feedback")

    # Two parallel branches
    res1 = m(ReservoirLayer(100, feedback_size=10), inputs=feedback, name="branch1")
    res2 = m(ReservoirLayer(80, feedback_size=10), inputs=feedback, name="branch2")

    # Two outputs
    out1 = m(ReadoutLayer(100, 5, name="output1"), inputs=res1)
    out2 = m(ReadoutLayer(80, 3, name="output2"), inputs=res2)

    model = m.build(outputs=[out1, out2])
```

### Merging Branches

```python
import torch.nn as nn

class ConcatLayer(nn.Module):
    def forward(self, x1, x2):
        return torch.cat([x1, x2], dim=-1)

with model_scope() as m:
    feedback = m.input("feedback")

    # Parallel branches
    res1 = m(ReservoirLayer(100, feedback_size=10), inputs=feedback)
    res2 = m(ReservoirLayer(80, feedback_size=10), inputs=feedback)

    # Merge and readout
    merged = m(ConcatLayer(), inputs=[res1, res2])
    readout = m(ReadoutLayer(180, 5, name="output"), inputs=merged)

    model = m.build(outputs=readout)
```

### Multi-Input Models

```python
with model_scope() as m:
    feedback = m.input("feedback")
    driving = m.input("driving")

    # Reservoir with both inputs
    reservoir = m(
        ReservoirLayer(100, feedback_size=10, input_size=5),
        inputs=[feedback, driving]
    )

    readout = m(ReadoutLayer(100, 3, name="output"), inputs=reservoir)

    model = m.build(outputs=readout)

# Usage
inputs = {
    "feedback": torch.randn(2, 10, 10),
    "driving": torch.randn(2, 10, 5)
}
output = model(inputs)
```

### Complex DAG

```python
with model_scope() as m:
    # Multiple inputs
    feedback = m.input("feedback")
    driving = m.input("driving")

    # First reservoir with both inputs
    res1 = m(
        ReservoirLayer(100, feedback_size=10, input_size=5),
        inputs=[feedback, driving],
        name="res1"
    )

    # Branch into two reservoirs
    res2 = m(ReservoirLayer(80, feedback_size=100), inputs=res1, name="res2")
    res3 = m(ReservoirLayer(60, feedback_size=100), inputs=res1, name="res3")

    # Multiple outputs
    out1 = m(ReadoutLayer(80, 5, name="output1"), inputs=res2)
    out2 = m(ReadoutLayer(60, 3, name="output2"), inputs=res3)

    model = m.build(outputs=[out1, out2])
```

## API Reference

### `model_scope`

Context manager for building ESN models with functional syntax.

**Methods:**

- **`input(name: str) -> NodeRef`**: Add an input node

  - `name`: Name for the input
  - Returns: Reference to the input node

- **`__call__(module, inputs: Union[NodeRef, List[NodeRef]], name: Optional[str] = None) -> NodeRef`**: Add a module

  - `module`: PyTorch module to add
  - `inputs`: Single NodeRef or list of NodeRefs
  - `name`: Optional custom name
  - Returns: Reference to the added node

- **`build(outputs: Optional[Union[NodeRef, List[NodeRef]]] = None) -> ESNModel`**: Build the model
  - `outputs`: Single NodeRef, list of NodeRefs, or None (auto-detect)
  - Returns: Compiled ESNModel

### `NodeRef`

Lightweight reference to a node in the computation graph.

**Attributes:**

- `name`: The node name in the DAG
- `scope`: The model_scope that owns this node

## Comparison: ModelBuilder vs model_scope

### ModelBuilder (Explicit API)

```python
from torch_rc.composition import ModelBuilder

builder = ModelBuilder()
feedback = builder.input("feedback")
reservoir = builder.add(ReservoirLayer(100, feedback_size=10), inputs=feedback)
readout = builder.add(ReadoutLayer(100, 5, name="output"), inputs=reservoir)
model = builder.build(outputs=readout)
```

### model_scope (Functional API)

```python
from torch_rc.composition import model_scope

with model_scope() as m:
    feedback = m.input("feedback")
    reservoir = m(ReservoirLayer(100, feedback_size=10), inputs=feedback)
    readout = m(ReadoutLayer(100, 5, name="output"), inputs=reservoir)
    model = m.build(outputs=readout)
```

### Key Differences

| Feature        | ModelBuilder                      | model_scope                      |
| -------------- | --------------------------------- | -------------------------------- |
| Style          | Explicit builder pattern          | Functional context manager       |
| Adding modules | `builder.add(module, inputs=...)` | `m(module, inputs=...)`          |
| Context        | Manual instance creation          | `with` statement                 |
| Outputs        | Identical `ESNModel`              | Identical `ESNModel`             |
| Use case       | When you prefer builder pattern   | When you prefer functional style |

## When to Use Which?

### Use `ModelBuilder` when:

- You prefer explicit builder pattern
- Building models programmatically in loops
- You want maximum clarity

### Use `model_scope` when:

- You prefer functional/context manager style
- Building models interactively
- You want more concise syntax

## Implementation Notes

- **No monkey-patching**: `model_scope` is a pure wrapper around `ModelBuilder`
- **Identical outputs**: Both APIs produce the same `ESNModel` objects
- **Same performance**: No overhead, just syntactic sugar
- **Composable**: Can nest contexts if needed

## Examples

See `examples/05_functional_api.py` for comprehensive examples including:

1. Simple sequential models
2. Deep sequential models
3. Branching models
4. Merging branches
5. Multi-input models
6. Complex DAGs
7. API comparison

## Best Practices

1. **Use descriptive names**: Name your nodes clearly for easier debugging

   ```python
   reservoir = m(ReservoirLayer(100, 10), inputs=feedback, name="main_reservoir")
   ```

2. **Group related operations**: Structure your code logically

   ```python
   with model_scope() as m:
       # Define inputs
       feedback = m.input("feedback")
       driving = m.input("driving")

       # Build architecture
       reservoir = m(ReservoirLayer(...), inputs=[feedback, driving])

       # Define outputs
       readout = m(ReadoutLayer(...), inputs=reservoir)

       # Build model
       model = m.build(outputs=readout)
   ```

3. **Use custom merge layers**: Create reusable merge operations

   ```python
   class WeightedSum(nn.Module):
       def __init__(self, num_inputs):
           super().__init__()
           self.weights = nn.Parameter(torch.ones(num_inputs) / num_inputs)

       def forward(self, *inputs):
           stacked = torch.stack(inputs, dim=0)
           weights = self.weights.view(-1, 1, 1, 1)
           return (stacked * weights).sum(dim=0)
   ```

4. **Visualize complex models**: Use `print_structure()` or `plot_model()`
   ```python
   model.print_structure()
   model.plot_model("model.png", show_params=True)
   ```
