# Model Visualization

## Overview

ESNModel provides a `plot_model()` method to visualize the DAG structure, similar to Keras's `plot_model()` function.

## Available Libraries

### 1. **graphviz** (Recommended for DAG visualization)

- **Pros**: Direct graph visualization, we already have the DAG structure
- **Cons**: Requires system installation of Graphviz
- **Best for**: Showing the explicit DAG topology we built

### 2. **torchinfo**

- **Pros**: Clean text summaries, no extra dependencies
- **Cons**: Text-only, no visual plots
- **Best for**: Quick parameter counts and shapes

### 3. **torchviz**

- **Pros**: Shows computation graph
- **Cons**: Shows autograd graph (not our logical DAG), requires forward pass
- **Best for**: Debugging gradient flow

### 4. **hiddenlayer**

- **Pros**: Designed for PyTorch, nice styling
- **Cons**: May not respect our DAG structure, shows traced execution
- **Best for**: Standard PyTorch models

## Recommended Approach

**Use `graphviz` directly** because:

1. ✅ We already have the DAG structure
2. ✅ Can show exactly what we want (node names, connections, metadata)
3. ✅ Lightweight and customizable
4. ✅ Works with our composition API
5. ✅ Can show trainable status, shapes, parameter counts

## Implementation Plan

```python
class ESNModel(nn.Module):
    def plot_model(
        self,
        filename="model",
        format="png",
        show_shapes=True,
        show_params=True,
        show_trainable=True,
        show_dtypes=False,
        rankdir="TB",  # Top to Bottom or Left to Right (LR)
    ):
        """
        Visualize the model DAG structure.

        Args:
            filename: Output filename (without extension)
            format: Output format (png, pdf, svg, etc.)
            show_shapes: Show layer output shapes
            show_params: Show parameter counts
            show_trainable: Show trainable status
            show_dtypes: Show data types
            rankdir: Graph direction (TB=top-to-bottom, LR=left-to-right)

        Returns:
            Path to saved file
        """
```

### Node Information to Display

For each node:

- **Name**: Module name or input name
- **Type**: Layer type (ReservoirLayer, ReadoutLayer, etc.)
- **Shape**: Output shape (if known from a forward pass)
- **Parameters**: Total parameter count
- **Trainable**: Whether parameters are trainable
- **Dtype**: Data type (optional)

### Visual Style

```
Input Nodes:     [Oval, light blue]
Reservoir Nodes: [Box, light green]
Readout Nodes:   [Box, light orange]
Other Nodes:     [Box, light gray]
Edges:           [Solid arrows]
```

## Installation

```bash
# Install Python package
uv add graphviz

# Install system Graphviz (required)
# Ubuntu/Debian
sudo apt-get install graphviz

# macOS
brew install graphviz

# Windows
# Download from https://graphviz.org/download/
```

## Usage Examples

### Basic Plot

```python
from resdag.composition import ModelBuilder
from resdag.layers import ReservoirLayer, ReadoutLayer

builder = ModelBuilder()
feedback = builder.input("feedback")
reservoir = builder.add(ReservoirLayer(100, feedback_size=10), inputs=feedback)
readout = builder.add(ReadoutLayer(in_features=100, out_features=5, name="output"), inputs=reservoir)

model = builder.build(outputs=readout)

# Plot model
model.plot_model("simple_model.png")
```

### Complex Model with Options

```python
# Complex branching model
builder = ModelBuilder()
feedback = builder.input("feedback")

res1 = builder.add(ReservoirLayer(100, feedback_size=10), inputs=feedback, name="res1")
res2 = builder.add(ReservoirLayer(80, feedback_size=10), inputs=feedback, name="res2")

out1 = builder.add(ReadoutLayer(in_features=100, out_features=5, name="out1"), inputs=res1)
out2 = builder.add(ReadoutLayer(in_features=80, out_features=3, name="out2"), inputs=res2)

model = builder.build(outputs=[out1, out2])

# Plot with all options
model.plot_model(
    "complex_model.png",
    show_shapes=True,
    show_params=True,
    show_trainable=True,
    rankdir="LR"  # Left to right layout
)
```

### Getting Shapes

```python
# Run a forward pass first to get shapes
sample_input = {"feedback": torch.randn(1, 10, 10)}
_ = model(sample_input)

# Now plot with shapes
model.plot_model("model_with_shapes.png", show_shapes=True)
```

## Alternative: torchinfo Summary

For a quick text summary (no visualization):

```python
from torchinfo import summary

# Requires a sample forward pass
model.eval()
summary(
    model,
    input_data={"feedback": torch.randn(4, 20, 10)},
    depth=5
)
```

Output:

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
ESNModel                                 [4, 20, 5]                --
├─ModuleDict: 1-1                        --                        --
│    └─ReservoirLayer: 2-1               [4, 20, 100]              11,100
│    └─ReadoutLayer: 2-2                 [4, 20, 5]                505
==========================================================================================
Total params: 11,605
Trainable params: 0
Non-trainable params: 11,605
==========================================================================================
```

## Fallback: Simple Text Representation

If Graphviz is not available, provide a simple text representation:

```python
model.print_structure()
```

Output:

```
ESNModel Structure:
==================
Inputs: ['feedback']
Outputs: ['output']

Execution Order:
1. reservoir (ReservoirLayer) <- feedback
   - Size: 100, Params: 11,100, Trainable: False
2. output (ReadoutLayer) <- reservoir
   - Size: 5, Params: 505, Trainable: False

Total Parameters: 11,605 (0 trainable)
```

## Summary

**Recommended**: Use `graphviz` for visual plots

- Best integration with our DAG structure
- Highly customizable
- Shows exactly what we built

**Alternative**: Use `torchinfo` for text summaries

- No system dependencies
- Quick parameter counts
- Good for debugging

**Implementation**: Add both to ESNModel

- `plot_model()` for visual (graphviz)
- `print_structure()` for text (built-in)
- `summary()` for detailed text (torchinfo, optional)
