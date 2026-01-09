# Input/Feedback Weight Initializers

## Overview

The input/feedback initializer system provides specialized initialization strategies for the rectangular weight matrices that connect inputs (feedback or external driving signals) to reservoir neurons.

Unlike graph topologies (which are for square recurrent weight matrices), these initializers work with arbitrary rectangular matrices and are designed for the specific role of input connections.

## PyTorch Convention

All initializers follow PyTorch's F.linear convention where weights are stored as **(out_features, in_features)**:

- Feedback weights: `(reservoir_size, feedback_size)`
- Input weights: `(reservoir_size, input_size)`

This is the transpose of Keras/TensorFlow convention, and all initializers have been adapted accordingly.

## Available Initializers

### Core Initializers

#### RandomInputInitializer

**Most common baseline** - Uniform random values in [-1, 1]

```python
from torch_rc.init.input_feedback import RandomInputInitializer

init = RandomInputInitializer(input_scaling=1.0, seed=42)
```

**Parameters:**

- `input_scaling`: Scaling factor (typical: 0.1-5.0)
- `seed`: Random seed for reproducibility

**Use case:** Default choice, good baseline for most tasks

#### RandomBinaryInitializer

Binary values in {-1, +1}

```python
init = RandomBinaryInitializer(input_scaling=0.5, seed=42)
```

**Advantages:**

- Memory efficient (can store as bits)
- Computational efficiency (multiply → add/subtract)
- Often surprisingly effective

### Structured Initializers

#### PseudoDiagonalInitializer

Block-diagonal structure where each input connects to a contiguous block of neurons

```python
init = PseudoDiagonalInitializer(
    input_scaling=1.0,
    binarize=False,  # Can make it binary
    seed=42
)
```

**Use case:** When input dimensions have semantic meaning (e.g., different sensors)

#### ChessboardInitializer

Deterministic alternating {-1, +1} pattern

```python
init = ChessboardInitializer(input_scaling=1.0)
```

**Pattern:** W[i, j] = (-1)^(i+j)

**Use case:** High-frequency structured pattern, fully deterministic

### Deterministic Chaotic Initializers

#### ChebyshevInitializer

Uses Chebyshev polynomial mapping for deterministic chaotic initialization

```python
init = ChebyshevInitializer(
    p=0.3,           # Initial amplitude (0, 1)
    q=5.9,           # Sinusoidal spread
    k=3.8,           # Chaos parameter (2, 4)
    input_scaling=0.8
)
```

**How it works:**

- First column: Sinusoidal initialization
- Subsequent columns: Chebyshev recurrence creates chaotic structure
- Fully deterministic, no randomness

**Use case:** Structured chaos, reproducible patterns

#### BinaryBalancedInitializer

Hadamard-based balanced binary initialization

```python
init = BinaryBalancedInitializer(
    input_scaling=0.5,
    balance_global=True,
    step=None,  # Auto-select
    seed=None   # Unused, fully deterministic
)
```

**Properties:**

- Values in {-1, +1}
- Each column sums to 0 (balanced)
- Low inter-column correlation (nearly orthogonal)
- Fully deterministic

**Use case:** When you want balanced, uncorrelated binary inputs

### Topology-Specific Initializers

#### OppositeAnchorsInitializer

For ring topologies - connects each input to two opposite points

```python
init = OppositeAnchorsInitializer(gain=1.0)
```

**Use case:** Ring/cycle reservoirs, bipolar activation patterns

#### DendrocycleInputInitializer

Specific to dendrocycle topologies

```python
init = DendrocycleInputInitializer(
    c=0.2,  # or C=20 (number of core nodes)
    input_scaling=0.5,
    seed=42
)
```

**Behavior:** Only core (cycle) nodes receive inputs, rest are zero

#### ChainOfNeuronsInputInitializer

For parallel chain reservoirs

```python
init = ChainOfNeuronsInputInitializer(
    features=3,  # Number of chains
    weights=1.0  # or [1.0, 0.5, 0.8] for per-chain weights
)
```

**Behavior:** Each input connects only to the first neuron of its chain

#### RingWindowInputInitializer

Windowed inputs on ring topology

```python
init = RingWindowInputInitializer(
    c=0.5,           # Core fraction
    window=10,       # Window size (int or fraction)
    taper="cosine",  # "flat", "triangle", "cosine"
    signed="alt_ring",  # "allpos", "alt_ring", "alt_inputs"
    gain=1.0
)
```

**Use case:** Dendrocycle+chords with localized input windows

## Usage

### Basic Usage

```python
from torch_rc.layers import ReservoirLayer
from torch_rc.init.input_feedback import RandomInputInitializer, ChebyshevInitializer

# Initialize feedback weights with RandomInput
feedback_init = RandomInputInitializer(input_scaling=0.5, seed=42)

# Initialize driving input weights with Chebyshev
input_init = ChebyshevInitializer(p=0.3, k=3.5, input_scaling=0.8)

# Create reservoir with custom initializers
reservoir = ReservoirLayer(
    reservoir_size=200,
    feedback_size=10,
    input_size=5,
    feedback_initializer=feedback_init,
    input_initializer=input_init,
    topology="erdos_renyi",
    spectral_radius=0.9
)
```

### Direct Initialization

```python
import torch
from torch_rc.init.input_feedback import BinaryBalancedInitializer

# Create weight tensor
weight = torch.empty(100, 10)  # (reservoir_size, input_dim)

# Initialize
init = BinaryBalancedInitializer(input_scaling=0.5)
init.initialize(weight)

# Or use callable interface
init(weight)
```

## Choosing an Initializer

### Decision Guide

**Start simple:**

1. **RandomInputInitializer** - Good default, well-understood
2. Tune `input_scaling` based on task

**For specific needs:**

- **Binary/sparse weights** → RandomBinaryInitializer, BinaryBalancedInitializer
- **Structured connectivity** → PseudoDiagonalInitializer
- **Reproducible patterns** → ChessboardInitializer, ChebyshevInitializer
- **Balanced inputs** → BinaryBalancedInitializer
- **Specific topologies** → Use topology-specific initializers

### Input Scaling Guidelines

- **0.1-0.5**: Weak input, reservoir dynamics dominate
- **0.5-1.0**: Balanced input and reservoir dynamics
- **1.0-5.0**: Strong input-driven dynamics
- **Lower is often better** for chaotic/complex reservoirs

## Implementation Details

### Matrix Shape Convention

All initializers expect PyTorch convention:

- Shape: `(out_features, in_features) = (reservoir_size, input_dim)`
- Computation: `F.linear(input, weight)` = `input @ weight.T`

This is the **transpose** of Keras/TF where shapes were `(input_dim, reservoir_size)`.

### In-Place Modification

All initializers modify the weight tensor in-place:

```python
weight = torch.empty(100, 10)
init.initialize(weight)  # Modifies weight in-place
# weight now contains initialized values
```

### Reproducibility

Use `seed` parameters for reproducible initialization:

```python
init1 = RandomInputInitializer(input_scaling=1.0, seed=42)
init2 = RandomInputInitializer(input_scaling=1.0, seed=42)

weight1 = torch.empty(100, 10)
weight2 = torch.empty(100, 10)

init1.initialize(weight1)
init2.initialize(weight2)

assert torch.allclose(weight1, weight2)  # True
```

**Note:** Some initializers (Chessboard, BinaryBalanced) are fully deterministic and don't use seeds.

## Examples

See `examples/02_input_feedback_initializers.py` for comprehensive examples.

## References

- Rodan & Tiňo (2011): "Minimum complexity echo state network"
- Xie, Wang & Yu (2024): "Time Series Prediction of ESN Based on Chebyshev Mapping"
- Hadamard matrices: Walsh-Hadamard transform for balanced binary initialization
