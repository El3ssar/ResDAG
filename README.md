# torch_rc - PyTorch Reservoir Computing Library

A modern, GPU-accelerated reservoir computing library built on PyTorch.

## Features

- 🚀 **GPU-Accelerated**: All operations run on GPU by default
- 🧩 **Modular Architecture**: Build arbitrary DAGs with reservoir + readout layers
- 🔧 **Flexible APIs**: Builder pattern or functional context manager
- 🎓 **Dual Training**: Classical ESN or standard PyTorch SGD
- 📊 **Production-Ready**: Pure nn.Module semantics, TorchScript compatible

## Installation

```bash
# From PyPI (when published)
pip install torch_rc

# For development
uv sync --dev
```

## Performance

`torch_rc` is optimized for speed:

- **10k step forecast:** ~2.2s (4,500 steps/sec on CPU)
- **100k step forecast:** ~22s (linear scaling)
- Pre-allocated tensors and specialized forward paths
- See `FORECASTING_PERFORMANCE.md` for detailed benchmarks

## Quick Start

### Functional API (Recommended)

```python
import torch
from torch_rc.composition import model_scope
from torch_rc.layers import ReservoirLayer, ReadoutLayer

# Build model with functional API
with model_scope() as m:
    feedback = m.input("feedback")
    reservoir = m(
        ReservoirLayer(100, feedback_size=10, topology="erdos_renyi"),
        inputs=feedback
    )
    readout = m(
        ReadoutLayer(100, 5, name="output"),
        inputs=reservoir
    )
    model = m.build(outputs=readout)

# Forward pass
inputs = {"feedback": torch.randn(2, 10, 10)}
output = model(inputs)  # Shape: (2, 10, 5)
```

### Builder API (Alternative)

```python
from torch_rc.composition import ModelBuilder

# Build model with builder pattern
builder = ModelBuilder()
feedback = builder.input("feedback")
reservoir = builder.add(
    ReservoirLayer(100, feedback_size=10, topology="erdos_renyi"),
    inputs=feedback
)
readout = builder.add(
    ReadoutLayer(100, 5, name="output"),
    inputs=reservoir
)
model = builder.build(outputs=readout)
```

## Development Status

**v0.3.0 (Alpha)** - Phase 3 Complete

### Completed Features

- ✅ **Phase 1**: Core Layer Infrastructure
  - `TorchRCModule` base class
  - `ReservoirLayer` with feedback and driving inputs
  - `ReadoutLayer` with per-timestep processing
- ✅ **Phase 2**: Graph Topology System
  - 15 graph topology initializers (Erdős-Rényi, Watts-Strogatz, etc.)
  - 10 input/feedback weight initializers
  - Spectral radius scaling
- ✅ **Phase 3**: Model Composition APIs
  - DAG-based model composition
  - `ModelBuilder` (explicit builder pattern)
  - `model_scope` (functional context manager)
  - GPU support and `torch.compile` compatibility
  - Trainable weights feature
  - Model visualization (text and Graphviz)

### In Progress

- 🚧 **Phase 4**: Training Infrastructure (Next)
  - Conjugate gradient solver
  - ESNTrainer core
  - ReadoutLayer fitting

See `_bmad-output/implementation-artifacts/tech-spec-torch-rc-v1.md` for full roadmap.

## License

MIT License - See LICENSE file for details
