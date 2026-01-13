# ResDAG

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A modern, GPU-accelerated reservoir computing library for PyTorch.**


`resdag` brings the power of Echo State Networks (ESNs) and reservoir computing to PyTorch with a clean, modular API. Built for researchers and practitioners who need fast, flexible, and production-ready reservoir computing models.

---

## ✨ Key Features

- 🚀 **GPU-Accelerated**: Full GPU support for training and inference
- 🎯 **Pure PyTorch**: Native `nn.Module` components, TorchScript compatible
- 🧩 **Modular Design**: Build complex architectures with simple building blocks
- 📊 **Multiple Topologies**: 15+ graph topologies for reservoir initialization
- 🔬 **Algebraic Training**: Efficient ridge regression via Conjugate Gradient
- 🎨 **Flexible API**: Compose models using `pytorch_symbolic`
- 📈 **HPO Ready**: Built-in Optuna integration for hyperparameter optimization
- 🔧 **Production Ready**: Stateful layers, model persistence, GPU compilation

---

## 📦 Installation

### From pip (recommended)

```bash
pip install resdag

# With hyperparameter optimization support
pip install resdag[hpo]
```

### From source

```bash
git clone https://github.com/El3ssar/resdag.git
cd resdag
pip install -e .

# Or using uv (faster)
uv sync
```

---

## 🚀 Quick Start

### Your First ESN in 30 Seconds

```python
import torch
import pytorch_symbolic as ps
from resdag import ESNModel, ReservoirLayer, CGReadoutLayer, ESNTrainer

# 1. Define the model architecture
inp = ps.Input((100, 3))  # (seq_len, features)
reservoir = ReservoirLayer(
    reservoir_size=500,
    feedback_size=3,
    spectral_radius=0.9,
    topology="erdos_renyi"
)(inp)
readout = CGReadoutLayer(500, 3, alpha=1e-6, name="output")(reservoir)
model = ESNModel(inp, readout)

# 2. Train the model (algebraic, not SGD!)
trainer = ESNTrainer(model)
trainer.fit(
    warmup_inputs=(warmup_data,),
    train_inputs=(train_data,),
    targets={"output": train_targets}
)

# 3. Make predictions
predictions = model.forecast(forecast_warmup, horizon=1000)
```

### Using Premade Models

```python
from resdag.models import ott_esn

# Ott's ESN for chaotic systems (with state augmentation)
model = ott_esn(
    reservoir_size=500,
    feedback_size=3,
    output_size=3,
    spectral_radius=0.95,
)

# Train and forecast as above
```

---

## 📖 Core Concepts

### Reservoir Layers

The heart of ESNs - stateful RNN layers with randomly initialized, fixed recurrent weights:

```python
from resdag.layers import ReservoirLayer
from resdag.init.topology import get_topology

reservoir = ReservoirLayer(
    reservoir_size=500,        # Number of neurons
    feedback_size=3,           # Dimension of feedback input
    input_size=5,              # Optional: dimension of driving inputs
    spectral_radius=0.9,       # Controls memory/stability
    leak_rate=1.0,             # Leaky integration (1.0 = no leak)
    activation="tanh",         # Activation function
    topology=get_topology("watts_strogatz", k=4, p=0.3),
)

# Forward pass
states = reservoir(feedback)                    # Feedback only
states = reservoir(feedback, driving_input)     # With driving input
```

### Readout Layers

Linear layers trained via ridge regression (not gradient descent):

```python
from resdag.layers.readouts import CGReadoutLayer

readout = CGReadoutLayer(
    in_features=500,           # Reservoir size
    out_features=3,            # Output dimension
    alpha=1e-6,                # Ridge regularization
    name="output",             # Name for multi-readout models
)

# Fit using conjugate gradient
readout.fit(reservoir_states, targets)
output = readout(reservoir_states)
```

### Model Composition

Build models using `pytorch_symbolic` for clean, functional composition:

```python
import pytorch_symbolic as ps
from resdag import ESNModel
from resdag.layers import ReservoirLayer, Concatenate
from resdag.layers.readouts import CGReadoutLayer

# Multi-input model with driving signal
feedback = ps.Input((100, 3))
driver = ps.Input((100, 5))

reservoir = ReservoirLayer(500, feedback_size=3, input_size=5)(feedback, driver)
readout = CGReadoutLayer(500, 3, name="output")(reservoir)

model = ESNModel([feedback, driver], readout)
```

### Training

Efficient algebraic training via `ESNTrainer`:

```python
from resdag.training import ESNTrainer

trainer = ESNTrainer(model)

# Two-phase training: warmup + fitting
trainer.fit(
    warmup_inputs=(warmup_feedback, warmup_driver),  # Synchronize states
    train_inputs=(train_feedback, train_driver),      # Fit readout
    targets={"output": targets},                      # One target per readout
)
```

### Forecasting

Two-phase forecasting: teacher-forced warmup + autoregressive generation:

```python
# Simple forecast (feedback only)
predictions = model.forecast(warmup_data, horizon=1000)

# Input-driven forecast (with external signals)
predictions = model.forecast(
    warmup_feedback,
    warmup_driver,
    horizon=1000,
    forecast_drivers=(future_driver,),  # Provide future driving inputs
)

# Include warmup in output
full_output = model.forecast(
    warmup_data,
    horizon=1000,
    return_warmup=True,
)
```

---

## 🎯 Advanced Usage

### Graph Topologies

`resdag` supports 15+ graph topologies for reservoir initialization:

```python
from resdag.init.topology import get_topology, show_topologies

# List all available topologies
show_topologies()

# Get details for a specific topology
show_topologies("erdos_renyi")

# Create topology initializer
topology = get_topology("watts_strogatz", k=6, p=0.3, seed=42)

# Use in reservoir
reservoir = ReservoirLayer(
    reservoir_size=500,
    feedback_size=3,
    topology=topology,
    spectral_radius=0.95,
)
```

**Available topologies:**

- `erdos_renyi` - Random graphs with edge probability
- `watts_strogatz` - Small-world networks
- `barabasi_albert` - Scale-free networks
- `complete` - Fully connected
- `ring_chord` - Ring with chords
- `dendrocycle` - Dendritic cycles
- And many more!

### Input/Feedback Initializers

Custom initialization strategies for input/feedback weights:

```python
from resdag.init.input_feedback import get_input_feedback

# List available initializers
from resdag.init.input_feedback import show_input_initializers
show_input_initializers()

# Use custom initializer
reservoir = ReservoirLayer(
    reservoir_size=500,
    feedback_size=3,
    feedback_initializer=get_input_feedback("chebyshev"),
    input_initializer=get_input_feedback("random", input_scaling=0.5),
)
```

### Multi-Readout Models

Train models with multiple outputs:

```python
# Define multiple readouts
reservoir = ReservoirLayer(500, feedback_size=3)(inp)

readout1 = CGReadoutLayer(500, 3, name="position")(reservoir)
readout2 = CGReadoutLayer(500, 3, name="velocity")(reservoir)

model = ESNModel(inp, [readout1, readout2])

# Train both readouts
trainer.fit(
    warmup_inputs=(warmup,),
    train_inputs=(train,),
    targets={
        "position": position_targets,
        "velocity": velocity_targets,
    },
)
```

### Data Utilities

Built-in utilities for data loading and preparation:

```python
from resdag.utils.data import load_file, prepare_esn_data

# Load time series
data = load_file("lorenz.csv")  # Auto-detects format

# Split into ESN training phases
warmup, train, target, f_warmup, val = prepare_esn_data(
    data,
    warmup_steps=100,     # Reservoir synchronization
    train_steps=500,      # Readout training
    val_steps=200,        # Validation
    normalize="minmax",   # Normalization method
)
```

### Hyperparameter Optimization

Built-in Optuna integration for HPO:

```python
from resdag.hpo import run_hpo
from resdag.models import ott_esn

def model_creator(reservoir_size, spectral_radius):
    return ott_esn(
        reservoir_size=reservoir_size,
        feedback_size=3,
        output_size=3,
        spectral_radius=spectral_radius,
    )

def search_space(trial):
    return {
        "reservoir_size": trial.suggest_int("reservoir_size", 100, 1000, step=100),
        "spectral_radius": trial.suggest_float("spectral_radius", 0.5, 1.5),
    }

def data_loader(trial):
    return {
        "warmup": warmup,
        "train": train,
        "target": target,
        "f_warmup": f_warmup,
        "val": val,
    }

# Run optimization
study = run_hpo(
    model_creator=model_creator,
    search_space=search_space,
    data_loader=data_loader,
    n_trials=100,
    loss="efh",           # Expected Forecast Horizon (best for chaotic systems)
    n_workers=4,          # Parallel optimization
)

print(f"Best params: {study.best_params}")
```

**Available loss functions:**

- `efh` - Expected Forecast Horizon (recommended for chaotic systems)
- `horizon` - Contiguous valid forecast steps
- `lyap` - Lyapunov-weighted loss
- `standard` - Mean geometric error
- `discounted` - Time-discounted RMSE

---

## 📚 Examples

Comprehensive examples are available in the [`examples/`](examples/) directory:

- **00_registry_system.py** - Working with topology and initializer registries
- **01_reservoir_with_topology.py** - Using different graph topologies
- **02_input_feedback_initializers.py** - Custom weight initialization
- **06_premade_models.py** - Using premade ESN architectures
- **07_save_load_models.py** - Model persistence and checkpointing
- **08_forecasting.py** - Time series forecasting examples
- **09_training.py** - Training workflows with ESNTrainer
- **10_hpo.py** - Hyperparameter optimization examples

Run any example:

```bash
python examples/08_forecasting.py
```

---

## 🔬 Use Cases

### Chaotic System Prediction

```python
from resdag.models import ott_esn
from resdag.training import ESNTrainer

# Lorenz attractor prediction
model = ott_esn(reservoir_size=500, feedback_size=3, output_size=3)

trainer = ESNTrainer(model)
trainer.fit(warmup_inputs=(warmup,), train_inputs=(train,), targets={"output": target})

# Long-term prediction
predictions = model.forecast(forecast_warmup, horizon=5000)
```

### Input-Driven Systems

```python
# System with external forcing
feedback = ps.Input((100, 3))
driver = ps.Input((100, 2))

reservoir = ReservoirLayer(500, feedback_size=3, input_size=2)(feedback, driver)
readout = CGReadoutLayer(500, 3, name="output")(reservoir)
model = ESNModel([feedback, driver], readout)

# Forecast with future driving signals
predictions = model.forecast(
    warmup_feedback,
    warmup_driver,
    horizon=1000,
    forecast_drivers=(future_driver,),
)
```

### Multi-Scale Predictions

```python
# Predict at multiple timescales
reservoir = ReservoirLayer(1000, feedback_size=3)(inp)

short_term = CGReadoutLayer(1000, 3, name="1step")(reservoir)
medium_term = CGReadoutLayer(1000, 3, name="10step")(reservoir)
long_term = CGReadoutLayer(1000, 3, name="100step")(reservoir)

model = ESNModel(inp, [short_term, medium_term, long_term])
```

---

## 🎓 Documentation

Full documentation is available in the [`docs/`](docs/) directory:

- **[Topology System](docs/topology_system.md)** - Graph topologies for reservoirs
- **[Input/Feedback Initializers](docs/input_feedback_initializers.md)** - Weight initialization strategies
- **[Model Composition](docs/model_composition.md)** - Building complex architectures
- **[Training Guide](docs/training.md)** - ESN training workflows
- **[Hyperparameter Optimization](docs/hyperparameter_optimization.md)** - HPO best practices
- **[Save/Load Models](docs/save_load.md)** - Model persistence

### API Reference

Generate API documentation using Sphinx:

```bash
cd docs/
sphinx-apidoc -o api/ ../src/resdag
make html
```

---

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=resdag --cov-report=html

# Run specific test module
pytest tests/test_layers/test_reservoir.py
```

Current test coverage: **57%** (240 tests passing)

---

## 🛠️ Development

### Setting up Development Environment

```bash
# Clone repository
git clone https://github.com/yourusername/resdag.git
cd resdag

# Install with development dependencies
uv sync --dev

# Or with pip
pip install -e ".[dev]"
```

### Code Quality

We use `ruff` for linting and `black` for formatting:

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

### Project Structure

```
resdag/
├── src/resdag/
│   ├── composition/       # Model composition (pytorch_symbolic)
│   ├── layers/            # Reservoir, Readout, custom layers
│   ├── init/              # Weight initialization
│   │   ├── topology/      # Graph topologies
│   │   ├── input_feedback/ # Input/feedback initializers
│   │   └── graphs/        # Graph generation functions
│   ├── training/          # ESNTrainer and training utilities
│   ├── models/            # Premade model architectures
│   ├── hpo/               # Hyperparameter optimization
│   └── utils/             # Data loading and utilities
├── tests/                 # Comprehensive test suite
├── examples/              # Example scripts
└── docs/                  # Documentation
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Ensure tests pass (`pytest`)
5. Format code (`black src/ tests/`)
6. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📄 Citation

If you use `resdag` in your research, please cite:

```bibtex
@software{resdag2024,
  author = {Daniel Estevez-Moya},
  title = {resdag: A PyTorch Library for Reservoir Computing},
  year = {2024},
  url = {https://github.com/yourusername/resdag}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built on [PyTorch](https://pytorch.org/) and [pytorch_symbolic](https://github.com/pytorch-labs/pytorch-symbolic)
- Inspired by [ReservoirPy](https://github.com/reservoirpy/reservoirpy) and classical ESN literature
- Graph generation powered by [NetworkX](https://networkx.org/)

---

## 📬 Contact

- **Author**: Daniel Estevez-Moya
- **Email**: kemossabee@gmail.com
- **Issues**: [GitHub Issues](https://github.com/yourusername/resdag/issues)

---

## 🗺️ Roadmap

- [ ] Additional premade architectures (Liquid State Machines, Next-Gen RC)
- [ ] Online learning capabilities
- [ ] TorchScript export for production
- [ ] ONNX support
- [ ] Distributed training for large reservoirs
- [ ] Interactive visualization tools
- [ ] Benchmarking suite against other ESN libraries

---

**⭐ Star us on GitHub if you find this useful!**
