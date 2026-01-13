# resdag Examples

This directory contains examples demonstrating the various features of the `resdag` library.

## Migration to pytorch_symbolic

**Note**: Examples have been updated to use the new `pytorch_symbolic`-based API. The main differences:

### Old API (deprecated):

```python
from resdag.composition import ModelBuilder

builder = ModelBuilder()
inp = builder.input("input")
reservoir = builder.add(ReservoirLayer(100, 1, 0), inputs=inp)
readout = builder.add(CGReadoutLayer(100, 1), inputs=reservoir)
model = builder.build(outputs=readout)

# Dict-based forward
output = model({"input": x})
```

### New API (current):

```python
import pytorch_symbolic as ps
from resdag.composition import ESNModel

inp = ps.Input((100, 1))  # seq_len, features
reservoir = ReservoirLayer(100, 1, 0)(inp)
readout = CGReadoutLayer(100, 1)(reservoir)
model = ESNModel(inp, readout)

# Direct forward (no dict!)
output = model(x)
```

## Examples Overview

### Basic Examples

1. **01_reservoir_with_topology.py** - Using graph topologies for reservoir initialization

   - Pre-registered topologies
   - Custom topology parameters
   - Comparing different topologies

2. **02_input_feedback_initializers.py** - Input/feedback weight initialization methods
   - Standard initializers
   - Custom initializers
   - Comparing initialization strategies

### Model Building

3. **03_model_composition.py** - Building ESN models with ModelBuilder (legacy)

   - Sequential models
   - Branching models
   - Multi-input models
   - **Note**: Uses legacy ModelBuilder API for backward compatibility

4. **04_model_visualization.py** - Visualizing model architectures

   - Model summary (Keras-style)
   - Graph visualization with torchvista

5. **05_functional_api.py** - Using model_scope for concise model building (legacy)
   - Context manager API
   - **Note**: Uses legacy model_scope API

### Premade Models

6. **06_premade_models.py** ✅ - Using premade architectures (pytorch_symbolic)
   - `classic_esn`: Traditional ESN with input concatenation
   - `ott_esn`: Ott's ESN with state augmentation
   - `headless_esn`: Reservoir-only for analysis
   - `linear_esn`: Linear reservoir for baselines

### Advanced Features

7. **07_save_load_models.py** ✅ - Model persistence (pytorch_symbolic)

   - Basic save/load
   - Training workflows
   - Checkpoint systems
   - Cross-device loading

8. **08_forecasting.py** ✅ - Time series forecasting (pytorch_symbolic)

   - Basic forecasting
   - With driving inputs
   - State history tracking
   - Long-horizon prediction

9. **09_training.py** ✅ - ESN training with ESNTrainer (pytorch_symbolic)

   - Simple training workflow
   - Multi-readout training
   - Training with driving inputs
   - Different regularization per readout

10. **10_hpo.py** ✅ - Hyperparameter Optimization with Optuna
    - Basic HPO setup
    - Different loss functions (EFH, Lyapunov, discounted RMSE)
    - Study persistence and resumption
    - Parallel execution
    - Advanced conditional search spaces
    - Custom loss functions

## Running Examples

Each example is standalone and can be run directly:

```bash
# Run a specific example
uv run python examples/06_premade_models.py

# Run all examples
for example in examples/*.py; do
    echo "Running $example..."
    uv run python "$example"
done
```

## Migration Status

| Example                           | Status               | Notes                               |
| --------------------------------- | -------------------- | ----------------------------------- |
| 01_reservoir_with_topology.py     | ✅ No changes needed | Layer-level only                    |
| 02_input_feedback_initializers.py | ✅ No changes needed | Layer-level only                    |
| 03_model_composition.py           | ⚠️ Legacy            | Uses ModelBuilder (still supported) |
| 04_model_visualization.py         | ⚠️ Legacy            | Uses ModelBuilder (still supported) |
| 05_functional_api.py              | ⚠️ Legacy            | Uses model_scope (still supported)  |
| 06_premade_models.py              | ✅ Migrated          | Uses pytorch_symbolic               |
| 07_save_load_models.py            | ✅ Migrated          | Uses pytorch_symbolic               |
| 08_forecasting.py                 | ✅ Migrated          | Uses pytorch_symbolic               |
| 09_training.py                    | ✅ New               | ESNTrainer with pytorch_symbolic    |
| 10_hpo.py                         | ✅ New               | Optuna-based HPO                    |

## Key Features Demonstrated

### pytorch_symbolic Features:

- Direct tensor input (no dicts)
- Keras-style `model.summary()`
- Cleaner, more Pythonic API
- Better IDE support

### ESN Features:

- Graph-based reservoir topologies
- Custom weight initialization
- Stateful processing
- Multi-input/multi-output models
- Time series forecasting
- Model persistence
- GPU acceleration

## Need Help?

- Check the main README: `../README.md`
- Read migration guide: `../PYTORCH_SYMBOLIC_MIGRATION.md`
- View API docs: `../docs/`
