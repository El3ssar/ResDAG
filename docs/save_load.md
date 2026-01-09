# Model Save and Load

`torch_rc` provides PyTorch-native save and load functionality for ESN models, allowing you to persist trained models to disk and load them for inference or continued training.

## Quick Start

```python
from torch_rc.models import classic_esn

# Create and train model
model = classic_esn(100, 1, 1)
# ... training code ...

# Save trained model
model.save('trained_model.pt')

# Load later
new_model = classic_esn(100, 1, 1)
new_model.load('trained_model.pt')
```

## API Reference

### `model.save(path, include_states=False, **kwargs)`

Save model parameters to file using PyTorch's standard format.

**Parameters:**

- `path` (str | Path): File path to save to (e.g., 'model.pt' or 'checkpoints/model.pth')
- `include_states` (bool): If True, also saves reservoir states (default: False)
- `**kwargs`: Additional arguments passed to `torch.save` (e.g., `pickle_protocol`)

**Example:**

```python
# Basic save
model.save('model.pt')

# Save with reservoir states
model.save('model_with_states.pt', include_states=True)

# Save to nested directory (creates parent dirs automatically)
model.save('checkpoints/experiment1/model.pt')
```

**Notes:**

- Saves only parameters (weights/biases), not model architecture
- Parent directories are created automatically if they don't exist
- Reservoir states are NOT saved by default (they're transient runtime state)

### `model.load(path, strict=True, load_states=False)`

Load model parameters from file.

**Parameters:**

- `path` (str | Path): File path to load from
- `strict` (bool): If True, requires exact parameter match (default: True)
- `load_states` (bool): If True, also loads reservoir states if present (default: False)

**Raises:**

- `RuntimeError`: If `strict=True` and state_dict doesn't match model architecture
- `FileNotFoundError`: If path doesn't exist

**Example:**

```python
# Basic load
model.load('model.pt')

# Load with reservoir states
model.load('model_with_states.pt', load_states=True)

# Non-strict loading (allows missing/unexpected keys)
model.load('model.pt', strict=False)
```

**Notes:**

- Model architecture must match the saved model
- Reservoir states are NOT loaded by default
- Use `strict=False` to allow missing or unexpected keys (but not size mismatches)

### `ESNModel.load_from_file(path, model, strict=True, load_states=False)`

Class method to load a model from file (requires model instance).

**Parameters:**

- `path` (str | Path): File path to load from
- `model` (ESNModel): ESNModel instance with correct architecture
- `strict` (bool): If True, requires exact parameter match
- `load_states` (bool): If True, also loads reservoir states if present

**Returns:**

- The model instance with loaded parameters

**Example:**

```python
# Method 1: Using class method
model = classic_esn(100, 1, 1)
ESNModel.load_from_file('trained_model.pt', model=model)

# Method 2: Using instance method (more common)
model = classic_esn(100, 1, 1)
model.load('trained_model.pt')
```

**Note:** The instance method `model.load()` is usually more convenient.

## Common Workflows

### 1. Training and Inference

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch_rc.models import classic_esn

# Training
model = classic_esn(
    100, 1, 1,
    reservoir_config={'trainable': False},
    readout_config={'trainable': True}
)

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

model.train()
for epoch in range(10):
    optimizer.zero_grad()
    output = model({'input': x_train})
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

# Save trained model
model.save('trained_model.pt')

# Later: Load for inference
inference_model = classic_esn(
    100, 1, 1,
    reservoir_config={'trainable': False},
    readout_config={'trainable': True}
)
inference_model.load('trained_model.pt')
inference_model.eval()

with torch.no_grad():
    predictions = inference_model({'input': x_test})
```

### 2. Checkpoint System

```python
from pathlib import Path

checkpoint_dir = Path('checkpoints')
checkpoint_dir.mkdir(exist_ok=True)

best_loss = float('inf')
best_checkpoint = None

for epoch in range(100):
    # Training...
    loss = train_one_epoch(model, optimizer, criterion, train_loader)

    # Save checkpoint
    checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
    model.save(checkpoint_path)

    # Track best model
    if loss < best_loss:
        best_loss = loss
        best_checkpoint = checkpoint_path

# Load best model
model.load(best_checkpoint)
```

### 3. Cross-Device Loading

```python
# Save on CPU
model_cpu = classic_esn(100, 1, 1)
model_cpu.save('model.pt')

# Load on GPU
model_gpu = classic_esn(100, 1, 1).cuda()
model_gpu.load('model.pt')  # Automatically handles device transfer

# Or load CPU model and then move to GPU
model = classic_esn(100, 1, 1)
model.load('model.pt')
model = model.cuda()
```

### 4. Reservoir State Management

```python
from torch_rc.models import headless_esn

model = headless_esn(100, 2)

# Process sequence
x = torch.randn(4, 50, 2)
states = model({'input': x})

# Save with states (for continuing processing later)
model.save('model_with_states.pt', include_states=True)

# Later: Resume processing from saved state
model = headless_esn(100, 2)
model.load('model_with_states.pt', load_states=True)

# Continue processing
x_continue = torch.randn(4, 20, 2)
states_continue = model({'input': x_continue})
```

### 5. Model Versioning

```python
import torch

# Save model
model.save('model_v1.pt')

# Add metadata
checkpoint = torch.load('model_v1.pt', weights_only=False)
checkpoint['metadata'] = {
    'version': '1.0',
    'reservoir_size': 100,
    'trained_on': '2024-01-01',
    'training_loss': 0.123,
    'notes': 'Initial model with default hyperparameters'
}
torch.save(checkpoint, 'model_v1.pt')

# Load and inspect metadata
checkpoint = torch.load('model_v1.pt', weights_only=False)
print(checkpoint['metadata'])

# Load model
model.load('model_v1.pt')
```

## File Format

Models are saved as PyTorch checkpoint dictionaries with the following structure:

```python
{
    'state_dict': {
        # All model parameters (weights, biases)
        '_modules_dict.reservoir.weight_hh': tensor(...),
        '_modules_dict.reservoir.weight_feedback': tensor(...),
        '_modules_dict.output.weight': tensor(...),
        # ... more parameters ...
    },
    'reservoir_states': {  # Optional, only if include_states=True
        'reservoir': tensor(...),  # Current reservoir state
        # ... more reservoir states if multiple reservoirs ...
    }
}
```

## Best Practices

### 1. Always Match Architecture

The model architecture must match when loading:

```python
# ✓ Correct
model_save = classic_esn(100, 1, 1)
model_save.save('model.pt')

model_load = classic_esn(100, 1, 1)  # Same architecture
model_load.load('model.pt')

# ✗ Incorrect
model_load = classic_esn(200, 1, 1)  # Different size!
model_load.load('model.pt')  # RuntimeError!
```

### 2. Don't Save Reservoir States by Default

Reservoir states are transient runtime state and usually shouldn't be saved:

```python
# ✓ Typical use case
model.save('model.pt')  # Don't save states

# ✓ Special case: continuing a long sequence
model.save('model.pt', include_states=True)
```

### 3. Use Checkpoints During Training

Save checkpoints regularly to avoid losing progress:

```python
for epoch in range(100):
    train_one_epoch(...)

    if epoch % 10 == 0:
        model.save(f'checkpoint_epoch_{epoch}.pt')
```

### 4. Separate Training and Inference Code

```python
# training.py
model = classic_esn(100, 1, 1, readout_config={'trainable': True})
# ... training ...
model.save('trained_model.pt')

# inference.py
model = classic_esn(100, 1, 1, readout_config={'trainable': True})
model.load('trained_model.pt')
model.eval()
# ... inference ...
```

### 5. Handle Device Transfer Explicitly

```python
# Save on any device
model.save('model.pt')

# Load and move to desired device
model = classic_esn(100, 1, 1)
model.load('model.pt')

if torch.cuda.is_available():
    model = model.cuda()
```

## Limitations

1. **Architecture Required**: You must reconstruct the model architecture before loading parameters. The save file only contains parameters, not the architecture definition.

2. **Size Mismatches**: Even with `strict=False`, PyTorch will raise errors for parameter size mismatches. `strict=False` only allows missing or unexpected keys.

3. **Custom Modules**: If you use custom layers or modules, they must be importable when loading.

4. **Reservoir States**: Reservoir states are device-specific and may not transfer well across different batch sizes or devices.

## Troubleshooting

### RuntimeError: size mismatch

**Problem:** Model architecture doesn't match saved model.

**Solution:** Ensure the model you're loading into has the exact same architecture as the saved model.

```python
# Check saved model info
checkpoint = torch.load('model.pt', weights_only=False)
print(checkpoint['state_dict'].keys())
```

### FileNotFoundError

**Problem:** Save path doesn't exist.

**Solution:** Check the path or use `include_states=True` if you need states.

```python
from pathlib import Path
path = Path('model.pt')
if not path.exists():
    print(f"File not found: {path}")
```

### UserWarning: no reservoir states found

**Problem:** Trying to load states that weren't saved.

**Solution:** Save with `include_states=True` or don't use `load_states=True`.

```python
# Save with states
model.save('model.pt', include_states=True)

# Load with states
model.load('model.pt', load_states=True)
```

## See Also

- [Model Composition](phase3_summary.md)
- [Premade Models](premade_models.md)
- [Training Example](../examples/07_save_load_models.py)
- [PyTorch Save/Load Documentation](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
