# Trainable Weights Feature

## Overview

Both `ReservoirLayer` and `ReadoutLayer` now support a `trainable` parameter that controls whether their weights require gradients and can be trained using standard PyTorch optimizers.

## Motivation

By default, ESN layers have **fixed, randomly initialized weights** that don't require gradients. This provides:

- **Better performance**: `torch.no_grad()` context reduces memory and computation
- **Classical ESN behavior**: Reservoir weights are typically fixed
- **Fast inference**: No gradient tracking overhead

However, for research and experimentation, you may want to:

- Fine-tune reservoir weights end-to-end
- Train readout layers with backpropagation (alternative to ridge regression)
- Explore hybrid training approaches

## API

### ReservoirLayer

```python
reservoir = ReservoirLayer(
    reservoir_size=100,
    feedback_size=10,
    trainable=False,  # Default: weights frozen, no gradients
)

# Or trainable
reservoir_trainable = ReservoirLayer(
    reservoir_size=100,
    feedback_size=10,
    trainable=True,  # Weights require gradients
)
```

### ReadoutLayer

```python
readout = ReadoutLayer(
    in_features=100,
    out_features=5,
    trainable=False,  # Default: weights frozen
)

# Or trainable
readout_trainable = ReadoutLayer(
    in_features=100,
    out_features=5,
    trainable=True,  # Weights require gradients
)
```

## Behavior

### Non-Trainable (Default: `trainable=False`)

- **Weights frozen**: `requires_grad=False` for all parameters
- **No gradient tracking**: Forward pass wrapped in `torch.no_grad()`
- **Better performance**: Reduced memory and computation
- **Classical ESN**: Matches traditional reservoir computing

```python
reservoir = ReservoirLayer(100, feedback_size=10)  # trainable=False by default

# Check parameters
for p in reservoir.parameters():
    print(p.requires_grad)  # False

# Forward pass - no gradients tracked
output = reservoir(feedback)  # Runs in torch.no_grad() context
```

### Trainable (`trainable=True`)

- **Weights trainable**: `requires_grad=True` for all parameters
- **Gradient tracking**: Forward pass allows gradient computation
- **Standard PyTorch training**: Works with any optimizer
- **End-to-end learning**: Can backpropagate through entire model

```python
reservoir = ReservoirLayer(100, feedback_size=10, trainable=True)

# Check parameters
for p in reservoir.parameters():
    print(p.requires_grad)  # True

# Forward pass - gradients tracked
output = reservoir(feedback)
loss = criterion(output, target)
loss.backward()  # Gradients flow through reservoir
```

## Training Examples

### Example 1: Manual Model

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch_rc.layers import ReservoirLayer, ReadoutLayer

# Create trainable model
reservoir = ReservoirLayer(100, feedback_size=10, trainable=True)
readout = ReadoutLayer(in_features=100, out_features=5, trainable=True)

# Setup training
optimizer = optim.Adam(
    list(reservoir.parameters()) + list(readout.parameters()),
    lr=0.01
)
criterion = nn.MSELoss()

# Training loop
for epoch in range(100):
    optimizer.zero_grad()

    # Forward pass
    res_out = reservoir(feedback)
    predictions = readout(res_out)

    # Compute loss and backprop
    loss = criterion(predictions, targets)
    loss.backward()
    optimizer.step()

    # Reset reservoir state for next epoch
    reservoir.reset_state()
```

### Example 2: ModelBuilder

```python
from torch_rc.composition import ModelBuilder

# Build trainable model
builder = ModelBuilder()
feedback = builder.input("feedback")

reservoir = builder.add(
    ReservoirLayer(100, feedback_size=10, trainable=True),
    inputs=feedback
)
readout = builder.add(
    ReadoutLayer(in_features=100, out_features=5, trainable=True, name="output"),
    inputs=reservoir
)

model = builder.build(outputs=readout)

# Train with standard PyTorch
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(100):
    optimizer.zero_grad()
    predictions = model({"feedback": feedback_data})
    loss = criterion(predictions, targets)
    loss.backward()
    optimizer.step()
    model.reset_reservoirs()
```

### Example 3: Mixed Trainable/Frozen

```python
# Frozen reservoir, trainable readout
# (Common pattern: fix reservoir, train only readout)
reservoir = ReservoirLayer(100, feedback_size=10, trainable=False)
readout = ReadoutLayer(in_features=100, out_features=5, trainable=True)

# Only readout parameters in optimizer
optimizer = optim.Adam(readout.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()

    # Reservoir forward (no gradients)
    with torch.no_grad():  # Explicit, but not necessary (already in no_grad)
        res_out = reservoir(feedback)

    # Readout forward (with gradients)
    predictions = readout(res_out)

    loss = criterion(predictions, targets)
    loss.backward()  # Only readout weights updated
    optimizer.step()

    reservoir.reset_state()
```

## Performance Implications

### Non-Trainable (Default)

**Advantages:**

- ✅ Faster forward pass (no gradient tracking)
- ✅ Lower memory usage
- ✅ Suitable for large reservoirs
- ✅ Classical ESN behavior

**Use when:**

- Using classical ESN training (ridge regression)
- Reservoir weights are fixed by design
- Performance is critical
- Working with very large reservoirs

### Trainable

**Advantages:**

- ✅ End-to-end learning
- ✅ Can fine-tune reservoir dynamics
- ✅ Standard PyTorch workflow
- ✅ Research flexibility

**Disadvantages:**

- ⚠️ Slower (gradient tracking overhead)
- ⚠️ Higher memory usage
- ⚠️ May require careful hyperparameter tuning

**Use when:**

- Exploring end-to-end training
- Fine-tuning pre-initialized weights
- Research on trainable reservoirs
- Small to medium reservoir sizes

## Implementation Details

### Internal Mechanism

When `trainable=False`:

1. `_freeze_weights()` called in `__init__`
2. Sets `requires_grad=False` for all parameters
3. `forward()` wraps `_forward_impl()` in `torch.no_grad()`

When `trainable=True`:

1. Parameters keep default `requires_grad=True`
2. `forward()` calls `_forward_impl()` directly (gradients tracked)

### Code Structure

```python
class ReservoirLayer(TorchRCModule):
    def __init__(self, ..., trainable=False):
        ...
        self.trainable = trainable
        self._initialize_weights()

        if not self.trainable:
            self._freeze_weights()

    def _freeze_weights(self):
        """Freeze all weights."""
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, feedback, *driving_inputs):
        """Forward with conditional gradient tracking."""
        if not self.trainable:
            with torch.no_grad():
                return self._forward_impl(feedback, *driving_inputs)
        else:
            return self._forward_impl(feedback, *driving_inputs)

    def _forward_impl(self, feedback, *driving_inputs):
        """Actual forward computation."""
        # ... reservoir dynamics ...
```

## Testing

Comprehensive tests in `tests/test_trainable.py`:

1. **Manual model training**: Verifies trainable layers work with PyTorch optimizers
2. **Builder model training**: Verifies ModelBuilder-created models are trainable
3. **Comparison**: Manual vs Builder models achieve similar performance
4. **Frozen weights**: Non-trainable layers have frozen parameters
5. **Mixed models**: Trainable readout + frozen reservoir
6. **GPU training**: Trainable models work on CUDA

All tests verify:

- Parameters have correct `requires_grad` status
- Loss decreases during training
- Gradients flow correctly
- Models work on CPU and GPU

## Best Practices

1. **Default to non-trainable** for classical ESN workflows
2. **Use trainable** for research and experimentation
3. **Mixed approach**: Frozen reservoir + trainable readout is common
4. **Reset states**: Always call `reset_state()` or `reset_reservoirs()` between epochs
5. **Learning rates**: Trainable reservoirs may need lower learning rates than readouts
6. **Regularization**: Consider weight decay for trainable reservoirs

## Future Work

In Phase 4, we'll add:

- **Ridge regression training**: Classical ESN readout fitting
- **Hybrid training**: Combine backprop and ridge regression
- **ESNTrainer**: High-level training API

The `trainable` parameter provides flexibility for both classical and modern approaches!
