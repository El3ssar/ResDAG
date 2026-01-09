# Forecasting with ESN Models

`torch_rc` provides a comprehensive forecasting API for autoregressive prediction with Echo State Networks. The forecasting process consists of two phases: teacher-forced warmup and closed-loop generation.

## Quick Start

```python
from torch_rc.models import classic_esn
import torch

# Create model
model = classic_esn(100, 1, 1)

# Prepare data
warmup = torch.randn(4, 100, 1)  # Ground truth for initialization

# Forecast
predictions = model.forecast(
    warmup_feedback=warmup,
    forecast_steps=50
)  # Returns: (4, 50, 1)
```

## Two-Phase Forecasting

### Phase 1: Teacher-Forced Warmup

The warmup phase initializes reservoir states using ground truth data:

```python
# Warmup with 100 steps of ground truth
warmup = y_true[:, :100, :]

# Model processes ground truth sequentially
# Reservoir states evolve based on actual data
# This provides a good initial condition for forecasting
```

**Purpose:**
- Initialize reservoir dynamics properly
- Avoid cold-start problems
- Capture recent history in reservoir states

### Phase 2: Closed-Loop Forecast

The forecast phase generates predictions autoregressively:

```python
# Generate 50 steps into the future
predictions = model.forecast(
    warmup_feedback=warmup,
    forecast_steps=50
)

# Each prediction becomes the next input (closed-loop)
# y_t+1 = f(y_t, states_t)
```

**Behavior:**
- Uses previous prediction as next input
- Fully autoregressive (no teacher forcing)
- Errors can accumulate over long horizons

## API Reference

### `model.forecast(...)`

```python
def forecast(
    self,
    # Warmup phase
    warmup_feedback: torch.Tensor,
    warmup_driving: Optional[Dict[str, torch.Tensor]] = None,
    
    # Forecast phase
    forecast_steps: int = 1,
    forecast_driving: Optional[Dict[str, torch.Tensor]] = None,
    
    # Configuration
    feedback_input: str = 'input',
    output_key: Optional[str] = None,
    driving_input_map: Optional[Dict[str, str]] = None,
    forecast_initial_feedback: Optional[torch.Tensor] = None,
    
    # Output options
    return_warmup: bool = False,
    return_state_history: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
```

#### Parameters

**Warmup Phase:**
- `warmup_feedback`: Ground truth sequence, shape `(batch, warmup_steps, feedback_dim)`
- `warmup_driving`: Optional dict mapping reservoir names to driving signals
  - Each value: `(batch, warmup_steps, driving_dim)`
  - Example: `{'reservoir': weather_past}`

**Forecast Phase:**
- `forecast_steps`: Number of steps to predict
- `forecast_driving`: Optional dict of future driving signals
  - Each value: `(batch, forecast_steps, driving_dim)`
  - Example: `{'reservoir': weather_future}`

**Configuration:**
- `feedback_input`: Input node name for feedback (default: `'input'`)
- `output_key`: Which output to feed back (required for multi-output models)
- `driving_input_map`: Explicit mapping from reservoir names to input nodes
  - Overrides convention-based naming
  - Example: `{'reservoir': 'external_input'}`
- `forecast_initial_feedback`: Custom initial feedback for forecast
  - If `None`, uses last warmup prediction
  - Shape: `(batch, 1, feedback_dim)`

**Output Options:**
- `return_warmup`: If `True`, include warmup predictions in output
- `return_state_history`: If `True`, return reservoir state trajectories
  - Warning: Memory-intensive for long sequences

#### Returns

**Without state history:**
```python
predictions: torch.Tensor  # (batch, forecast_steps, output_dim)
# or (batch, warmup_steps + forecast_steps, output_dim) if return_warmup=True
```

**With state history:**
```python
(predictions, state_history): Tuple[torch.Tensor, Dict[str, torch.Tensor]]
# state_history = {'reservoir_name': (batch, total_steps, reservoir_size), ...}
```

## Common Use Cases

### 1. Simple Time Series Forecasting

```python
model = classic_esn(100, 1, 1)

# Historical data
y_past = torch.randn(4, 100, 1)

# Forecast 50 steps
y_future = model.forecast(
    warmup_feedback=y_past,
    forecast_steps=50
)
```

### 2. Forecasting with Exogenous Variables

Requires ModelBuilder (premade models don't support driving inputs):

```python
from torch_rc.composition import ModelBuilder
from torch_rc.layers import ReservoirLayer
from torch_rc.layers.readouts import CGReadoutLayer

# Build model with driving input
builder = ModelBuilder()
feedback = builder.input('input')
driving = builder.input('reservoir_driving')  # Convention name
reservoir = builder.add(
    ReservoirLayer(100, 1, 5),
    inputs=[feedback, driving],
    name='reservoir'
)
readout = builder.add(CGReadoutLayer(100, 1), inputs=reservoir)
model = builder.build(outputs=readout)

# Forecast with known future drivers (e.g., weather forecast)
predictions = model.forecast(
    warmup_feedback=y_past,
    warmup_driving={'reservoir': weather_past},
    forecast_steps=50,
    forecast_driving={'reservoir': weather_future}
)
```

### 3. Multi-Horizon Forecasting

```python
# Forecast different horizons
for horizon in [10, 50, 100]:
    model.reset_reservoirs()  # Reset between forecasts
    predictions = model.forecast(
        warmup_feedback=warmup,
        forecast_steps=horizon
    )
    print(f"Horizon {horizon}: MSE = {compute_mse(predictions, ground_truth)}")
```

### 4. State Analysis

Track reservoir dynamics during forecasting:

```python
predictions, states = model.forecast(
    warmup_feedback=warmup,
    forecast_steps=50,
    return_state_history=True
)

# Analyze state evolution
reservoir_states = states['reservoir']  # (batch, 150, 100)

# Compute Lyapunov exponents, entropy, etc.
state_norms = torch.norm(reservoir_states, dim=2)
print(f"Average state norm: {state_norms.mean()}")
```

### 5. Visualization

Include warmup for complete trajectory:

```python
full_trajectory = model.forecast(
    warmup_feedback=warmup,
    forecast_steps=50,
    return_warmup=True
)  # (batch, 150, 1) - warmup + forecast

# Plot
import matplotlib.pyplot as plt
plt.plot(full_trajectory[0, :, 0].numpy())
plt.axvline(100, color='r', linestyle='--', label='Forecast start')
plt.legend()
```

### 6. Multiple Reservoirs with Different Drivers

```python
# Build model with two reservoirs
builder = ModelBuilder()
feedback = builder.input('input')
driving1 = builder.input('reservoir1_driving')
driving2 = builder.input('reservoir2_driving')

res1 = builder.add(ReservoirLayer(50, 1, 3), inputs=[feedback, driving1], name='reservoir1')
res2 = builder.add(ReservoirLayer(60, 1, 5), inputs=[feedback, driving2], name='reservoir2')

# ... concat and readout ...

# Forecast with different drivers
predictions = model.forecast(
    warmup_feedback=y_past,
    warmup_driving={
        'reservoir1': local_features_past,
        'reservoir2': global_features_past,
    },
    forecast_steps=50,
    forecast_driving={
        'reservoir1': local_features_future,
        'reservoir2': global_features_future,
    }
)
```

## Driving Input Conventions

When using driving inputs, `torch_rc` follows naming conventions:

### Convention-Based Naming

For a reservoir named `'reservoir'`, the forecast method looks for:
1. `'{reservoir_name}_driving'` (e.g., `'reservoir_driving'`)
2. `'{reservoir_name}_input'` (e.g., `'reservoir_input'`)

```python
# This works automatically:
builder.input('reservoir_driving')  # Matches convention
reservoir = builder.add(..., name='reservoir')

# Forecast will find it:
model.forecast(
    ...,
    warmup_driving={'reservoir': data}  # Automatically maps to 'reservoir_driving'
)
```

### Explicit Mapping

Override conventions with `driving_input_map`:

```python
# Non-standard input name
builder.input('external_weather')

# Explicit mapping
model.forecast(
    ...,
    warmup_driving={'reservoir': weather_data},
    driving_input_map={'reservoir': 'external_weather'}
)
```

### Premade Models

**Important:** Premade models (`classic_esn`, `ott_esn`, etc.) don't support driving inputs by design (they follow specific papers). Use `ModelBuilder` for models with driving inputs.

## Best Practices

### 1. Warmup Length

Choose warmup length based on reservoir dynamics:

```python
# Rule of thumb: 2-5x the reservoir's characteristic timescale
# For leak_rate=0.1, use ~20-50 steps
# For leak_rate=0.9, use ~5-10 steps

warmup_steps = int(5 / leak_rate)  # Rough estimate
```

### 2. Reset Between Forecasts

Always reset when forecasting multiple sequences:

```python
for sequence in test_sequences:
    model.reset_reservoirs()  # Important!
    predictions = model.forecast(sequence, forecast_steps=50)
```

### 3. Batch Processing

Process multiple sequences in parallel:

```python
# Stack sequences into batch
warmup_batch = torch.stack([seq1, seq2, seq3, seq4])  # (4, 100, 1)

# Single forecast call
predictions = model.forecast(warmup_batch, forecast_steps=50)  # (4, 50, 1)
```

### 4. Error Accumulation

Closed-loop forecasting accumulates errors:

```python
# Short horizon: Usually accurate
predictions_short = model.forecast(warmup, forecast_steps=10)

# Long horizon: Errors accumulate
predictions_long = model.forecast(warmup, forecast_steps=200)

# Consider: Multi-step ahead prediction, ensemble methods, or periodic re-initialization
```

### 5. State History Memory

Use `return_state_history=True` sparingly:

```python
# Memory usage: batch * total_steps * sum(reservoir_sizes) * 4 bytes
# For batch=32, steps=1000, reservoir=500: ~64 MB per reservoir

# Only use for analysis, not production forecasting
predictions, states = model.forecast(
    ...,
    return_state_history=True  # Only when needed!
)
```

## Troubleshooting

### ValueError: Feedback input not found

**Problem:** Specified `feedback_input` doesn't exist.

**Solution:** Check model input names:
```python
print(model.input_names)  # ['input', 'reservoir_driving']
model.forecast(..., feedback_input='input')  # Must match
```

### ValueError: Driving input for reservoir not found

**Problem:** Driving input convention not followed.

**Solution:** Use correct naming or explicit mapping:
```python
# Option 1: Follow convention
builder.input('reservoir_driving')  # Matches 'reservoir'

# Option 2: Explicit mapping
model.forecast(
    ...,
    driving_input_map={'reservoir': 'actual_input_name'}
)
```

### ValueError: Model has multiple outputs

**Problem:** Multi-output model requires `output_key`.

**Solution:** Specify which output to feed back:
```python
print(model.output_names)  # ['output1', 'output2']
model.forecast(..., output_key='output1')
```

### Predictions Diverge Quickly

**Problem:** Forecast becomes unstable after few steps.

**Solutions:**
1. Increase warmup length
2. Adjust spectral radius (try 0.9-0.95)
3. Tune leak rate
4. Use more training data
5. Consider ensemble methods

### Memory Error with State History

**Problem:** `return_state_history=True` uses too much memory.

**Solution:** Don't track states, or reduce batch/sequence length:
```python
# Option 1: Don't track states
predictions = model.forecast(..., return_state_history=False)

# Option 2: Process in smaller batches
for batch in data_loader:
    predictions = model.forecast(batch, ...)
```

## See Also

- [Model Composition](phase3_summary.md)
- [Premade Models](premade_models.md)
- [Save/Load](save_load.md)
- [Example Script](../examples/08_forecasting.py)
