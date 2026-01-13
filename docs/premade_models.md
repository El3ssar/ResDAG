# Premade Model Architectures

`resdag` provides four premade ESN architectures that can be used directly or customized for specific tasks. These architectures follow best practices from reservoir computing literature while providing a simple, config-based API for customization.

## Available Architectures

### 1. Classic ESN (`classic_esn`)

The traditional Echo State Network architecture where the input is concatenated with the reservoir output before passing to the readout layer.

**Architecture:**

```
Input -> Reservoir -> Concatenate(Input, Reservoir) -> Readout
```

**Usage:**

```python
from resdag.models import classic_esn

# Simple usage
model = classic_esn(
    reservoir_size=100,
    input_size=1,
    output_size=1
)

# With custom configuration
model = classic_esn(
    reservoir_size=100,
    input_size=1,
    output_size=1,
    reservoir_config={
        'topology': 'erdos_renyi',
        'spectral_radius': 0.9,
        'leak_rate': 0.3,
        'input_scaling': 0.5,
    },
    readout_config={
        'max_iter': 200,
        'tol': 1e-6,
    }
)
```

**Key Features:**

- Input concatenation provides readout with both raw input and reservoir states
- Good general-purpose architecture for many tasks
- Readout sees `reservoir_size + input_size` features

### 2. Ott's ESN (`ott_esn`)

An architecture proposed by Edward Ott that augments reservoir states by squaring even-indexed units, helping capture higher-order dynamics.

**Architecture:**

```
Input -> Reservoir -> SelectiveExponentiation -> Concatenate(Input, Augmented) -> Readout
```

**Usage:**

```python
from resdag.models import ott_esn

model = ott_esn(
    reservoir_size=200,
    input_size=1,
    output_size=1,
    reservoir_config={
        'topology': 'watts_strogatz',
        'spectral_radius': 0.95,
        'leak_rate': 0.2,
    }
)
```

**Key Features:**

- State augmentation via squaring even-indexed units
- Provides polynomial feature expansion
- Useful for nonlinear tasks
- Reference: Ott et al., Physical Review Letters, vol. 120, no. 2, p. 024102, Jan. 2018

### 3. Headless ESN (`headless_esn`)

A reservoir-only model with no readout layer, useful for analyzing reservoir dynamics, feature extraction, or state space properties.

**Architecture:**

```
Input -> Reservoir
```

**Usage:**

```python
from resdag.models import headless_esn

model = headless_esn(
    reservoir_size=100,
    input_size=2,
    reservoir_config={
        'topology': 'ring_chord',
        'spectral_radius': 1.2,  # Chaotic regime
        'leak_rate': 0.5,
    }
)

# Get reservoir states directly
x = torch.randn(4, 50, 2)
states = model({"input": x})  # Shape: (4, 50, 100)

# Analyze dynamics
print(f"Mean: {states.mean():.3f}, Std: {states.std():.3f}")
```

**Key Features:**

- Direct access to reservoir states
- Useful for reservoir analysis and debugging
- Can be used for feature extraction
- No readout layer overhead

### 4. Linear ESN (`linear_esn`)

A reservoir with linear (identity) activation, equivalent to a linear dynamical system. Useful for baseline comparison or studying linear dynamics.

**Architecture:**

```
Input -> Reservoir (identity activation)
```

**Usage:**

```python
from resdag.models import linear_esn

model = linear_esn(
    reservoir_size=100,
    input_size=1,
    reservoir_config={
        'topology': 'cycle_jumps',
        'spectral_radius': 0.95,
        'leak_rate': 1.0,  # No leaking for pure linear system
    }
)
```

**Key Features:**

- Linear dynamics (identity activation)
- Useful as baseline for comparison
- Can be analyzed with linear systems theory
- No nonlinearity in reservoir

## Configuration System

All premade models use a config dict approach that provides:

1. **Explicit Configuration**: Users see exactly what they're configuring
2. **Partial Configs**: Unspecified parameters use layer defaults
3. **Architecture Overrides**: Some parameters are managed by the architecture (e.g., `feedback_size`)
4. **Forward Compatible**: New layer parameters won't break existing code

### Precedence Rules

When parameters can be specified in multiple places:

```python
model = classic_esn(
    reservoir_size=100,          # Function parameter (convenience)
    reservoir_config={
        'reservoir_size': 200    # Config dict (takes precedence)
    }
)
# Result: reservoir_size = 200
```

**Precedence:** `config_dict` > `function_params` > `layer_defaults`

### Architecture-Managed Parameters

Some parameters are automatically set by the architecture and should not be specified in configs:

- **`feedback_size`**: Always set to `input_size` (architecture requirement)
- **`input_size`**: Set to 0 for these architectures (no driving input)
- **`in_features` (readout)**: Calculated based on architecture (e.g., `reservoir_size + input_size`)
- **`activation` (linear_esn)**: Forced to `'identity'`

## Available Configuration Options

### Reservoir Config

All parameters from `ReservoirLayer`:

```python
reservoir_config = {
    'reservoir_size': 100,        # Number of reservoir units
    'topology': 'erdos_renyi',    # Graph topology for recurrent weights
    'spectral_radius': 0.9,       # Spectral radius scaling
    'leak_rate': 0.3,             # Leaky integration rate
    'input_scaling': 1.0,         # Input weight scaling
    'feedback_scaling': 1.0,      # Feedback weight scaling
    'bias': True,                 # Include bias term
    'trainable': False,           # Whether weights require gradients
    'input_initializer': None,    # Custom input weight initializer
    'feedback_initializer': None, # Custom feedback weight initializer
}
```

### Readout Config

All parameters from `CGReadoutLayer`:

```python
readout_config = {
    'max_iter': 100,    # Max CG iterations
    'tol': 1e-5,        # Convergence tolerance
    'bias': True,       # Include bias term
    'trainable': False, # Whether weights require gradients
}
```

## GPU Support

All premade models support GPU:

```python
model = classic_esn(100, 1, 1).cuda()
x = torch.randn(4, 50, 1).cuda()
output = model({"input": x})
```

## Model Comparison Example

```python
import torch
from resdag.models import classic_esn, ott_esn, headless_esn, linear_esn

# Same input for all models
x = torch.randn(4, 50, 2)

# Classic ESN
classic = classic_esn(100, 2, 1)
out_classic = classic({"input": x})

# Ott's ESN
ott = ott_esn(100, 2, 1)
out_ott = ott({"input": x})

# Headless (no readout)
headless = headless_esn(100, 2)
states = headless({"input": x})

# Linear (baseline)
linear = linear_esn(100, 2)
linear_states = linear({"input": x})

print(f"Classic output: {out_classic.shape}")
print(f"Ott output: {out_ott.shape}")
print(f"Headless states: {states.shape}")
print(f"Linear states: {linear_states.shape}")
```

## Best Practices

1. **Start Simple**: Use default configs first, then customize as needed
2. **Topology Selection**: Choose topology based on task (e.g., `erdos_renyi` for general use, `ring_chord` for sequential tasks)
3. **Spectral Radius**: Start with 0.9, increase for longer memory, decrease for stability
4. **Leak Rate**: Lower values (0.1-0.3) for longer memory, higher (0.7-0.9) for faster dynamics
5. **Baseline Comparison**: Use `linear_esn` or `headless_esn` to understand reservoir contribution

## See Also

- [Reservoir Layer Documentation](../README.md#reservoir-layer)
- [Topology System](topology_system.md)
- [Model Composition](phase3_summary.md)
- [Example Script](../examples/06_premade_models.py)
