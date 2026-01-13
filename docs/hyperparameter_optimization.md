# Hyperparameter Optimization

resdag provides an Optuna-based hyperparameter optimization (HPO) system designed specifically for Echo State Networks. It supports specialized loss functions for chaotic time series forecasting and integrates seamlessly with the library's training infrastructure.

## Installation

The HPO module requires `optuna` as an optional dependency. Install it with:

```bash
# Install resdag with HPO support
pip install resdag[hpo]

# Or install optuna separately
pip install optuna
```

Note: The loss functions (`LOSSES`, `get_loss`, etc.) are always available without optuna. Only `run_hpo` and related utilities require optuna.

## Quick Start

```python
from resdag.hpo import run_hpo, get_study_summary
from resdag.models import ott_esn

# 1. Define how to create a model
def model_creator(reservoir_size, spectral_radius):
    return ott_esn(
        reservoir_size=reservoir_size,
        feedback_size=3,
        output_size=3,
        spectral_radius=spectral_radius,
    )

# 2. Define what hyperparameters to search
def search_space(trial):
    return {
        "reservoir_size": trial.suggest_int("reservoir_size", 100, 500, step=50),
        "spectral_radius": trial.suggest_float("spectral_radius", 0.5, 1.5),
    }

# 3. Define how to load data
def data_loader(trial):
    # Your data loading logic here
    return {
        "warmup": warmup_tensor,
        "train": train_tensor,
        "target": target_tensor,
        "f_warmup": forecast_warmup_tensor,
        "val": validation_tensor,
    }

# 4. Run optimization
study = run_hpo(
    model_creator=model_creator,
    search_space=search_space,
    data_loader=data_loader,
    n_trials=100,
)

# 5. Analyze results
print(get_study_summary(study))
print(f"Best params: {study.best_params}")
```

## Core Concepts

### The Three Callbacks

The HPO system uses three user-defined callback functions:

#### model_creator

Creates a fresh model instance for each trial. Must accept all hyperparameters from `search_space` as keyword arguments.

```python
def model_creator(reservoir_size: int, spectral_radius: float, **kwargs):
    """Create a model with the given hyperparameters."""
    return ott_esn(
        reservoir_size=reservoir_size,
        feedback_size=3,
        output_size=3,
        spectral_radius=spectral_radius,
        topology=("random", {"density": 0.1}),
    )
```

#### search_space

Defines the hyperparameter search space using Optuna's `trial.suggest_*` methods. Returns a dictionary that will be passed to `model_creator`.

```python
def search_space(trial):
    """Define hyperparameter ranges."""
    return {
        "reservoir_size": trial.suggest_int("reservoir_size", 100, 500, step=50),
        "spectral_radius": trial.suggest_float("spectral_radius", 0.5, 1.5),
        "leak_rate": trial.suggest_float("leak_rate", 0.1, 1.0),
    }
```

Available suggest methods:

- `trial.suggest_int(name, low, high, step=1)` - Integer values
- `trial.suggest_float(name, low, high, log=False)` - Float values
- `trial.suggest_categorical(name, choices)` - Categorical choices

#### data_loader

Loads and prepares training/validation data. Must return a dictionary with specific keys:

```python
def data_loader(trial):
    """Load and prepare data."""
    # Load your data (e.g., using load_and_prepare)
    warmup, train, target, f_warmup, val = load_and_prepare(
        data_files,
        warmup_steps=500,
        train_steps=5000,
        val_steps=500,
    )

    return {
        "warmup": warmup,      # (B, warmup_steps, D)
        "train": train,        # (B, train_steps, D)
        "target": target,      # (B, train_steps, D)
        "f_warmup": f_warmup,  # (B, warmup_steps, D)
        "val": val,            # (B, val_steps, D)
    }
```

Required keys:

- `warmup`: Teacher-forced warmup for training
- `train`: Training input sequence
- `target`: Training targets (usually `train` shifted by 1)
- `f_warmup`: Teacher-forced warmup before forecasting
- `val`: Ground truth for validation

## Loss Functions

resdag provides specialized loss functions for evaluating multi-step forecasts:

### Expected Forecast Horizon (EFH) - Default

```python
study = run_hpo(..., loss="efh")
```

A smooth, differentiable approximation of the forecast horizon. Computes the expected number of time steps before the error exceeds a threshold. **Recommended for chaotic systems.**

Parameters:

- `threshold` (default=0.2): Error threshold for "good" predictions
- `softness` (default=0.02): Controls smoothness of threshold transition
- `metric` (default="nrmse"): Error metric ("rmse", "mse", "mae", "nrmse")

### Forecast Horizon

```python
study = run_hpo(..., loss="horizon")
```

Negative log of the contiguous valid forecast horizon. Counts consecutive steps where error stays below threshold.

### Lyapunov-weighted

```python
study = run_hpo(..., loss="lyap", loss_params={"lle": 0.9, "dt": 0.02})
```

Weights errors by exponential decay based on the Lyapunov exponent. Accounts for the natural exponential divergence in chaotic systems.

Parameters:

- `lle`: Largest Lyapunov exponent of the system
- `dt`: Time step size

### Standard Loss

```python
study = run_hpo(..., loss="standard")
```

Simple mean of geometric mean errors. Good baseline for any system.

### Discounted RMSE

```python
study = run_hpo(..., loss="discounted", loss_params={"half_life": 64})
```

Time-discounted error with exponential half-life. Emphasizes early predictions.

Parameters:

- `half_life`: Steps until discount factor reaches 0.5

### Custom Loss Functions

You can define custom loss functions:

```python
def my_loss(y_true, y_pred, my_param=1.0):
    """Custom loss function.

    Args:
        y_true: Ground truth, shape (B, T, D)
        y_pred: Predictions, shape (B, T, D)
        my_param: Custom parameter

    Returns:
        float: Loss value to minimize
    """
    errors = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=(0, 2)))
    return float(np.mean(errors) * my_param)

study = run_hpo(..., loss=my_loss)
```

## Advanced Usage

### Study Persistence

Save studies to SQLite for resumable optimization:

```python
study = run_hpo(
    ...,
    storage="sqlite:///my_study.db",
    study_name="lorenz_optimization",
    n_trials=100,
)
```

Resume an interrupted study:

```python
# This automatically loads existing trials and continues
study = run_hpo(
    ...,
    storage="sqlite:///my_study.db",
    study_name="lorenz_optimization",
    n_trials=200,  # Target 200 total
)
```

### Parallel Execution

Run trials in parallel:

```python
study = run_hpo(
    ...,
    n_workers=4,  # Use 4 parallel workers
)
```

Note: Each worker creates independent models and loads data separately.

### Conditional Search Spaces

Create complex search spaces with dependencies:

```python
def search_space(trial):
    topology = trial.suggest_categorical("topology", ["random", "small_world"])

    if topology == "random":
        density = trial.suggest_float("density", 0.05, 0.3)
        topology_config = ("random", {"density": density})
    else:
        p = trial.suggest_float("rewiring_prob", 0.1, 0.5)
        k = trial.suggest_int("neighbors", 2, 8, step=2)
        topology_config = ("small_world", {"p": p, "k": k})

    return {
        "reservoir_size": trial.suggest_int("reservoir_size", 100, 500, step=50),
        "topology_config": topology_config,
    }
```

### Input-driven Models

For models with driver inputs, provide driver data for each phase:

```python
def data_loader(trial):
    return {
        # Feedback data (main signal)
        "warmup": feedback_warmup,
        "train": feedback_train,
        "target": target,
        "f_warmup": forecast_feedback_warmup,
        "val": val,
        # Driver inputs for training
        "warmup_driver": driver_warmup,       # Training warmup phase
        "train_driver": driver_train,          # Training phase
        # Driver inputs for forecasting
        "f_warmup_driver": driver_f_warmup,   # Forecast warmup phase
        "forecast_driver": forecast_driver,   # Autoregressive phase
    }

study = run_hpo(
    ...,
    drivers_keys=["driver"],  # Specifies which drivers to use
)
```

The naming convention for driver keys:

- `warmup_{driver}`: Driver during training warmup
- `train_{driver}`: Driver during training
- `f_warmup_{driver}`: Driver during forecast warmup
- `forecast_{driver}`: Driver during autoregressive forecasting

For models with multiple drivers, use separate keys for each (e.g., `driver0`, `driver1`):

```python
study = run_hpo(
    ...,
    drivers_keys=["driver0", "driver1"],  # Multiple drivers
)
```

### Multi-Readout Models

For models with multiple readouts:

```python
def data_loader(trial):
    return {
        "warmup": warmup,
        "train": train,
        "target": main_target,      # Used for specified targets_key
        "target_aux": aux_target,   # Additional targets if needed
        "f_warmup": f_warmup,
        "val": val,
    }

study = run_hpo(
    ...,
    targets_key="main_readout",  # Name of the readout to train
)
```

## Best Practices

### Choosing Loss Functions

| System Type            | Recommended Loss | Why                                     |
| ---------------------- | ---------------- | --------------------------------------- |
| Chaotic (Lorenz, etc.) | `"efh"`          | Smooth optimization, handles divergence |
| Chaotic with known LLE | `"lyap"`         | Physics-informed weighting              |
| Periodic/Stable        | `"standard"`     | Simple, effective                       |
| Short-term focus       | `"discounted"`   | Emphasizes early accuracy               |

### Search Space Design

1. **Start broad, then narrow**: Begin with wide ranges, then focus on promising regions
2. **Use log-scale for regularization**: `trial.suggest_float("alpha", 1e-8, 1e-4, log=True)`
3. **Step sizes for discrete params**: `trial.suggest_int("size", 100, 500, step=50)`
4. **Categorical for architecture choices**: `trial.suggest_categorical("activation", ["tanh", "relu"])`

### Data Preparation

1. **Use multiple trajectories** for robust training
2. **Normalize globally** across all batches
3. **Sufficient warmup** (typically 500-1000 steps for chaotic systems)
4. **Match forecast horizon** to validation length

### Memory Management

The HPO system automatically cleans up between trials. For large models:

```python
study = run_hpo(
    ...,
    catch_exceptions=True,   # Don't fail on OOM
    penalty_value=1e10,      # Penalize failed trials
)
```

## API Reference

### run_hpo

```python
def run_hpo(
    model_creator: Callable[..., ESNModel],
    search_space: Callable[[Trial], dict[str, Any]],
    data_loader: Callable[[Trial], dict[str, Any]],
    n_trials: int,
    loss: str | LossProtocol = "efh",
    loss_params: dict[str, Any] | None = None,
    targets_key: str = "output",
    drivers_keys: list[str] | None = None,
    study_name: str | None = None,
    storage: str | None = None,
    sampler: BaseSampler | None = None,
    seed: int | None = None,
    n_workers: int = 1,
    verbosity: int = 1,
    catch_exceptions: bool = True,
    penalty_value: float = 1e10,
) -> optuna.Study:
```

### get_study_summary

```python
def get_study_summary(study: optuna.Study, top_n: int = 5) -> str:
```

Returns a formatted summary of the study results.

### LOSSES

```python
LOSSES: dict[str, LossProtocol] = {
    "efh": expected_forecast_horizon,
    "horizon": forecast_horizon,
    "lyap": lyapunov_weighted,
    "standard": standard_loss,
    "discounted": discounted_rmse,
}
```

### get_loss

```python
def get_loss(key_or_callable: str | LossProtocol) -> LossProtocol:
```

Get a loss function by name or pass through a custom callable.

## Examples

See `examples/10_hpo.py` for complete working examples including:

1. Basic HPO with synthetic data
2. Comparing different loss functions
3. Study persistence and resumption
4. Parallel execution
5. Advanced conditional search spaces
6. Using real chaotic data
7. Custom loss functions
