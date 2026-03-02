# CLAUDE.md — AI Assistant Guide for ResDAG

## Project Overview

`resdag` is a **PyTorch-native reservoir computing library** (v0.2.0) for building Echo State Networks (ESNs). It provides GPU-accelerated, modular components for reservoir computing research: stateful reservoir layers, graph-based topology initialization, algebraic readout training, model composition via `pytorch_symbolic`, and Optuna-based hyperparameter optimization.

- **Package name**: `resdag`
- **Author**: Daniel Estevez-Moya
- **License**: MIT
- **Homepage**: https://github.com/El3ssar/resdag
- **Python**: >=3.11 (runtime); classifiers include 3.9–3.12

---

## Repository Layout

```
resdag/
├── src/resdag/              # Main package (src layout)
│   ├── __init__.py          # Public API + lazy HPO imports; version string
│   ├── composition/
│   │   └── symbolic.py      # ESNModel (extends pytorch_symbolic.SymbolicModel)
│   ├── layers/
│   │   ├── reservoir.py     # ReservoirLayer — core stateful RNN
│   │   ├── readouts/
│   │   │   ├── base.py      # ReadoutLayer (abstract)
│   │   │   └── cg_readout.py # CGReadoutLayer — CG ridge regression
│   │   └── custom/          # Concatenate, SelectiveExponentiation, etc.
│   ├── init/
│   │   ├── topology/        # Graph topology registry + base classes
│   │   ├── input_feedback/  # Input/feedback weight initializer registry
│   │   ├── graphs/          # NetworkX graph generation functions (15+ types)
│   │   └── utils/           # resolve_topology(), resolve_initializer()
│   ├── training/
│   │   └── trainer.py       # ESNTrainer — algebraic readout fitting
│   ├── models/              # Premade architectures
│   │   ├── classic_esn.py
│   │   ├── ott_esn.py       # Ott state-augmented ESN (recommended for chaos)
│   │   ├── headless_esn.py
│   │   └── linear_esn.py
│   ├── hpo/                 # Optuna HPO integration (optional dep)
│   │   ├── run.py           # run_hpo()
│   │   ├── losses.py        # efh, horizon, lyap, standard, discounted
│   │   └── objective.py     # build_objective()
│   └── utils/
│       ├── data/            # load_file(), prepare_esn_data()
│       ├── states/          # esp_index()
│       └── general.py
├── tests/                   # Pytest test suite (mirrors src structure)
├── examples/                # Numbered example scripts (00–10)
├── pyproject.toml           # Build, dependencies, tool configs
├── uv.lock                  # Locked dependency tree
└── .github/workflows/
    └── release.yml          # Auto-release on version tags
```

---

## Development Setup

```bash
# Preferred: uv (faster)
uv sync --dev

# Alternative: pip
pip install -e ".[dev]"

# HPO extras (optional)
pip install -e ".[dev,hpo]"
# or: uv sync --extra hpo
```

**Runtime dependencies**: `torch>=2.9.1`, `numpy>=2.4.0`, `networkx>=3.0`, `pytorch-symbolic>=1.1.1`, `graphviz>=0.21`, `basedpyright>=1.37.1`, `scipy>=1.17.0`

**Dev dependencies**: `pytest`, `pytest-cov`, `black`, `ruff`, `mypy`, `optuna`

---

## Running Tests

```bash
# All tests (with coverage)
pytest

# Specific module
pytest tests/test_layers/test_reservoir.py

# Specific test function
pytest tests/test_layers/test_reservoir.py::test_forward_shape

# Without coverage (faster)
pytest --no-cov

# HTML coverage report
pytest --cov=resdag --cov-report=html
```

Test configuration is in `pyproject.toml` under `[tool.pytest.ini_options]`. Tests live in `tests/` and mirror the `src/resdag/` structure. Test files are named `test_*.py`, classes `Test*`, functions `test_*`.

---

## Code Quality

```bash
# Format (line length = 100)
black src/ tests/

# Lint (E, F, I, N, W rules; E501 ignored — handled by black)
ruff check src/ tests/

# Auto-fix lint issues
ruff check --fix src/ tests/

# Type checking
mypy src/
```

**Key formatting rules**:
- Line length: **100 characters** (black + ruff)
- Target: Python 3.9+ syntax (ruff target-version)
- `__init__.py` files: unused imports (`F401`) are allowed — they expose the public API

---

## Core Architecture Concepts

### Tensor Conventions

- **3D tensors**: `(batch, timesteps, features)` throughout
- **Feedback input**: always the first positional argument to `ReservoirLayer.forward()`
- **Driving inputs**: optional additional positional args to `forward()`
- Reservoir state shape: `(batch, reservoir_size)`

### ReservoirLayer (core component)

```python
from resdag.layers import ReservoirLayer

reservoir = ReservoirLayer(
    reservoir_size=500,        # Number of neurons
    feedback_size=3,           # Dim of feedback signal (required)
    input_size=5,              # Dim of driving input (optional)
    spectral_radius=0.9,       # Controls memory/stability
    leak_rate=1.0,             # 1.0 = no leaking (standard)
    activation="tanh",         # "tanh"|"relu"|"identity"|"sigmoid"
    topology="erdos_renyi",    # str | (name, params) tuple | TopologyInitializer | None
    feedback_initializer=None, # str | (name, params) tuple | InputFeedbackInitializer | None
    trainable=False,           # False = frozen weights (standard ESN)
)

# Forward: states of shape (batch, time, reservoir_size)
states = reservoir(feedback)                    # feedback-only
states = reservoir(feedback, driving_input)     # with driver
```

**State management** (critical — reservoir is stateful):
```python
reservoir.reset_state()            # Reset to None (lazy re-init on next forward)
reservoir.reset_state(batch_size=4) # Reset to zeros with explicit batch size
reservoir.get_state()              # Returns clone or None
reservoir.set_state(state_tensor)  # Restore a saved state
```

### ESNModel and Composition

Models are built with `pytorch_symbolic` functional API, then wrapped in `ESNModel`:

```python
import pytorch_symbolic as ps
from resdag import ESNModel, ReservoirLayer, CGReadoutLayer

inp = ps.Input((100, 3))                          # (seq_len, features)
reservoir = ReservoirLayer(200, feedback_size=3)(inp)
readout = CGReadoutLayer(200, 3, name="output")(reservoir)
model = ESNModel(inp, readout)
```

`ESNModel` extends `pytorch_symbolic.SymbolicModel` with:
- `reset_reservoirs()` — reset all reservoir states
- `warmup(*inputs)` — teacher-forced state synchronization
- `forecast(*warmup_inputs, horizon=N, ...)` — two-phase autoregressive forecasting
- `save(path)` / `load(path)` — model persistence
- `plot_model()` — architecture visualization

### CGReadoutLayer (algebraic training)

Readouts are **not trained by gradient descent** — they use ridge regression via Conjugate Gradient:

```python
from resdag.layers.readouts import CGReadoutLayer

readout = CGReadoutLayer(
    in_features=500,   # Reservoir size
    out_features=3,    # Output dimension
    alpha=1e-6,        # L2 regularization strength
    name="output",     # Used as key in targets dict (important!)
    max_iter=100,      # Max CG iterations
    tol=1e-5,          # Convergence tolerance
)
```

The `name` parameter is the key used when passing `targets` to `ESNTrainer.fit()`.

### ESNTrainer (training workflow)

```python
from resdag.training import ESNTrainer

trainer = ESNTrainer(model)
trainer.fit(
    warmup_inputs=(warmup_feedback,),           # Tuple — synchronize reservoir states
    train_inputs=(train_feedback,),             # Tuple — fit readout
    targets={"output": train_targets},          # Dict keyed by readout name
)
```

With driving inputs:
```python
trainer.fit(
    warmup_inputs=(warmup_feedback, warmup_driver),
    train_inputs=(train_feedback, train_driver),
    targets={"output": targets},
)
```

Training process:
1. Reset reservoir states
2. Warmup phase (teacher-forced forward pass)
3. Single forward pass with pre-hooks that fit each readout in topological order

### Topology System

**Three ways to specify topology** (used in `ReservoirLayer(topology=...)`):
```python
# 1. String name (uses registry defaults)
topology = "erdos_renyi"

# 2. Tuple (name + kwargs override)
topology = ("watts_strogatz", {"k": 6, "p": 0.3})

# 3. Pre-configured object
from resdag.init.topology import get_topology
topology = get_topology("barabasi_albert", m=3, seed=42)
```

**Available topologies** (all in `src/resdag/init/graphs/`):
`barabasi_albert`, `chord_dendrocycle`, `complete`, `connected_erdos_renyi`, `connected_watts_strogatz`, `dendrocycle`, `erdos_renyi`, `kleinberg_small_world`, `multi_cycle`, `newman_watts_strogatz`, `random`, `regular`, `ring_chord`, `simple_cycle_jumps`, `spectral_cascade`, `watts_strogatz`, `zero`

**Registering a new topology**:
```python
from resdag.init.topology import register_graph_topology
import networkx as nx

@register_graph_topology("my_graph", p=0.1, directed=True)
def my_graph(n: int, p: float = 0.1, directed: bool = True, seed=None) -> nx.DiGraph:
    # Must accept n as first arg, return nx.Graph or nx.DiGraph with weighted edges
    ...
```

### Input/Feedback Initializer System

Same three-way specification as topology (string | tuple | object):
```python
from resdag.init.input_feedback import get_input_feedback

reservoir = ReservoirLayer(
    reservoir_size=500,
    feedback_size=3,
    feedback_initializer="chebyshev",
    input_initializer=("random", {"input_scaling": 0.5}),
)
```

**Available initializers**: `binary_balanced`, `chain_of_neurons_input`, `chebyshev`, `chessboard`, `dendrocycle_input`, `opposite_anchors`, `pseudo_diagonal`, `random`, `random_binary`, `ring_window`, `zero`

---

## Premade Models

All in `src/resdag/models/` and importable from `resdag.models`:

```python
from resdag.models import ott_esn, classic_esn, headless_esn, linear_esn

# Ott's ESN — best for chaotic systems (state augmentation)
model = ott_esn(reservoir_size=500, feedback_size=3, output_size=3)

# Classic ESN — simple feedback-only
model = classic_esn(reservoir_size=500, feedback_size=3, output_size=3)

# Headless — reservoir states only, no readout
model = headless_esn(reservoir_size=500, feedback_size=3)

# Linear — reservoir + linear readout
model = linear_esn(reservoir_size=500, feedback_size=3, output_size=3)
```

The `ott_esn` architecture: `Input → Reservoir → SelectiveExponentiation (square even indices) → Concatenate(Input, Augmented) → CGReadout`

---

## Forecasting

```python
# Feedback-only model
predictions = model.forecast(warmup_data, horizon=1000)
# shape: (batch, 1000, output_dim)

# Input-driven model (requires forecast_drivers)
predictions = model.forecast(
    warmup_feedback, warmup_driver,
    horizon=1000,
    forecast_drivers=(future_driver,),  # shape: (batch, horizon, driver_dim)
)

# Include warmup in output
full_output = model.forecast(warmup_data, horizon=1000, return_warmup=True)
# shape: (batch, warmup_steps + 1000, output_dim)
```

**Key constraint**: First model output dimension must match feedback input dimension (required for autoregression).

---

## Hyperparameter Optimization (optional dep)

```bash
pip install resdag[hpo]  # or uv sync --extra hpo
```

```python
from resdag.hpo import run_hpo

study = run_hpo(
    model_creator=my_model_creator,   # Callable(**hparams) -> ESNModel
    search_space=my_search_space,      # Callable(trial) -> dict
    data_loader=my_data_loader,        # Callable(trial) -> {warmup,train,target,f_warmup,val}
    n_trials=100,
    loss="efh",                        # "efh"|"horizon"|"lyap"|"standard"|"discounted"
    n_workers=4,
    storage="sqlite:///study.db",      # optional persistence
)
```

---

## Model Persistence

```python
# Save (weights only)
model.save("model.pt")

# Save with reservoir states and metadata
model.save("checkpoint.pt", include_states=True, epoch=10, loss=0.05)

# Load into existing model
model.load("model.pt")
model.load("checkpoint.pt", load_states=True)

# Class method load
model = ESNModel.load_from_file("weights.pt", model=pre_built_model)
```

**Important**: You must re-create the model architecture before loading, as `save()`/`load()` only handle the state dict, not the graph definition.

---

## Version Management

Version is defined in one place: `src/resdag/__init__.py`:
```python
__version__ = "0.2.0"
```

The `pyproject.toml` reads this dynamically via:
```toml
[tool.hatch.version]
path = "src/resdag/__init__.py"
```

When bumping version, **only update `__init__.py`**.

---

## Release Process

Releases are triggered automatically by pushing a version tag:
```bash
git tag v0.3.0
git push origin v0.3.0
```

The `.github/workflows/release.yml` workflow:
1. Builds package with `python -m build`
2. Publishes to PyPI via trusted publishing (`pypa/gh-action-pypi-publish`)
3. Creates GitHub Release with auto-generated notes

---

## Key Conventions

### Adding a New Topology

1. Create `src/resdag/init/graphs/my_topology.py` with a graph function
2. Decorate with `@register_graph_topology("my_topology", **defaults)`
3. Import it in `src/resdag/init/graphs/__init__.py` so registration runs at import time

### Adding a New Input/Feedback Initializer

1. Create `src/resdag/init/input_feedback/my_init.py` extending `InputFeedbackInitializer`
2. Register via `@register_input_feedback("my_init")`
3. Import in `src/resdag/init/input_feedback/__init__.py`

### Adding a New Premade Model

1. Create `src/resdag/models/my_model.py` with a factory function returning `ESNModel`
2. Import in `src/resdag/models/__init__.py`
3. Add to `src/resdag/__init__.py` public API and `__all__`

### Public API Changes

When adding symbols to the public API, update **both**:
- `src/resdag/__init__.py` imports
- `__all__` list in `src/resdag/__init__.py`

### Type Annotations

All functions should have type annotations. `mypy` is configured in `pyproject.toml` with:
- `disallow_untyped_defs = true`
- `ignore_missing_imports = true` (for pytorch_symbolic etc.)

### Docstring Style

The codebase uses **NumPy-style docstrings** with `Parameters`, `Returns`, `Raises`, `Examples`, and `See Also` sections.

---

## Common Pitfalls

1. **Stateful reservoir**: Always call `model.reset_reservoirs()` before a new sequence unless intentionally continuing state.

2. **Readout name must match targets key**: If `CGReadoutLayer(..., name="output")`, then `targets={"output": ...}` in `trainer.fit()`.

3. **input_size=0 in ott_esn**: The `ott_esn` factory passes `input_size=0` — this is intentional to create the driving-input weight matrix as a zero-size placeholder.

4. **Topology spec is resolved at init time**: The `topology` argument to `ReservoirLayer` is resolved during `__init__`, not lazily.

5. **float64 in CG solver**: `CGReadoutLayer._solve_ridge_cg` internally casts to `float64` for numerical stability, then copies back to the layer's dtype.

6. **Multi-output forecasting**: For multi-output models, the **first** output is used as feedback in `forecast()`. Ensure its dimension matches the feedback input.

7. **HPO is an optional dependency**: `from resdag.hpo import run_hpo` will fail if `optuna` is not installed. Use `pip install resdag[hpo]`.
