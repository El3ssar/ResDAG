# CLAUDE.md — AI Assistant Guide for ResDAG

## Project Overview

`resdag` is a **PyTorch-native reservoir computing library** (v0.3.0) for building Echo State Networks (ESNs) and Next Generation Reservoir Computers (NG-RC). It provides GPU-accelerated, modular components for reservoir computing research: stateful reservoir layers, graph-based topology initialization, algebraic readout training, model composition via `pytorch_symbolic`, and Optuna-based hyperparameter optimization.

- **Package name**: `resdag`
- **Author**: Daniel Estevez-Moya
- **License**: MIT
- **Homepage**: https://github.com/El3ssar/resdag
- **Python**: >=3.11,<3.15 (classifiers: 3.11–3.14)

---

## Repository Layout

```
resdag/
├── src/resdag/              # Main package (src layout)
│   ├── __init__.py          # Public API + lazy HPO imports; version string
│   ├── composition/
│   │   └── symbolic.py      # ESNModel (extends pytorch_symbolic.SymbolicModel)
│   ├── layers/
│   │   ├── cells/
│   │   │   ├── base_cell.py  # ReservoirCell (abstract single-step interface)
│   │   │   ├── esn_cell.py   # ESNCell — concrete leaky-ESN single-step update
│   │   │   └── ngrc_cell.py  # NGCell — NG-RC feature construction (no weights)
│   │   ├── reservoirs/
│   │   │   ├── base_reservoir.py  # BaseReservoirLayer — sequence loop + state mgmt
│   │   │   ├── esn.py             # ESNLayer — public-facing stateful RNN
│   │   │   └── ngrc.py            # NGReservoir — NG-RC sequence wrapper
│   │   ├── readouts/
│   │   │   ├── base.py       # ReadoutLayer (abstract)
│   │   │   └── cg_readout.py # CGReadoutLayer — CG ridge regression
│   │   └── custom/           # Concatenate, SelectiveExponentiation, Power, etc.
│   ├── init/
│   │   ├── topology/        # Graph topology registry + base classes
│   │   ├── input_feedback/  # Input/feedback weight initializer registry
│   │   ├── graphs/          # NetworkX graph generation functions (17 types)
│   │   └── utils/           # resolve_topology(), resolve_initializer()
│   ├── training/
│   │   └── trainer.py       # ESNTrainer — algebraic readout fitting
│   ├── models/              # Premade architectures
│   │   ├── classic_esn.py
│   │   ├── ott_esn.py       # Ott state-augmented ESN (recommended for chaos)
│   │   ├── power_augmented.py  # Generalized power-augmented ESN
│   │   ├── headless_esn.py
│   │   └── linear_esn.py
│   ├── hpo/                 # Optuna HPO integration (optional dep)
│   │   ├── run.py           # run_hpo()
│   │   ├── losses.py        # efh, horizon, lyap, standard, discounted
│   │   ├── objective.py     # build_objective()
│   │   ├── runners.py       # run_single, run_multiprocess
│   │   ├── storage.py       # Storage backend resolution
│   │   └── utils.py         # get_study_summary, make_study_name, get_best_params
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

**Runtime dependencies**: `torch>=2.10.0`, `numpy>=2.0.0`, `networkx>=3.0`, `pytorch-symbolic>=1.1.1`, `graphviz>=0.21`, `scipy>=1.17.0`

**Dev dependencies**: `basedpyright`, `pytest`, `pytest-cov`, `black`, `ruff`, `mypy`, `optuna`

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
- Target: Python 3.11+ syntax (black target `py311`–`py314`, ruff target `py311`)
- `__init__.py` files: unused imports (`F401`) are allowed — they expose the public API

---

## Core Architecture Concepts

### Tensor Conventions

- **3D tensors**: `(batch, timesteps, features)` throughout
- **Feedback input**: always the first positional argument to `ESNLayer.forward()`
- **Driving inputs**: optional additional positional args to `forward()`
- Reservoir state shape: `(batch, reservoir_size)` for ESN, `(batch, state_size, input_dim)` for NG-RC

### Layer Hierarchy

The reservoir stack has two levels of abstraction, with two concrete cell implementations:

| Class | Location | Responsibility |
|---|---|---|
| `ReservoirCell` | `layers/cells/base_cell.py` | Abstract single-step interface |
| `ESNCell` | `layers/cells/esn_cell.py` | Concrete leaky-ESN single-step update; owns all weights |
| `NGCell` | `layers/cells/ngrc_cell.py` | NG-RC feature construction; no weights, delay buffer state |
| `BaseReservoirLayer` | `layers/reservoirs/base_reservoir.py` | Abstract sequence loop + full state-management API |
| `ESNLayer` | `layers/reservoirs/esn.py` | Public-facing stateful RNN; wraps `ESNCell` |
| `NGReservoir` | `layers/reservoirs/ngrc.py` | Public-facing NG-RC layer; wraps `NGCell` |

### ESNLayer (core component)

```python
from resdag.layers import ESNLayer

reservoir = ESNLayer(
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
reservoir.reset_state()             # Reset to None (lazy re-init on next forward)
reservoir.reset_state(batch_size=4) # Reset to zeros with explicit batch size
reservoir.get_state()               # Returns clone or None
reservoir.set_state(state_tensor)   # Restore a saved state
reservoir.set_random_state()        # Set state to standard-normal random values
```

`ESNLayer` delegates unknown attribute lookups to its inner `ESNCell` via `__getattr__`, so `reservoir.reservoir_size`, `reservoir.weight_hh`, etc. all work directly.

### NGReservoir (Next Generation Reservoir Computing)

Implements the NG-RC architecture from Gauthier et al. (arXiv:2106.07688v2). Unlike traditional ESNs, NG-RC uses **no recurrent weights** — it constructs features via time-delayed input embeddings and polynomial monomials.

```python
from resdag.layers import NGReservoir

layer = NGReservoir(
    input_dim=3,              # Dimensionality of input vector
    k=2,                      # Number of delay taps (including current)
    s=1,                      # Spacing between taps in timesteps
    p=2,                      # Polynomial degree for monomials
    include_constant=True,    # Prepend constant 1.0 feature
    include_linear=True,      # Include linear delay-embedded features
)

x = torch.randn(4, 100, 3)   # (batch, seq_len, features)
features = layer(x)           # (4, 100, feature_dim)
```

**Feature construction**:
1. **O_lin**: Linear delay-embedded features `[X_i || X_{i-s} || ... || X_{i-(k-1)s}]` — dimension `D = input_dim * k`
2. **O_nonlin**: All degree-p monomials from O_lin — `C(D+p-1, p)` combinations
3. **O_total**: `[constant] + O_lin + O_nonlin` concatenated

**Feature dimension**: `int(include_constant) + int(include_linear)*D + C(D+p-1, p)`

**Key differences from ESN**:
- State is a FIFO delay buffer of shape `(batch, (k-1)*s, input_dim)`, not a recurrent hidden state
- No learnable parameters (no weights)
- Warmup length = `(k-1)*s` steps for buffer to fill
- When `k=1`: state_size=0 (no delay buffer needed)
- Warns if `feature_dim > 10,000` (combinatorial explosion risk)

**State management** inherits from `BaseReservoirLayer` with 3D buffer validation in `set_state()`.

### ESNModel and Composition

Models are built with `pytorch_symbolic` functional API, then wrapped in `ESNModel`:

```python
import pytorch_symbolic as ps
from resdag import ESNModel, ESNLayer, CGReadoutLayer

inp = ps.Input((100, 3))                          # (seq_len, features)
reservoir = ESNLayer(200, feedback_size=3)(inp)
readout = CGReadoutLayer(200, 3, name="output")(reservoir)
model = ESNModel(inp, readout)
```

`ESNModel` extends `pytorch_symbolic.SymbolicModel` with:
- `reset_reservoirs()` — reset all reservoir states
- `set_random_reservoir_states()` — set all states to standard-normal random
- `get_reservoir_states()` → `dict[str, Tensor]` — get state clones
- `set_reservoir_states(states)` — restore states from dict
- `warmup(*inputs)` — teacher-forced state synchronization
- `forecast(*warmup_inputs, horizon=N, ...)` — two-phase autoregressive forecasting
- `save(path)` / `load(path)` — model persistence
- `plot_model()` — architecture visualization via graphviz

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

### Custom Layers

All in `src/resdag/layers/custom/`:

| Layer | Purpose |
|---|---|
| `Concatenate` | Concatenates inputs along feature dimension (parameterless) |
| `SelectiveExponentiation` | Exponentiates even/odd feature indices (used in `ott_esn`) |
| `Power` | Exponentiates all features to a given power (used in `power_augmented`) |
| `SelectiveDropout` | Per-feature dropout with selectivity control |
| `FeaturePartitioner` | Partitions features into overlapping groups |
| `OutliersFilteredMean` | Computes mean with outlier filtering |

### Topology System

**Three ways to specify topology** (used in `ESNLayer(topology=...)`):
```python
# 1. String name (uses registry defaults)
topology = "erdos_renyi"

# 2. Tuple (name + kwargs override)
topology = ("watts_strogatz", {"k": 6, "p": 0.3})

# 3. Pre-configured object
from resdag.init.topology import get_topology
topology = get_topology("barabasi_albert", m=3, seed=42)
```

**Available topologies** (17, all in `src/resdag/init/graphs/`):
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

reservoir = ESNLayer(
    reservoir_size=500,
    feedback_size=3,
    feedback_initializer="chebyshev",
    input_initializer=("random", {"input_scaling": 0.5}),
)
```

**Available initializers** (11): `binary_balanced`, `chain_of_neurons_input`, `chebyshev`, `chessboard`, `dendrocycle_input`, `opposite_anchors`, `pseudo_diagonal`, `random`, `random_binary`, `ring_window`, `zero`

---

## Premade Models

All in `src/resdag/models/` and importable from `resdag.models`:

```python
from resdag.models import ott_esn, classic_esn, headless_esn, linear_esn, power_augmented

# Ott's ESN — best for chaotic systems (state augmentation)
model = ott_esn(reservoir_size=500, feedback_size=3, output_size=3)

# Power Augmented — generalized state augmentation with configurable exponent
model = power_augmented(reservoir_size=500, feedback_size=3, output_size=3, exponent=2.0)

# Classic ESN — simple feedback-only
model = classic_esn(reservoir_size=500, feedback_size=3, output_size=3)

# Headless — reservoir states only, no readout
model = headless_esn(reservoir_size=500, feedback_size=3)

# Linear — reservoir + linear readout
model = linear_esn(reservoir_size=500, feedback_size=3, output_size=3)
```

**Architectures**:
- `ott_esn`: `Input → Reservoir → SelectiveExponentiation (square even indices) → Concatenate(Input, Augmented) → CGReadout`
- `power_augmented`: `Input → Reservoir → Power(exponent) → Concatenate(Input, Augmented) → CGReadout`
- `classic_esn`: `Input → Reservoir → CGReadout`
- `headless_esn`: `Input → Reservoir`
- `linear_esn`: `Input → Reservoir → Linear Readout`

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
    storage="study.log",               # journal file (recommended for multi-worker)
)
```

**Loss functions** (always available, no optuna required):
- `"efh"` — Expected Forecast Horizon (default, recommended for chaotic systems)
- `"horizon"` — Forecast Horizon (contiguous valid steps)
- `"lyap"` — Lyapunov-weighted (exponential decay)
- `"standard"` — Standard Loss (mean geometric mean error)
- `"discounted"` — Discounted RMSE (half-life weighted)

**Additional `run_hpo()` parameters**:
- `loss_params` — kwargs for the loss function
- `targets_key` — readout layer name (default `"output"`)
- `drivers_keys` — driver input keys for input-driven models
- `monitor_losses` — additional losses to compute and log (not optimized on)
- `monitor_params` — kwargs per monitor loss
- `sampler` — custom Optuna sampler (default: `TPESampler(multivariate=True)`)
- `seed` — reproducibility seed (seeds sampler + per-trial `seed + trial.number`)
- `device` — target device for models/data
- `verbosity` — 0=silent, 1=normal, 2=verbose
- `catch_exceptions` — catch and return `penalty_value` on failure
- `clip_value` / `prune_on_clip` — upper-bound clamping / pruning

**Storage backends**:
- `None` — in-memory (single worker) or temp journal file (multi-worker)
- `"study.log"` — `JournalFileStorage` (recommended for multi-worker)
- `"study.db"` or `"sqlite:///study.db"` — SQLite with WAL mode

**Multi-worker** uses real OS processes (not threads), throttles BLAS/OpenMP threads before fork.

**Utility functions** (require optuna): `get_study_summary()`, `make_study_name()`, `get_best_params()`

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
__version__ = "0.3.0"
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

### Adding a New Reservoir Cell

1. Create `src/resdag/layers/cells/my_cell.py` extending `ReservoirCell`
2. Implement `state_size`, `output_size` properties and `forward(inputs, state) -> (output, new_state)` method
3. Implement `init_state(batch_size, device, dtype) -> Tensor` for initial state creation
4. Import in `src/resdag/layers/cells/__init__.py`

### Adding a New Reservoir Layer

1. Create `src/resdag/layers/reservoirs/my_layer.py` extending `BaseReservoirLayer`
2. Create an appropriate cell and pass it to `super().__init__(cell)`
3. Optionally override `set_state()` for custom state shape validation
4. Add `__getattr__` delegation to inner cell for convenience access
5. Import in `src/resdag/layers/reservoirs/__init__.py`
6. Re-export from `src/resdag/layers/__init__.py` if it belongs in the public API

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

3. **input_size=0 in ott_esn/power_augmented**: These factories pass `input_size=0` — this is intentional to create the driving-input weight matrix as a zero-size placeholder.

4. **Topology spec is resolved at init time**: The `topology` argument to `ESNLayer` (and `ESNCell`) is resolved during `__init__`, not lazily.

5. **float64 in CG solver**: `CGReadoutLayer._solve_ridge_cg` internally casts to `float64` for numerical stability, then copies back to the layer's dtype.

6. **Multi-output forecasting**: For multi-output models, the **first** output is used as feedback in `forecast()`. Ensure its dimension matches the feedback input.

7. **HPO is an optional dependency**: `from resdag.hpo import run_hpo` will fail if `optuna` is not installed. Use `pip install resdag[hpo]`.

8. **NG-RC combinatorial explosion**: `NGCell` warns when `feature_dim > 10,000`. High values of `k`, `p`, or `input_dim` cause combinatorial blowup in monomial count (`C(D+p-1, p)` where `D = input_dim * k`).

9. **NG-RC warmup**: The delay buffer needs `(k-1)*s` steps to fill. Earlier outputs contain zeros from unfilled buffer slots — discard the first `warmup_length` outputs if accuracy matters.

10. **NG-RC state shape**: Unlike ESN's 2D state `(batch, reservoir_size)`, NG-RC state is 3D: `(batch, state_size, input_dim)`. The `set_state()` override validates this shape.
