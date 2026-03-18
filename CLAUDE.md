# CLAUDE.md — AI Assistant Guide for ResDAG

## Project Overview

`resdag` is a **PyTorch-native reservoir computing library** (v0.3.0) for building Echo State Networks (ESNs) and Next Generation Reservoir Computers (NG-RC). It provides GPU-accelerated, modular components for reservoir computing research: stateful reservoir layers, graph-based topology initialization, algebraic readout training, model composition via `pytorch_symbolic`, and Optuna-based hyperparameter optimization.

- **Package name**: `resdag`
- **Author**: Daniel Estevez-Moya
- **License**: MIT
- **Homepage**: https://github.com/El3ssar/resdag
- **Python**: >=3.11,<3.15 (`.python-version` pins `3.14`; classifiers: 3.11–3.14)

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
│   │   │   ├── base.py       # ReadoutLayer (extends nn.Linear; abstract fit())
│   │   │   └── cg_readout.py # CGReadoutLayer — CG ridge regression
│   │   └── custom/           # Concatenate, SelectiveExponentiation, Power, etc.
│   ├── init/
│   │   ├── topology/        # Graph topology registry + base classes
│   │   ├── input_feedback/  # Input/feedback weight initializer registry
│   │   ├── graphs/          # NetworkX graph generation functions (17 types)
│   │   └── utils/           # resolve_topology(), resolve_initializer(); TopologySpec/InitializerSpec type aliases
│   ├── training/
│   │   └── trainer.py       # ESNTrainer — algebraic readout fitting via pre-hooks
│   ├── models/              # Premade architectures
│   │   ├── classic_esn.py
│   │   ├── ott_esn.py       # Ott state-augmented ESN (recommended for chaos)
│   │   ├── power_augmented.py  # Generalized power-augmented ESN
│   │   ├── headless_esn.py
│   │   └── linear_esn.py
│   ├── hpo/                 # Optuna HPO integration (optional dep)
│   │   ├── run.py           # run_hpo()
│   │   ├── losses.py        # efh, forecast_horizon, lyapunov, standard, soft_horizon + LOSSES registry
│   │   ├── objective.py     # build_objective()
│   │   ├── runners.py       # run_single, run_multiprocess
│   │   ├── storage.py       # Storage backend resolution
│   │   └── utils.py         # get_study_summary, make_study_name, get_best_params
│   └── utils/
│       ├── data/            # load_file(), prepare_esn_data(), normalize_data(), load_and_prepare(), save_*()
│       ├── states/          # esp_index()
│       └── general.py       # create_rng()
├── tests/                   # Pytest test suite (mirrors src structure)
│   ├── test_composition/    # test_save_load.py
│   ├── test_gpu_and_compile.py
│   ├── test_hpo/            # test_losses.py, test_run.py
│   ├── test_layers/         # test_cg_readout.py, test_custom_layers.py, test_ngrc.py,
│   │                        # test_readout.py, test_reservoir.py, test_reservoir_topology.py
│   ├── test_models/         # test_premade_models.py
│   ├── test_topology/       # test_graph_topology.py
│   └── test_training/       # test_trainer.py
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

# Docs extras (optional)
pip install -e ".[docs]"
```

**Runtime dependencies**: `torch>=2.10.0`, `numpy>=2.0.0`, `networkx>=3.0`, `pytorch-symbolic>=1.1.1`, `graphviz>=0.21`, `scipy>=1.17.0`

**Dev dependencies**: `basedpyright>=1.37.1`, `pytest>=9.0.2`, `pytest-cov>=4.0.0`, `black>=25.12.0`, `ruff>=0.14.10`, `mypy>=1.0.0`, `optuna>=4.0.0`

**Docs dependencies**: `mkdocs>=1.6.0`, `mkdocs-material>=9.5.0`, `mkdocstrings[python]>=0.25.0`, `mkdocs-minify-plugin>=0.8.0`

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

Test configuration is in `pyproject.toml` under `[tool.pytest.ini_options]`. Tests live in `tests/` and largely mirror the `src/resdag/` structure. Test files are named `test_*.py`, classes `Test*`, functions `test_*`. Note that `tests/test_composition/` and `tests/test_models/` do **not** have `__init__.py` files.

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
- **Feedback input**: always the first positional argument to reservoir layer `forward()`
- **Driving inputs**: optional additional positional args to `forward()` — at most one
- ESN reservoir state shape: `(batch, reservoir_size)` (2D)
- NG-RC state shape: `(batch, state_size, input_dim)` (3D delay buffer)

### Layer Hierarchy

The reservoir stack has two levels of abstraction, with two concrete cell implementations:

| Class | Location | Responsibility |
|---|---|---|
| `ReservoirCell` | `layers/cells/base_cell.py` | Abstract single-step interface (`state_size`, `output_size`, `init_state`, `forward`) |
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
    input_size=5,              # Dim of driving input (optional; None = no driver weight matrix)
    spectral_radius=0.9,       # Controls memory/stability (None = no scaling)
    bias=True,                 # Whether to include a bias term (default True)
    activation="tanh",         # "tanh"|"relu"|"identity"|"sigmoid"
    leak_rate=1.0,             # 1.0 = no leaking (standard ESN)
    trainable=False,           # False = frozen weights (standard ESN)
    topology="erdos_renyi",    # str | (name, params) tuple | GraphTopology | None
    feedback_initializer=None, # str | (name, params) tuple | InputFeedbackInitializer | None
    input_initializer=None,    # str | (name, params) tuple | InputFeedbackInitializer | None
)

# Forward: states of shape (batch, time, reservoir_size)
states = reservoir(feedback)                    # feedback-only
states = reservoir(feedback, driving_input)     # with driver (at most one)
```

**`ESNCell` weight attributes** (accessible on `ESNLayer` via `__getattr__` delegation):
- `weight_feedback`: `(reservoir_size, feedback_size)` — feedback weight matrix
- `weight_hh`: `(reservoir_size, reservoir_size)` — recurrent weight matrix
- `weight_input`: `(reservoir_size, input_size)` or `None` — driving input weights
- `bias_h`: `(reservoir_size,)` or `None` — bias vector
- `reservoir_size`, `feedback_size`, `input_size`, `spectral_radius`, `leak_rate`, `trainable`, `activation` — configuration attributes

**State management** (critical — reservoir is stateful):
```python
reservoir.reset_state()             # Reset to None (lazy re-init on next forward)
reservoir.reset_state(batch_size=4) # Reset to zeros with explicit batch size
reservoir.get_state()               # Returns clone or None
reservoir.set_state(state_tensor)   # Restore a saved state (validates last dim == state_size)
reservoir.set_random_state()        # Set state to standard-normal random (requires non-None state)
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
2. **O_nonlin**: All degree-p monomials from O_lin via `combinations_with_replacement` — `C(D+p-1, p)` terms
3. **O_total**: `[constant] + O_lin + O_nonlin` concatenated

**Feature dimension**: `int(include_constant) + int(include_linear)*D + C(D+p-1, p)`

**Convenience properties on `NGReservoir`** (delegated to inner `NGCell`):
- `input_dim`, `feature_dim` — input and output dimensions
- `warmup_length` — `(k-1)*s`, the steps needed to fill the delay buffer

**Key differences from ESN**:
- State is a FIFO delay buffer of shape `(batch, (k-1)*s, input_dim)`, not a recurrent hidden state
- No learnable parameters (no weights); `monomial_indices` and `delay_indices` registered as buffers
- Warmup length = `(k-1)*s` steps for buffer to fill
- When `k=1`: `state_size=0` (empty buffer tensor)
- Warns if `feature_dim > 10,000` (combinatorial explosion risk)

**State management** inherits from `BaseReservoirLayer`; `NGReservoir.set_state()` overrides to validate the 3D buffer shape `(batch, state_size, input_dim)`.

### Readout Layers

**`ReadoutLayer`** (`layers/readouts/base.py`) extends `nn.Linear`:
- Handles 2D `(B, F)` and 3D `(B, T, F)` inputs — applies linear transform per-timestep for 3D
- `name` property used as target key in `ESNTrainer`
- `is_fitted` property becomes `True` after `fit()` is called
- `trainable=False` freezes weights by default (standard ESN behavior)
- Base `fit()` raises `NotImplementedError` — use `CGReadoutLayer` for actual fitting

**`CGReadoutLayer`** (`layers/readouts/cg_readout.py`) extends `ReadoutLayer`:
- Implements `fit(inputs, targets)` via Conjugate Gradient ridge regression
- Works in float64 internally for numerical stability, copies back to layer dtype
- Handles 3D `(batch, time, features)` by reshaping to 2D before solving
- Centers data for numerical stability; solves `||XW - Y||² + α||W||²`

```python
from resdag.layers.readouts import CGReadoutLayer

readout = CGReadoutLayer(
    in_features=500,   # Reservoir size
    out_features=3,    # Output dimension
    bias=True,         # Include bias (default True)
    alpha=1e-6,        # L2 regularization strength
    name="output",     # Used as key in targets dict (important!)
    trainable=False,   # Frozen weights (default)
    max_iter=100,      # Max CG iterations
    tol=1e-5,          # Convergence tolerance
)
```

### ESNModel and Composition

Models are built with `pytorch_symbolic` functional API, then wrapped in `ESNModel`:

```python
import pytorch_symbolic as ps
from resdag import ESNModel, ESNLayer, CGReadoutLayer

inp = ps.Input((100, 3))                          # (seq_len, features); batch dim implicit
reservoir = ESNLayer(200, feedback_size=3)(inp)
readout = CGReadoutLayer(200, 3, name="output")(reservoir)
model = ESNModel(inp, readout)

# Multi-input model with driving signal
feedback = ps.Input((100, 3))
driver = ps.Input((100, 5))
reservoir = ESNLayer(100, feedback_size=3, input_size=5)(feedback, driver)
readout = CGReadoutLayer(100, 3)(reservoir)
model = ESNModel([feedback, driver], readout)
```

`ESNModel` extends `pytorch_symbolic.SymbolicModel` with:
- `reset_reservoirs()` — reset all `BaseReservoirLayer` states to None
- `set_random_reservoir_states()` — set all states to standard-normal random
- `get_reservoir_states()` → `dict[str, Tensor]` — get state clones of all initialized reservoirs
- `set_reservoir_states(states)` — restore states from dict
- `warmup(*inputs, return_outputs=False)` — teacher-forced state synchronization; returns output if `return_outputs=True`
- `forecast(*warmup_inputs, horizon=N, forecast_drivers=None, initial_feedback=None, return_warmup=False)` — two-phase autoregressive forecasting
- `save(path, include_states=False, **metadata)` / `load(path, strict=True, load_states=False)` — model persistence
- `load_from_file(path, model, strict=True, load_states=False)` — class method load
- `plot_model(show_shapes=False, show_trainable=False, rankdir='TB', save_path=None, format='svg')` — architecture visualization via graphviz

### ESNTrainer (training workflow)

```python
from resdag.training import ESNTrainer

trainer = ESNTrainer(model)
trainer.fit(
    warmup_inputs=(warmup_feedback,),           # Tuple — synchronize reservoir states
    train_inputs=(train_feedback,),             # Tuple — fit readout
    targets={"output": train_targets},          # Dict keyed by readout name or auto-name
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
2. Warmup phase (teacher-forced forward pass, no gradient)
3. Register `forward_pre_hook` on each readout in topological order
4. Single forward pass — hooks call `readout.fit(inputs, target)` just before each readout executes
5. Remove hooks (always, even on exception)

**Target key resolution**: `readout.name` if set, otherwise the auto-generated module name (e.g. `"CGReadoutLayer_1"`). A `UserWarning` is emitted if `targets` contains keys with no matching readout.

### Custom Layers

All in `src/resdag/layers/custom/`:

| Layer | Purpose |
|---|---|
| `Concatenate` | Concatenates inputs along feature dimension (parameterless) |
| `SelectiveExponentiation` | Exponentiates even/odd feature indices; `index=0` → square even-indexed units (used in `ott_esn`) |
| `Power` | Exponentiates all features to a given power (used in `power_augmented`) |
| `SelectiveDropout` | Per-feature dropout with selectivity control |
| `FeaturePartitioner` | Partitions features into overlapping groups |
| `OutliersFilteredMean` | Computes mean with outlier filtering |

---

## Premade Models

All in `src/resdag/models/` and importable from `resdag.models`:

```python
from resdag.models import ott_esn, classic_esn, headless_esn, linear_esn, power_augmented

# Ott's ESN — best for chaotic systems (state augmentation: square even-indexed units)
model = ott_esn(reservoir_size=500, feedback_size=3, output_size=3)

# Power Augmented — generalized state augmentation with configurable exponent
model = power_augmented(reservoir_size=500, feedback_size=3, output_size=3, exponent=2.0)

# Classic ESN — simple feedback-only with readout
model = classic_esn(reservoir_size=500, feedback_size=3, output_size=3)

# Headless — reservoir states only, no readout
model = headless_esn(reservoir_size=500, feedback_size=3)

# Linear — reservoir + linear readout (uses nn.Linear, not CGReadoutLayer)
model = linear_esn(reservoir_size=500, feedback_size=3, output_size=3)
```

All factories (except `headless_esn`) accept: `topology`, `spectral_radius`, `leak_rate`, `feedback_initializer`, `activation`, `bias`, `trainable`, `readout_alpha`, `readout_bias`, `readout_name`, `**reservoir_kwargs`. `power_augmented` also accepts `exponent`.

**Architectures**:
- `ott_esn`: `Input → ESNLayer(input_size=0) → SelectiveExponentiation(index=0, exponent=2.0) → Concatenate(Input, Augmented) → CGReadout`
- `power_augmented`: `Input → ESNLayer(input_size=0) → Power(exponent) → Concatenate(Input, Augmented) → CGReadout`
- `classic_esn`: `Input → ESNLayer → CGReadout`
- `headless_esn`: `Input → ESNLayer` (returns ESNModel with no readout)
- `linear_esn`: `Input → ESNLayer → Linear Readout`

> **Note**: `power_augmented` is available in `resdag.models` but is **not** re-exported from the top-level `resdag` namespace.

---

## Topology System

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

**Introspection utilities** (from `resdag.init.topology`):
```python
from resdag.init.topology import show_topologies
show_topologies()              # Returns sorted list of all registered names
show_topologies("erdos_renyi") # Prints parameters and defaults for that topology
```

**Registering a new topology**:
```python
from resdag.init.topology import register_graph_topology
import networkx as nx

@register_graph_topology("my_graph", p=0.1, directed=True)
def my_graph(n: int, p: float = 0.1, directed: bool = True, seed=None) -> nx.DiGraph:
    # Must accept n as first arg, return nx.Graph or nx.DiGraph with weighted edges
    ...
```

---

## Input/Feedback Initializer System

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

**Available initializers** (11, all in `src/resdag/init/input_feedback/`):
`binary_balanced`, `chain_of_neurons_input`, `chebyshev`, `chessboard`, `dendrocycle_input`, `opposite_anchors`, `pseudo_diagonal`, `random`, `random_binary`, `ring_window`, `zero`

**Introspection utilities** (from `resdag.init.input_feedback`):
```python
from resdag.init.input_feedback import show_input_initializers
show_input_initializers()          # Returns sorted list of all registered initializer names
show_input_initializers("chebyshev")  # Prints parameters and defaults
```

**Registering a new initializer**:
```python
from resdag.init.input_feedback import register_input_feedback, InputFeedbackInitializer

@register_input_feedback("my_init", scaling=1.0)
class MyInitializer(InputFeedbackInitializer):
    def __init__(self, scaling: float = 1.0):
        self.scaling = scaling

    def initialize(self, weight: torch.Tensor, **kwargs) -> torch.Tensor:
        # weight is (reservoir_size, input_or_feedback_size)
        ...
```

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
# shape: (batch, warmup_steps + horizon, output_dim)

# Custom initial feedback
predictions = model.forecast(warmup_data, horizon=1000, initial_feedback=custom_start)
```

**Key constraints**:
- First model output dimension must match feedback input dimension (required for autoregression)
- For multi-output models, the **first** output is used as feedback
- `forecast()` calls `warmup()` internally, which updates reservoir state; call `reset_reservoirs()` first if needed

---

## Data Utilities

```python
from resdag.utils.data import load_file, prepare_esn_data, normalize_data, load_and_prepare

# Load a single file (auto-detects format: .csv, .npy, .npz, .nc)
data = load_file("lorenz.csv")  # returns (1, T, D) tensor

# Prepare ESN splits from a pre-loaded tensor
warmup, train, target, f_warmup, val = prepare_esn_data(
    data,
    warmup_steps=100,
    train_steps=500,
    val_steps=200,
    discard_steps=0,       # Discard initial transient
    normalize=False,       # Optional normalization (stats from train only)
    norm_method="minmax",  # "minmax"|"standard"|"noncentered"|"meanpreserving"
)

# Convenience: load + prepare in one call (concatenates along batch dim if multiple paths)
warmup, train, target, f_warmup, val = load_and_prepare(
    "lorenz.csv", warmup_steps=100, train_steps=500, val_steps=200
)

# Normalize only
normalized, stats = normalize_data(data, method="minmax")
# Apply same stats to new data (pass pre-computed stats)
new_normalized, _ = normalize_data(new_data, method="minmax", stats=stats)
```

**Data layout**: `[discard][warmup][train][val]`. `target` = `train` shifted forward 1 step. `f_warmup` = last `warmup_steps` of training data (for forecast initialization).

**Individual I/O functions** also available: `load_csv`, `load_npy`, `load_npz`, `load_nc` (requires xarray), `save_csv`, `save_npy`, `save_npz`, `save_nc`, `list_files`.

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
    data_loader=my_data_loader,        # Callable(trial) -> {warmup, train, target, f_warmup, val}
    n_trials=100,
    loss="efh",                        # loss key or LossProtocol callable
    n_workers=4,
    storage="study.log",               # journal file (recommended for multi-worker)
)
```

**Loss function registry** (`LOSSES` dict — always available without optuna):

| Registry key | Function | Description |
|---|---|---|
| `"efh"` | `expected_forecast_horizon` | Sigmoid-gated survival probability (default; recommended for chaotic systems) |
| `"forecast_horizon"` | `forecast_horizon` | Contiguous valid steps below threshold |
| `"lyapunov"` | `lyapunov_weighted` | Exponential time-decay weighting (characteristic Lyapunov time) |
| `"standard"` | `standard_loss` | Mean geometric mean error across timesteps |
| `"soft_horizon"` | `soft_valid_horizon` | Hill-function gated survival probability |

Loss functions are always importable (no optuna required):
```python
from resdag.hpo import LOSSES, get_loss, LossProtocol
loss_fn = get_loss("efh")
loss_fn = get_loss(my_custom_callable)  # any LossProtocol-compatible callable
```

All loss functions take `(y_true, y_pred, /, **kwargs)` with arrays of shape `(B, T, D)` and return a `float` (lower = better; EFH and horizon losses return negative values).

**Additional `run_hpo()` parameters**:
- `loss_params` — kwargs for the loss function (passed via `functools.partial`)
- `targets_key` — readout layer name (default `"output"`)
- `drivers_keys` — driver input keys for input-driven models
- `monitor_losses` — additional losses to compute and log as trial user attributes (not optimized on)
- `monitor_params` — kwargs per monitor loss, keyed by loss name or function name
- `sampler` — custom Optuna sampler (default: `TPESampler(multivariate=True)`)
- `seed` — reproducibility seed (seeds sampler + per-trial `seed + trial.number`)
- `device` — target device for models/data
- `verbosity` — 0=silent, 1=normal, 2=verbose
- `catch_exceptions` — catch and return `penalty_value` on failure (default `True`)
- `penalty_value` — value returned for failed trials (default `1e10`)
- `clip_value` / `prune_on_clip` — upper-bound clamping / prune instead of clamp

**Storage backends**:
- `None` — in-memory (single worker) or auto-created temp journal file (multi-worker)
- `"study.log"` — `JournalFileStorage` (recommended for multi-worker)
- `"study.db"` or `"sqlite:///study.db"` — SQLite with WAL mode

**Multi-worker** uses real OS processes (`multiprocessing`), throttles BLAS/OpenMP threads before fork, disposes storage references cleanly.

**Utility functions** (require optuna; available via `resdag.hpo`):
- `get_study_summary(study)` — print/return study summary
- `make_study_name(model_creator)` — auto-generate study name from function name
- `get_best_params(study)` — return best trial parameters dict

> **Top-level lazy imports**: `resdag.run_hpo`, `resdag.LOSSES`, `resdag.get_study_summary` are accessible via `resdag.__getattr__`. `get_best_params` and `make_study_name` are only accessible via `resdag.hpo`.

---

## Echo State Property (ESP) Utility

```python
from resdag.utils.states import esp_index

# Compute ESP index (only detects ESNLayer instances, not NGReservoir)
esp_values = esp_index(
    model, feedback_seq,
    iterations=10,    # Number of random initial states to average over
    transient=0,      # Timesteps to discard from start of sequence
    verbose=True,
)
# Returns: dict[layer_name, list[Tensor]] — ESP index per reservoir layer

# With driving inputs + history
esp_values, history = esp_index(
    model, feedback_seq, driving_seq,
    history=True,
)
# history: dict[layer_name, list[Tensor]] of shape (iterations, timesteps, batch)
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
model.load("checkpoint.pt", load_states=True)  # UserWarning if no states in checkpoint

# Class method load
model = ESNModel.load_from_file("weights.pt", model=pre_built_model)
```

**Important**: `save()`/`load()` handle the state dict only, not the graph definition. Re-create the model architecture before loading.

---

## Public API

Symbols importable directly from `resdag`:

```python
# Modules
from resdag import composition, hpo, init, layers, models, training, utils
# Convenience submodules
from resdag import graphs, topology, input_feedback
# Core layers
from resdag import CGReadoutLayer, Concatenate, ESNLayer, NGCell, NGReservoir
from resdag import OutliersFilteredMean, SelectiveExponentiation
# Model composition
from resdag import ESNModel
# Training
from resdag import ESNTrainer
# Premade models (note: power_augmented is NOT at top level)
from resdag import classic_esn, ott_esn, headless_esn, linear_esn
# Lazy HPO imports (require optuna for run_hpo; LOSSES and get_study_summary always resolve)
from resdag import run_hpo, LOSSES, get_study_summary
```

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
2. Register via `@register_input_feedback("my_init", **defaults)`
3. Import in `src/resdag/init/input_feedback/__init__.py`

### Adding a New Reservoir Cell

1. Create `src/resdag/layers/cells/my_cell.py` extending `ReservoirCell`
2. Implement `state_size` and `output_size` properties
3. Implement `init_state(batch_size, device, dtype) -> Tensor` — return zero initial state
4. Implement `forward(inputs: list[Tensor], state: Tensor) -> tuple[Tensor, Tensor]`
   - `inputs[0]` is always the feedback slice; `inputs[1]` is the optional driving input
   - Returns `(output, new_state)`
5. Import in `src/resdag/layers/cells/__init__.py`

### Adding a New Reservoir Layer

1. Create `src/resdag/layers/reservoirs/my_layer.py` extending `BaseReservoirLayer`
2. Create an appropriate cell and pass it to `super().__init__(cell)`
3. Optionally override `set_state()` for custom state shape validation (e.g. 3D for NG-RC)
4. Add `__getattr__` delegation to inner cell for convenience access
5. Import in `src/resdag/layers/reservoirs/__init__.py`
6. Re-export from `src/resdag/layers/__init__.py` if it belongs in the public API

### Adding a New Premade Model

1. Create `src/resdag/models/my_model.py` with a factory function returning `ESNModel`
2. Import in `src/resdag/models/__init__.py` and add to its `__all__`
3. If it should be top-level: add import and entry to `src/resdag/__init__.py` and `__all__`

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

2. **Readout name must match targets key**: If `CGReadoutLayer(..., name="output")`, then `targets={"output": ...}` in `trainer.fit()`. If no name is set, use the auto-generated module name (e.g. `"CGReadoutLayer_1"`).

3. **`input_size=0` in ott_esn/power_augmented**: These factories pass `input_size=0` — this creates a zero-size weight matrix `(reservoir_size, 0)`. No driving input is expected at runtime.

4. **Topology spec is resolved at init time**: The `topology` argument to `ESNLayer` (and `ESNCell`) is resolved during `__init__`, not lazily.

5. **float64 in CG solver**: `CGReadoutLayer._solve_ridge_cg` internally casts to `float64` for numerical stability, then copies back to the layer's dtype.

6. **Multi-output forecasting**: For multi-output models, the **first** output is used as feedback in `forecast()`. Ensure its dimension matches the feedback input.

7. **HPO is an optional dependency**: `from resdag.hpo import run_hpo` will fail if `optuna` is not installed. Use `pip install resdag[hpo]`. Loss functions (`LOSSES`, `get_loss`, etc.) are always available.

8. **Loss registry keys vs. run.py docstring**: The `run.py` docstring lists `"horizon"`, `"lyap"`, `"discounted"` as valid loss names — these are **incorrect**. The actual `LOSSES` registry keys are `"efh"`, `"forecast_horizon"`, `"lyapunov"`, `"standard"`, `"soft_horizon"`.

9. **NG-RC combinatorial explosion**: `NGCell` warns when `feature_dim > 10,000`. High values of `k`, `p`, or `input_dim` cause combinatorial blowup (`C(D+p-1, p)` where `D = input_dim * k`).

10. **NG-RC warmup**: The delay buffer needs `(k-1)*s` steps to fill (`layer.warmup_length`). Earlier outputs contain zeros from unfilled buffer slots — discard the first `warmup_length` outputs if accuracy matters.

11. **NG-RC state shape**: Unlike ESN's 2D state `(batch, reservoir_size)`, NG-RC state is 3D: `(batch, state_size, input_dim)`. `NGReservoir.set_state()` validates this shape.

12. **`set_random_state()` requires initialized state**: Call `reset_state(batch_size=N)` first to allocate the state tensor before calling `set_random_state()`.

13. **`power_augmented` not at top level**: Import it from `resdag.models`, not `resdag` directly.

14. **`esp_index` only supports `ESNLayer`**: It searches for `ESNLayer` instances by type; `NGReservoir` layers are not detected.
