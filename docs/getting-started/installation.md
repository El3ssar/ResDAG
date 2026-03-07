# Installation

## Requirements

| Dependency | Minimum Version | Notes |
|---|---|---|
| Python | 3.11 | Up to 3.14 supported |
| PyTorch | 2.10.0 | CPU or CUDA |
| NumPy | 2.0.0 | |
| NetworkX | 3.0 | Graph generation |
| pytorch-symbolic | 1.1.1 | Model composition |
| SciPy | 1.17.0 | Linear algebra |
| Graphviz | 0.21 | Model visualization |

---

## Install from PyPI

=== "pip"
    ```bash
    pip install resdag
    ```

=== "uv (recommended)"
    ```bash
    uv add resdag
    ```

---

## Optional Extras

### Hyperparameter Optimization

HPO support requires [Optuna](https://optuna.org/) ≥ 3.0:

=== "pip"
    ```bash
    pip install "resdag[hpo]"
    ```

=== "uv"
    ```bash
    uv add "resdag[hpo]"
    ```

### Development Environment

=== "pip"
    ```bash
    pip install "resdag[dev]"
    ```

=== "uv"
    ```bash
    uv sync --dev
    ```

This installs: `pytest`, `pytest-cov`, `black`, `ruff`, `mypy`, `basedpyright`, and `optuna`.

---

## Install from Source

```bash
git clone https://github.com/El3ssar/resdag.git
cd resdag

# With uv (faster, recommended)
uv sync --dev

# With pip
pip install -e ".[dev]"

# With HPO support as well
pip install -e ".[dev,hpo]"
# or: uv sync --extra hpo
```

---

## Verify Installation

```python
import resdag
print(resdag.__version__)  # e.g. 0.3.0

import torch
from resdag.layers import ESNLayer

reservoir = ESNLayer(reservoir_size=100, feedback_size=3)
x = torch.randn(1, 10, 3)
out = reservoir(x)
print(out.shape)  # torch.Size([1, 10, 100])
```

---

## GPU Support

resdag inherits GPU support from PyTorch — move your model and data to CUDA as usual:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ott_esn(reservoir_size=500, feedback_size=3, output_size=3)
model = model.to(device)

data = data.to(device)
```

!!! tip "CUDA Installation"
    Install PyTorch with CUDA support from [pytorch.org](https://pytorch.org/get-started/locally/).
    resdag does not require a separate CUDA installation.

---

## Building the Docs

To build this documentation site locally:

```bash
# Install MkDocs dependencies
pip install "resdag[docs]"

# Serve locally with live reload
mkdocs serve

# Build static site
mkdocs build
```

The documentation is configured in `mkdocs.yml` at the repository root.
