# Installation

## PyPI (recommended)

```bash
pip install resdag
```

Verify:

```bash
python -c "import resdag; print(resdag.__version__)"
```

## Optional extras

| Extra | Install | Provides |
|-------|---------|----------|
| HPO | `pip install resdag[hpo]` | Optuna integration (`run_hpo`, study utilities) |
| Docs | `pip install resdag[docs]` | MkDocs Material site build |
| Dev | `pip install resdag[dev]` | Tests, ruff, black, mypy, basedpyright |

Combined:

```bash
pip install "resdag[hpo,dev]"
```

## uv

```bash
uv add resdag
# or from a clone:
uv sync --dev
```

## Install from source

```bash
git clone https://github.com/El3ssar/resdag.git
cd resdag
pip install -e ".[dev]"
```

## Python version

ResDAG supports **Python 3.11–3.14** (see `pyproject.toml` classifiers).

## PyTorch and accelerators

ResDAG follows PyTorch device placement. After install:

```python
import torch
print(torch.cuda.is_available())  # NVIDIA GPU
print(torch.backends.mps.is_available())  # Apple Silicon
```

Move models and data with `.to(device)` as usual. No separate CUDA build of
ResDAG is required — use the [PyTorch install instructions](https://pytorch.org/get-started/locally/)
that match your hardware.

## Next

[Your first ESN](your-first-esn.md) or the [mental model](mental-model.md).
