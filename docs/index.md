---
title: ResDAG
description: PyTorch-native reservoir computing — modular layers, algebraic readouts, forecasting.
hide:
  - navigation
  - toc
---

<div class="resdag-hero" markdown>

# ResDAG

<p class="tagline">
PyTorch library for reservoir computing: composable recurrent layers,
algebraic readout fitting, and autoregressive forecasting on GPU.
</p>

<div class="cta-row" markdown>
[:material-book-open-variant: Documentation](getting-started/index.md){ .md-button .md-button--primary }
[GitHub](https://github.com/El3ssar/resdag){ .md-button }
[PyPI](https://pypi.org/project/resdag/){ .md-button }
</div>

</div>

## Install

```bash
pip install resdag
```

Optional extras:

```bash
pip install "resdag[hpo]"    # Optuna hyperparameter search
pip install "resdag[docs]"   # build this documentation site
```

From source:

```bash
git clone https://github.com/El3ssar/resdag.git
cd resdag
pip install -e ".[dev]"
```

Requires Python 3.11–3.14 and PyTorch 2.x. Verify:

```bash
python -c "import resdag; print(resdag.__version__)"
```

## Overview

ResDAG implements the standard reservoir-computing workflow as `torch.nn.Module`
components wired through `pytorch_symbolic`:

1. A **reservoir layer** maps input sequences to state trajectories (weights
   typically fixed after initialization).
2. A **readout layer** maps states to targets; training is algebraic (ridge-type
   solvers), not SGD over the reservoir.
3. **`ESNModel.forecast`** runs teacher-forced warmup, then autoregressive
   generation for a chosen horizon.

Data splits for training and evaluation should use
[`prepare_esn_data`](reference/utils/data.md): it returns `warmup`, `train`,
`target`, `f_warmup`, and `val`. The tensor `f_warmup` is the **last
`warmup_steps` timesteps of `train`** — the segment immediately before the held-out
`val` series. That is the input drive the reservoir has already seen when
forecasting starts; `val` is what you score against.

## Minimal example

```python
import torch
from resdag import classic_esn
from resdag.training import ESNTrainer
from resdag.utils.data import prepare_esn_data

# (1, T, features)
data = torch.randn(1, 25_000, 1).cumsum(dim=1) * 0.01

warmup, train, target, f_warmup, val = prepare_esn_data(
    data,
    warmup_steps=2_000,
    train_steps=18_000,
    val_steps=5_000,
    normalize=True,
)

model = classic_esn(reservoir_size=500, feedback_size=1, output_size=1)
ESNTrainer(model).fit(
    warmup_inputs=(warmup,),
    train_inputs=(train,),
    targets={"output": target},
)
model.reset_reservoirs()
pred = model.forecast(f_warmup, horizon=val.shape[1])
```

## Documentation map

| Section | Contents |
|---------|----------|
| [Get started](getting-started/index.md) | Installation, mental model, first model, Lorenz example |
| [Guides](guides/index.md) | Task workflows: data prep, forecasting, HPO, ensembles, persistence |
| [Extend](extending/index.md) | Register topologies, initializers, cells, readouts, losses |
| [Reference](reference/index.md) | API generated from source docstrings |
| [About](about/index.md) | Changelog, citation, contributing, related libraries |

## Links

- Repository: [github.com/El3ssar/resdag](https://github.com/El3ssar/resdag)
- Issues: [github.com/El3ssar/resdag/issues](https://github.com/El3ssar/resdag/issues)
- Package: [pypi.org/project/resdag](https://pypi.org/project/resdag)

## Status

Current release: v0.4.0 (alpha). Public symbols are listed in
[`resdag.__init__.py`](https://github.com/El3ssar/resdag/blob/main/src/resdag/__init__.py).
