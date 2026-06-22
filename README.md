<p align="center">
  <img src="https://el3ssar.github.io/ResDAG/assets/logo.svg" width="88" alt="ResDAG">
</p>

<h1 align="center">ResDAG</h1>

<p align="center"><strong>Reservoir computing for PyTorch.</strong><br>
Compose reservoir models as DAGs, train readouts with a single algebraic solve, run it all on the GPU.</p>

<p align="center">
  <a href="https://pypi.org/project/resdag/"><img src="https://img.shields.io/pypi/v/resdag" alt="PyPI"></a>
  <a href="https://pypi.org/project/resdag/"><img src="https://img.shields.io/pypi/pyversions/resdag" alt="Python"></a>
  <a href="https://github.com/El3ssar/ResDAG/actions/workflows/ci.yml"><img src="https://github.com/El3ssar/ResDAG/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://codecov.io/gh/El3ssar/ResDAG"><img src="https://codecov.io/gh/El3ssar/ResDAG/branch/main/graph/badge.svg" alt="Coverage"></a>
  <a href="https://el3ssar.github.io/ResDAG/"><img src="https://img.shields.io/badge/docs-el3ssar.github.io%2FResDAG-2a63d4" alt="Documentation"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT"></a>
</p>

---

ResDAG treats reservoir models — echo state networks and beyond — as ordinary PyTorch layers. Reservoirs, readouts, and transforms are `nn.Module`s wired together with a functional API; training a readout is one teacher-forced pass and one conjugate-gradient ridge solve, with no epochs. Models move with `.to(device)`, serialize with `state_dict()`, and embed in larger PyTorch pipelines, optimizers included.

**[Documentation →](https://el3ssar.github.io/ResDAG/)**

## Installation

```bash
pip install resdag            # core
pip install "resdag[hpo]"     # + Optuna hyperparameter optimization
```

Python ≥ 3.11, PyTorch ≥ 2.10.

## Try it

A reservoir forecaster on a toy signal, end to end:

```python
import torch
import resdag as rd

t = torch.linspace(0, 60, 3000)
data = torch.sin(t).reshape(1, -1, 1)            # (batch, time, features)

warmup, train, target, f_warmup, val = rd.utils.prepare_esn_data(
    data, warmup_steps=100, train_steps=2000, val_steps=300)

model = rd.models.classic_esn(reservoir_size=300, feedback_size=1, output_size=1)
rd.ESNTrainer(model).fit((warmup,), (train,), targets={"output": target})

prediction = model.forecast(f_warmup, horizon=300)   # autoregressive, (1, 300, 1)
```

The [first forecast](https://el3ssar.github.io/ResDAG/start/first-forecast/) walkthrough does the same on the Lorenz attractor, with the math explained.

## Compose, don't configure

Architectures are DAGs you wire, not options you toggle. Two reservoirs on different timescales, read out together:

```python
from resdag import CGReadoutLayer, Concatenate, ESNModel, reservoir_input
from resdag.layers import ESNLayer

inp    = reservoir_input(3)
fast   = ESNLayer(64, feedback_size=3, leak_rate=1.0, spectral_radius=0.9)(inp)
slow   = ESNLayer(64, feedback_size=3, leak_rate=0.2, spectral_radius=0.9)(inp)
merged = Concatenate()(fast, slow)
model  = ESNModel(inp, CGReadoutLayer(128, 3, name="output")(merged))
```

<p align="center">
  <img src="https://el3ssar.github.io/ResDAG/assets/figures/readme/parallel_timescales.svg" width="640" alt="Two parallel reservoirs feeding one readout">
</p>

Branches, feature augmentations, and multiple readout heads compose the same way — one reservoir, squared-state augmentation, two heads:

<p align="center">
  <img src="https://el3ssar.github.io/ResDAG/assets/figures/readme/augmented_two_heads.svg" width="640" alt="Augmented states feeding two readout heads">
</p>

All heads fit in a single pass, in dependency order. The [composition handbook](https://el3ssar.github.io/ResDAG/build/) covers the patterns.

## Coming from scikit-learn?

The whole `fit` → `predict` loop fits in one object. `ESN.fit(series)` slices the warmup window, builds the one-step-ahead target, and runs the algebraic solve; `forecast(horizon=...)` re-synchronizes and rolls out — numpy in, numpy out:

```python
import numpy as np
from resdag import ESN

series = np.cumsum(np.random.randn(2000, 3), axis=0)   # (time, features)

esn = ESN(reservoir_size=300, spectral_radius=0.9).fit(series)
prediction = esn.forecast(horizon=200)                 # (200, 3)
```

`esn.model` drops you back into the full composable graph whenever you outgrow the facade. The [mental model](https://el3ssar.github.io/ResDAG/start/concepts/) maps `fit`/`predict` onto ResDAG's `warmup` + `ESNTrainer.fit` / `forecast` flow.

## Train through it with SGD

A reservoir is an ordinary PyTorch layer, so it drops into a pipeline as a frozen feature extractor and you train any head with a normal optimizer loop. The reservoir has zero trainable parameters, so the optimizer only ever touches the head:

```python
import torch
import torch.nn as nn
from resdag import ReservoirFeatureExtractor

net = nn.Sequential(
    ReservoirFeatureExtractor(500, feedback_size=3, spectral_radius=0.9),
    nn.Linear(500, 64), nn.Tanh(), nn.Linear(64, n_classes),
)
extractor, head = net[0], net[1:]
opt = torch.optim.Adam(head.parameters(), lr=1e-3)     # head only — reservoir is frozen

with torch.no_grad():                                  # frozen features: compute once
    extractor.on_epoch_start()
    feats = extractor(sequences)[:, -1]                # (batch, 500) last-step summary

for _ in range(300):
    loss = nn.functional.cross_entropy(head(feats), labels)
    opt.zero_grad(); loss.backward(); opt.step()
```

This is the pure-PyTorch path: gradient heads, full BPTT through the recurrence (`trainable=True`), and embedding frozen reservoirs in larger networks. [Work · Train](https://el3ssar.github.io/ResDAG/workflows/train/) covers all three training paths; [Work · Scale & deploy](https://el3ssar.github.io/ResDAG/workflows/deploy/) shows the frozen-backbone classifier inside a `nn.Module` pipeline.

## Documentation

| | |
| --- | --- |
| [Start](https://el3ssar.github.io/ResDAG/start/) | Install, a first trained forecaster, the mental model |
| [Build](https://el3ssar.github.io/ResDAG/build/) | Layers, readouts, architectures, topologies, initializers |
| [Work](https://el3ssar.github.io/ResDAG/workflows/) | Training paths, forecasting with drivers, tuning, GPU |
| [Theory](https://el3ssar.github.io/ResDAG/theory/) | Every equation, stated against the code |
| [Reference](https://el3ssar.github.io/ResDAG/reference/) | The full public API |

## Ecosystem

Built on [pytorch_symbolic](https://github.com/gahaalt/pytorch-symbolic) for graph composition. Pairs with [TSDynamics](https://github.com/El3ssar/TSDynamics), a companion library of dynamical systems — it generates the systems, ResDAG forecasts them.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) — releases are automated from conventional commits, and most component types are one registry decorator away.

## License

MIT — © Daniel Estevez-Moya
