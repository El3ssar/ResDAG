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
