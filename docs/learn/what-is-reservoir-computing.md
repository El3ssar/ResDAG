# What is reservoir computing?

!!! info "Why this exists"
    Reservoir computing (RC) is the family of methods where a **fixed, nonlinear
    dynamical system** maps inputs into a high-dimensional state space, and only a
    **linear readout** is trained. ResDAG implements this pattern in PyTorch so you
    can compose reservoirs like any other `nn.Module` stack.

## A short history

| Year / work | Contribution |
|-------------|--------------|
| **Jaeger (2001)** | Echo State Networks (ESNs): train only the readout; recurrent weights stay random and fixed. |
| **Maass (2002)** | Liquid State Machines (LSMs): biologically motivated reservoirs with spiking dynamics. |
| **Verstraeten et al. (2007)** | Unified the field under *reservoir computing* and practical training recipes. |
| **Gauthier et al. (2021)** | Next-Generation RC (NG-RC): rich features from delayed inputs + polynomials, **no recurrent weights**. |
| **Ott et al. (2018)** | State augmentation for chaotic forecasting — implemented in ResDAG as `ott_esn`. |

ResDAG also builds on the open ecosystem cited in the project README: [PyTorch](https://pytorch.org/), [NetworkX](https://networkx.org/) for graphs, [pytorch_symbolic](https://pytorch-symbolic.readthedocs.io/) for model graphs, and ideas from libraries such as [ReservoirPy](https://github.com/reservoirpy/reservoirpy).

## The core picture

At time step $t$, a leaky ESN updates its state $x_t$ and produces an output $y_t$:

$$
x_t = (1-\lambda)\, x_{t-1} + \lambda\, \tanh\!\big(W_{\mathrm{in}} u_t + W_{\mathrm{fb}} y_{t-1} + W_{\mathrm{res}} x_{t-1}\big)
$$

$$
y_t = W_{\mathrm{out}}\, x_t
$$

Only $W_{\mathrm{out}}$ is learned (ridge regression). $W_{\mathrm{res}}$, $W_{\mathrm{in}}$, and $W_{\mathrm{fb}}$ are set once at initialization and frozen.

**Why frozen randomness works:** a large, nonlinear, fading-memory system acts as a universal feature map for temporal inputs. The readout only needs to pick a linear combination of those features to approximate the target dynamics — analogous to random features in kernel methods, but with memory.

## What ResDAG adds

- **PyTorch-native** tensors, GPU, and `nn.Module` composition.
- **Graph topologies** for $W_{\mathrm{res}}$ (not only dense random matrices).
- **Algebraic training** via conjugate-gradient ridge (`CGReadoutLayer`, `ESNTrainer`).
- **Forecasting API** on `ESNModel` (warmup + autoregressive loop).
- **NG-RC** and **coupled ensembles** as first-class modules.

## Minimal example

```python
from resdag import ott_esn
from resdag.training import ESNTrainer

model = ott_esn(reservoir_size=300, feedback_size=3, output_size=3)
ESNTrainer(model).fit(
    warmup_inputs=(warmup,),
    train_inputs=(train,),
    targets={"output": target},
)
future = model.forecast(f_warmup, horizon=200)
```

## See also

- [Reservoir layers](reservoir-layers.md) — how ResDAG splits cells and layers
- [Readouts](readouts.md) — fitting $W_{\mathrm{out}}$
- [Premade models](../reference/models.md) — `classic_esn`, `ott_esn`, …
