---
description: How ResDAG compares to ReservoirPy and PyTorch-based RC ports — and when to pick each.
---

<span class="rd-eyebrow">About</span>

# Related work

ResDAG is not the first reservoir-computing library, and it doesn't pretend to
be. Here's the honest map of the neighborhood (features as of ResDAG 0.4),
then the three things this library does that the others don't.

| | **ResDAG** | [ReservoirPy](https://github.com/reservoirpy/reservoirpy) | EchoTorch / RcTorch-style PyTorch ports |
|---|------------|---------------------------|----------------------------------------|
| Backend | PyTorch | NumPy (+ optional PyTorch) | PyTorch |
| GPU | First-class | Partial / optional | Varies |
| Model graph | `pytorch_symbolic` DAG | Node API | Varies |
| Readout training | Ridge via CG (algebraic) | Ridge, sklearn, etc. | Often SGD |
| Topologies | 17 registered graphs | Several | Varies |
| NG-RC | `NGReservoir` | Limited / external | Rare |
| Coupled ensembles | `CoupledEnsembleESNModel` | Manual | Manual |
| HPO | Optuna integration | External | External |

**DAG composition.** Models are wired with `pytorch_symbolic`'s functional
API — call a layer on a symbolic tensor, exactly like Keras. Parallel
reservoirs, state augmentation, multiple readouts: if you can draw the graph,
you can build it, without subclassing anything or writing a forward pass.

**PyTorch citizenship.** A built — and even trained — model is an ordinary
`nn.Module`. It moves with `.to(device)`, serializes with `state_dict()`,
embeds in larger networks, and flips to gradient training with
`trainable=True`. No device copies at the NumPy border, no wrapper classes
between your reservoir and the rest of your pipeline.

**One-pass multi-readout training.** Readouts are fitted by forward
pre-hooks during a single pass, so every readout in an arbitrary DAG is
trained on exactly the features it will see at inference, in topological
order, for free. Add to that the chaos-oriented HPO losses and the Ott/NG-RC
premade models, and ResDAG is built squarely for dynamical-systems work.

**When to look elsewhere.** If you want a mature scikit-learn-style pipeline
with years of notebook tutorials and you're happy in NumPy, ReservoirPy
remains a solid choice. If your pipeline already lives in PyTorch, ResDAG
keeps everything on-device and keeps gradients available for hybrid models —
even though standard readouts are algebraic.

## Acknowledgments

ResDAG builds on [PyTorch](https://pytorch.org/), [NetworkX](https://networkx.org/),
[pytorch_symbolic](https://pytorch-symbolic.readthedocs.io/), and ideas from the
broader ESN / RC literature and community libraries.
