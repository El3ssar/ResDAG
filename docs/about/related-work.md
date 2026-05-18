# Related work

Honest comparison with common reservoir-computing ecosystems (features as of ResDAG 0.4).

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
| Docs | MkDocs + mkdocstrings | Sphinx / notebooks | Varies |

**ResDAG strengths:** PyTorch-native composition, chaos-oriented HPO losses, Ott/NG-RC
premade models, ensemble forecasting API.

**When to look elsewhere:** If you need a mature scikit-learn-style pipeline with
years of notebook tutorials only in NumPy, ReservoirPy remains a solid choice.
If you already live in PyTorch for the rest of your pipeline, ResDAG avoids
device copies and keeps gradients available for hybrid models (even though
standard readouts are algebraic).

## Acknowledgments

ResDAG builds on [PyTorch](https://pytorch.org/), [NetworkX](https://networkx.org/),
[pytorch_symbolic](https://pytorch-symbolic.readthedocs.io/), and ideas from the
broader ESN / RC literature and community libraries.
