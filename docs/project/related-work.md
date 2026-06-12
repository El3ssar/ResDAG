---
description: An honest map of the reservoir computing library landscape — what ReservoirPy and others do well, and where ResDAG differs.
---

<span class="nb-kicker">Project</span>

# Related work

ResDAG is not the first reservoir computing library, and for plenty of work
it should not be your first choice. Here is the landscape, stated plainly.

## ReservoirPy

[ReservoirPy](https://github.com/reservoirpy/reservoirpy) is the reference
open-source RC library: NumPy/SciPy-based, actively maintained by Inria,
with a rich ecosystem — extensive tutorials, many node types, hyperparameter
search helpers, and years of accumulated community knowledge. If you want
classical ESNs on CPU with the broadest documentation and the most worked
examples, use ReservoirPy.

What it is not is PyTorch-native. Models are not `nn.Module`s, there is no
device story beyond NumPy, and combining reservoirs with trained deep
learning components means leaving its world.

## The wider landscape

Most other RC codebases fall into two groups: research code published with a
paper (single architecture, rarely maintained) and PyTorch ports of the
classic ESN that wrap one reservoir and one readout in a fixed pipeline.
Both serve their purpose; neither treats model *composition* — multiple
reservoirs, multiple readouts, arbitrary wiring — as the primary object.

---

## Where ResDAG differs

- **DAG composition.** Architectures are built with the `pytorch_symbolic`
  functional API — any directed acyclic graph of reservoirs, transforms, and
  readouts, not a fixed reservoir→readout pipeline.
- **PyTorch citizenship.** Every component is an `nn.Module`: `.to("cuda")`,
  dtype control, `state_dict` persistence, and SGD interop — frozen
  reservoirs and algebraic readouts coexist with gradient-trained layers in
  the same graph.
- **One-pass multi-readout training.** `ESNTrainer` fits every readout in a
  model in topological order during a single forward pass, each against its
  own targets.
- **GPU performance.** Batched trajectories and large reservoirs run on
  CUDA throughout, with a dedicated fast path for the per-step update.

## When to use something else

Choose ReservoirPy (or plain NumPy) when your reservoirs are small, your
data is a single CPU-sized trajectory, and you value worked examples over
composability — a few hundred neurons gain nothing from a GPU. Choose a
deep-learning RNN (LSTM/GRU) when you have abundant data and no need for
the one-shot training that makes reservoirs attractive in the first place.
