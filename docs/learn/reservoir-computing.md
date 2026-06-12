---
description: The one idea behind reservoir computing — train the readout, not the network — and when it beats backprop.
---

<span class="rd-eyebrow">Learn · 01</span>

# Reservoir computing, in five minutes

One idea, three consequences, and an honest account of when it wins. No
equations on this page — they live [under the hood](../under-the-hood/reservoir-equations.md)
when you want them.

## The idea

Training a recurrent network with backprop means pushing gradients through
time: expensive, unstable, and famously allergic to long dependencies.
Reservoir computing makes a bet that sounds illegal the first time you hear
it:

> **Don't train the recurrent network at all.** Wire it randomly, freeze it,
> and train only a linear layer reading from its states.

The frozen network — the *reservoir* — is a pool of hundreds of randomly
connected neurons. Drive it with your signal and each neuron computes some
nonlinear, fading mixture of the input's history. You don't control *which*
mixtures; with enough neurons you don't have to. Somewhere in those 500
trajectories is almost everything a forecaster needs — the readout's job is
just to find the right linear combination.

And finding the best linear map is not gradient descent territory. It's
least squares — a problem with a closed-form solution, solved in one shot,
in milliseconds.

## Three consequences

**Training is instant.** No epochs, no learning-rate schedule, no early
stopping. One teacher-forced pass to collect states, one ridge-regression
solve. A 500-neuron forecaster trains in well under a second on a laptop.
This is why hyperparameter search over *thousands* of full retrainings is
routine in this field rather than a luxury.

**The dynamics are yours to design.** Since nothing inside the reservoir is
learned, everything inside it is a *choice*: the connectivity graph, the
spectral radius, the leak rate, the input weights. Reservoir research is
largely the study of which structures make good dynamics — which is why
ResDAG treats topologies and initializers as first-class,
[pluggable functions](../cookbook/topologies.md).

**Stability comes for free-ish.** A reservoir tuned to the *echo state
property* forgets its initial condition: drive it long enough and its state
depends only on the input history, not on where it started. That's what the
warmup phase exploits, and it's the closest thing recurrent networks have to
a stability guarantee.

## Where it shines, where it doesn't

Reservoirs excel at **dynamical systems**: chaotic attractors (Lorenz,
Kuramoto–Sivashinsky), physiological and climate signals, control loops —
anywhere the data is a trajectory and you care about multi-step forecasts.
On these tasks a well-tuned ESN routinely matches sequence models that take
four orders of magnitude longer to train, and the speed means you can
afford proper model selection.

They are the wrong tool for discrete, structured sequence tasks — language,
symbolic reasoning — where the heavy lifting is representation learning,
exactly the thing reservoirs refuse to do.

There is also a newer twist: **Next-Generation Reservoir Computing**
replaces the random network entirely with polynomial features of delayed
inputs — fully deterministic, zero random weights, often startlingly good
on low-dimensional chaos. ResDAG ships it as a layer
([`NGReservoir`](../cookbook/ngrc.md)) that slots into the same DAGs as any
ESN.

## Where ResDAG fits

Classic RC libraries give you *the* ESN: input → reservoir → readout, take
it or leave it. ResDAG's premise is that the field moved past that diagram —
modern architectures augment states, run reservoirs in parallel, branch into
multiple readouts, couple ensembles. So it gives you the *parts* as PyTorch
layers and a [functional API](building-models.md) to wire them into any DAG,
while keeping the one-pass training and the GPU.

## Next

[**02 · Quickstart**](quickstart.md) — see all of it run in sixty seconds.
