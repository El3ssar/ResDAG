---
description: The composition handbook — reservoir architectures are DAGs, and ResDAG gives you the parts and a functional API to wire them.
---

<span class="nb-kicker">Build</span>

# The composition handbook

The library's thesis in one sentence: modern reservoir architectures are
directed acyclic graphs — reservoirs, augmentations, concatenations,
readouts, wired in whatever order the problem demands — so ResDAG ships
parts and a functional API rather than a zoo of monolithic model classes.
Declare symbolic inputs, call layers on them, wrap the graph in `ESNModel`.
Everything on this track is that move, repeated.

## Pattern gallery

Five wirings cover most of published reservoir computing. All five share
one import line and run as written:

```python
from resdag import (
    CGReadoutLayer, Concatenate, ESNLayer, ESNModel,
    SelectiveExponentiation, reservoir_input,
)
```

**Minimal ESN** — the architecture every other one extends.

```python
inp = reservoir_input(3)
states = ESNLayer(200, feedback_size=3)(inp)
out = CGReadoutLayer(200, 3, name="output")(states)
model = ESNModel(inp, out)          # (batch, time, 3) -> (batch, time, 3)
```

<figure markdown>
![Minimal ESN architecture](../assets/figures/arch_minimal.svg)
<figcaption>One reservoir, one readout.</figcaption>
</figure>

**Input-driven** — the first input is always the feedback signal; further
inputs are exogenous drivers.

```python
feedback = reservoir_input(1)
driver = reservoir_input(5)
states = ESNLayer(150, feedback_size=1, input_size=5)(feedback, driver)
out = CGReadoutLayer(150, 1, name="output")(states)
model = ESNModel([feedback, driver], out)
```

<figure markdown>
![Input-driven architecture](../assets/figures/arch_input_driven.svg)
<figcaption>Feedback plus a known exogenous series.</figcaption>
</figure>

**State augmentation, Ott-style** — square the even-indexed states and let
the readout see the raw input; this is exactly what `ott_esn` builds.

```python
inp = reservoir_input(3)
states = ESNLayer(500, feedback_size=3)(inp)
augmented = SelectiveExponentiation(index=0, exponent=2.0)(states)
features = Concatenate()(inp, augmented)
out = CGReadoutLayer(3 + 500, 3, name="output")(features)
model = ESNModel(inp, out)
```

<figure markdown>
![Ott ESN architecture](../assets/figures/arch_ott_esn.svg)
<figcaption>The chaotic-systems workhorse, spelled out.</figcaption>
</figure>

**Parallel two-timescale reservoirs** — a fast and a slow reservoir read
the same signal; the readout mixes their features.

```python
inp = reservoir_input(3)
fast = ESNLayer(120, feedback_size=3, leak_rate=1.0, spectral_radius=0.7)(inp)
slow = ESNLayer(120, feedback_size=3, leak_rate=0.2, spectral_radius=0.95)(inp)
merged = Concatenate()(fast, slow)
out = CGReadoutLayer(240, 3, name="output")(merged)
model = ESNModel(inp, out)
```

<figure markdown>
![Parallel reservoirs architecture](../assets/figures/arch_parallel_reservoirs.svg)
<figcaption>Two timescales, one readout.</figcaption>
</figure>

**Multi-readout** — one reservoir, two named heads, fitted together in a
single `ESNTrainer.fit` call keyed by name.

```python
inp = reservoir_input(3)
states = ESNLayer(300, feedback_size=3)(inp)
coords = CGReadoutLayer(300, 3, name="coords")(states)
energy = CGReadoutLayer(300, 1, name="energy")(states)
model = ESNModel(inp, outputs=[coords, energy])
```

<figure markdown>
![Multi-readout architecture](../assets/figures/arch_multi_readout.svg)
<figcaption>Readout names become the keys of the targets dict.</figcaption>
</figure>

---

## The track

<div class="grid cards" markdown>

- **[Layers](layers/index.md)**

    ---

    The parts catalog: every reservoir family knob by knob — `ESNLayer`
    first among them — the conjugate-gradient readout, and the transform
    glue.

- **[Architectures](architectures/index.md)**

    ---

    Six premade factories, what each one actually wires, and the
    hand-built DAGs you graduate to — stacked reservoirs, drivers,
    multiple heads.

- **[Initialization](initialization/index.md)**

    ---

    Structure is a function: graph topologies, matrix builders, bare
    callables, `torch.nn.init`, the registries that name them all — and
    the generated catalogs of every registered component.

</div>
