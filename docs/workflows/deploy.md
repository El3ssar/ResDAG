---
description: GPU regimes, model persistence, and embedding frozen reservoirs in larger PyTorch pipelines.
---

<span class="nb-kicker">Work · Scale & deploy</span>

# Scale & deploy

A ResDAG model is a standard `torch.nn.Module`: it moves to the GPU,
checkpoints, and embeds in a larger network like any other module.
The reservoir-specific part is knowing when each of those is worth doing.

## GPU

```python
import resdag as rd

model = rd.ott_esn(reservoir_size=3000, feedback_size=3, output_size=3).to("cuda")
x = x.to("cuda")

rd.ESNTrainer(model).fit((warmup.cuda(),), (train.cuda(),), {"output": target.cuda()})
preds = model.forecast(f_warmup.cuda(), horizon=1000)   # stays on cuda
```

CUDA is the accelerated training backend covered by the GPU test lane. MPS
does not support float64, which the algebraic solvers and some diagnostics use
by default. For MPS inference, fit on CPU, move a float32 model and its inputs
to `"mps"`, and validate your workload. Passing `"mps"` to device discovery
does not guarantee that every solver or diagnostic supports it.

`.to("cuda")` moves parameters and buffers, including a warmed-up reservoir
state. Device or dtype moves preserve that state's values; move the inputs to
the same device and dtype too. A different batch size initializes a fresh
zero state, unless you deliberately restored a state with `set_state`, in
which case a mismatched batch size raises instead of discarding it.

**When the GPU helps.** The reservoir loop is sequential in time, so each
timestep is one small kernel launch; the GPU only wins once those kernels
carry real work. In practice:

- **Wins from ~2k units or batched trajectories.** Speedups grow with
  batch and reservoir size — `fit()` reaches up to ~20× at batch 16 with
  3000-unit reservoirs, where the readout's Gram formation becomes heavy
  GPU work.
- **Tiny configs are launch-bound.** A single trajectory through a
  1000-unit reservoir runs at CPU parity or slower; the GPU mostly waits.
- As a rule of thumb, large batches, large reservoirs, or many models
  favor the GPU; a single small model runs as fast or faster on CPU.
  Measure your own configuration with `examples/11_gpu_benchmark.py`.

!!! note "Readout precision and GPU throughput"
    Direct readouts such as `RidgeReadoutLayer` form their normal equations
    in the solve dtype by default: float64 with `use_float64=True`, on CPU
    and CUDA. This favors accuracy for correlated reservoir features, but
    float64 throughput can be substantially lower on consumer GPUs. For
    normalized, sufficiently regularized problems, explicitly setting
    `gram_dtype=torch.float32` can improve throughput; compare forecast
    accuracy before adopting that tradeoff. `CGReadoutLayer` has a different
    default: it forms the Gram matrix in the input dtype on CUDA and uses
    float64 for the subsequent solve when `use_float64=True`. Pass
    `gram_dtype=torch.float64` there when the input-dtype Gram loses accuracy.

### Long horizons: compile the step

The autoregressive forecast loop is sequential in time — one small kernel
per step — so on the GPU a long rollout spends most of its wall-clock in
Python and kernel-launch overhead, not arithmetic. `forecast(compile=True)`
wraps the flat single-step engine in `torch.compile` (`mode="reduce-overhead"`,
`fullgraph=True`), fusing the per-step work into one graph and cutting that
overhead:

```python
preds = model.forecast(f_warmup, horizon=10_000, compile=True)   # GPU: fewer, fatter launches
```

The default eager path is already flat and hook-free. Compiled results should
be compared with floating-point tolerances, especially for chaotic dynamics;
compilation can change rounding. Warm compiled executors are reused for
matching shapes, devices, and dtypes. `compile=True` is an opt-in speedup and
transparently falls back to eager on `torch < 2.10` or a compilation failure
(emitting a `RuntimeWarning`). It pays off for **long horizons on the GPU**;
for short rollouts or on CPU the compile warmup can cost more than it saves, so
leave it off. It is mutually exclusive with the differentiable
`no_grad=False` rollout ([Train · Differentiable
forecast](train.md#path-4-differentiable-forecast-bptt-through-the-rollout)) —
requesting both raises.

!!! tip "Compiling the teacher-forced pass too"
    Warmup and training run the teacher-forced sequence loop, which also
    accepts an opt-in `compile=True`: `reservoir(feedback, compile=True)`
    routes it through a compiled chunked scan (same `reduce-overhead` mode,
    same eager fallback). Worth it for very long teacher-forced passes on the
    GPU; the default is byte-for-byte the eager loop.

---

## Persistence

There are two ways to persist a model. **State-dict** (`save`/`load`)
stores weights only — compact and the safe choice for long-term archival
— but the architecture is **not** serialized, so you keep a build
function and load into a fresh instance:

```python
def build():                       # same factory call = same architecture
    return rd.ott_esn(reservoir_size=500, feedback_size=3, output_size=3)

model.save("model.pt")                                   # weights + metadata
restored = rd.ESNModel.load_from_file("model.pt", model=build(), map_location="cpu")

model.save("ckpt.pt", include_states=True, epoch=10, val_mse=0.012)
restored.load("ckpt.pt", load_states=True)               # weights + live states
```

`include_states=True` makes checkpoints resumable mid-sequence: warm up,
save, and a later process can `load(..., load_states=True)` and call
`forecast(..., reset=False)` to continue exactly where this one stopped.
Metadata is a plain `torch.load` payload — inspectable without building
the model.

**Whole-model** (`save_full`/`load_full`) serializes everything — the
graph, weights, and reservoir states — in one file, so there is no build
function to keep in sync:

```python
model.save_full("model_full.pt", epoch=10)               # everything
restored = rd.ESNModel.load_full("model_full.pt")        # no rebuild needed
restored.forecast(f_warmup, horizon=1000)
```

This rides on the pickling support added in `pytorch-symbolic` 1.2.
`load_full` unpickles arbitrary objects (`weights_only=False`), so only
open files you trust — for sharing weights publicly, prefer the state-dict
form. One constraint: any custom callable you pass as a `topology`,
`*_initializer`, or `activation` spec must be importable (a module-level
`def`, not a `lambda`) for the model to pickle; string, tuple, and
registered-object specs always work. `CoupledEnsembleESNModel` exposes the
same `save_full`/`load_full` pair.

## Inside a larger pipeline

A frozen reservoir is a feature extractor with zero trainable parameters
— `sum(p.numel() for p in model.parameters() if p.requires_grad)` is 0 by
design — so optimizers, gradient bookkeeping, and checkpoint diffs all
treat it as a constant. Wrap it like any frozen backbone:

<div class="nb-specimen" data-label="reservoir_classifier.py" markdown>

```python
import torch
import torch.nn as nn
import resdag as rd

class ReservoirClassifier(nn.Module):
    """Frozen reservoir features, gradient-trained head."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.features = rd.ReservoirFeatureExtractor(
            reservoir_size=500, feedback_size=3, stateful=False,
        )
        self.head = nn.Sequential(
            nn.Linear(500, 128), nn.ReLU(), nn.Linear(128, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (batch, time, 3)
        states = self.features(x)                          # (batch, time, 500)
        return self.head(states[:, -1])                    # classify on last state

clf = ReservoirClassifier(n_classes=4)
opt = torch.optim.Adam(clf.head.parameters(), lr=1e-3)

for xb, yb in loader:                                      # (B, T, 3), (B,)
    loss = nn.functional.cross_entropy(clf(xb), yb)
    opt.zero_grad()
    loss.backward()
    opt.step()
```

</div>

`stateful=False` starts each minibatch from zeros and never changes stored
reservoir states, so independent or shuffled sequences do not leak history
into each other. The default `stateful=True` carries state for streaming;
reset it between independent trajectories. Gradients flow *through* a frozen
reservoir to a trainable encoder upstream, while its weights stay fixed.
To fine-tune a readout
instead of adding a separate head, use the combination pattern from
[Train](train.md): build it with `trainable=True`, solve algebraically,
then continue with a small learning rate.

Reuse a trained model's reservoirs with
`rd.ReservoirFeatureExtractor.from_model(trained_model, stateful=False)`.
This shares reservoir weights and extracts their features; it does not include
the trained readout or reproduce arbitrary parallel branches and transforms.
To retain the complete trained model, place `trained_model` itself inside your
larger `nn.Module` and reset its reservoirs between independent sequences.

For explicit state threading, call
`features, states = extractor.forward_stateless(x)` and continue with
`extractor.forward_stateless(next_chunk, states=states)`. This leaves stored
states untouched and keeps gradients across chunks; detach the returned states
yourself when you want truncated backpropagation.

One operational note: reservoir layers carry mutable state, so a model
instance is not thread-safe — give each thread its own model.
`CoupledEnsembleESNModel.fit(n_workers=...)` parallelizes exactly this
way, one sub-model per thread.

## See also

- [Build · Architectures](../build/architectures/index.md) — premade model factories
- [Reference · Core](../reference/core.md) — `save`, `load`, `load_from_file` in full
