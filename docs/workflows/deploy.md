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

model = rd.ESN(3000, input_size=3, device="cuda").fit(training.to("cuda"))
preds = model.forecast(horizon=1000)   # original units, on cuda
```

CUDA has an optional self-hosted GPU test workflow; it was not run for this
continuation. MPS
does not support float64, which the algebraic solvers and some diagnostics use
by default. For MPS inference, fit on CPU, move a float32 model and its inputs
to `"mps"`, and validate your workload. Passing `"mps"` to device discovery
does not guarantee that every solver or diagnostic supports it.

`.to("cuda")` moves parameters and buffers, including a warmed-up reservoir
state. Device or dtype moves preserve that state's values; move the inputs to
the same device and dtype too. A different batch size initializes the configured
reset state (zeros by default), unless you deliberately restored a state with `set_state`, in
which case a mismatched batch size raises instead of discarding it.

**Measure the complete workflow.** Recurrent state depends on the previous
timestep, limiting parallelism across time. Larger batches and reservoirs
provide more matrix work per step, while small single-trajectory workloads
can spend much of their time in dispatch and kernel launches. Device, dtype,
solver and batch size all affect the crossover; there is no fixed reservoir
size at which GPU execution is guaranteed to win. Measure representative
fitting and forecasting with `examples/11_gpu_benchmark.py`, including device
transfers and synchronization. Earlier GPU numbers are
[historical measurements](benchmarks.md#archived-comparisons), not current
release guarantees. Phase-five continuation measured CPU execution and native
MPS correctness; it did not measure CUDA throughput.

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

The autoregressive loop can spend substantial time in Python and kernel
launches. `forecast(compile=True)` unrolls chunks of the flat executor into
`torch.compile` (`mode="reduce-overhead"`, `fullgraph=True`). For a hook-free
symbolic graph with ordinary next-step feedback from its first output to its
first input, the default chunk contains 64 forecast steps. This amortizes
dispatch across the chunk rather than compiling and calling each step separately:

```python
preds = model.forecast(history=observed, horizon=10_000, compile=True)
```

For hook-free models, the default eager path already uses a flat executor.
Models with PyTorch execution hooks automatically use native module calls so
their pre-hooks, output changes and backward hooks keep ordinary PyTorch
semantics. On backends that support compiled forecasts, `compile=True` emits
a warning and uses compatible eager execution while hooks are registered.
Removing the hooks restores the lean path.
Named non-first, multiple or delayed feedback routes use the sequence executor;
they do not use this 64-step flat rollout. Full-history execution and the
parallel/coupled workflow backends reject `compile=True`. Check the
[execution capabilities](../advanced/forecast-capabilities.md) for your graph.
Compiled results should
be compared with floating-point tolerances, especially for chaotic dynamics;
compilation can change rounding. Warm compiled executors are reused for
matching shapes, devices, and dtypes. `compile=True` is an opt-in execution choice and
transparently falls back to eager on `torch < 2.10` or a compilation failure
(emitting a `RuntimeWarning`). Measure cold compilation separately from repeated
warm calls: setup can cost more than it saves, and a long horizon alone does
not guarantee improvement. The low-level `ESNModel.forecast` API also supports
a differentiable `no_grad=False` rollout, which is mutually exclusive with
`compile=True` ([Train · Differentiable
forecast](train.md#path-4-differentiable-forecast-bptt-through-the-rollout)) —
requesting both raises. The common `rd.ESN`/`rd.Model.forecast` workflow performs
inference; use ordinary module calls, explicit states or the raw graph API for
gradient-based composition.

!!! tip "Compiling the teacher-forced pass too"
    Warmup and training run the teacher-forced sequence loop, which also
    accepts an opt-in `compile=True`: `reservoir(feedback, compile=True)`
    routes it through a compiled chunked scan (same `reduce-overhead` mode,
    same eager fallback) when CUDA inference is eligible. CPU and MPS use
    eager execution for this scan, as do calls requiring gradients. This
    control belongs to the reservoir call; the common `model.fit(...)` API
    has no `compile` option. Compare setup and repeated-call timing on your
    own workload; the default remains the eager loop.

---

## Persistence

The common `rd.ESN` and `rd.Model` workflow saves a complete model by default:

```python
model = rd.ESN(reservoir_size=500, seed=7).fit(series)
model.save("run.pt")
restored = rd.load("run.pt", map_location="cpu")
prediction = restored.forecast(horizon=1000)
model.save("run.svg")
```

The checkpoint describes the graph and includes fitted preprocessing, observed
history and current reservoir memory. A larger PyTorch application can instead
save `model.state_dict()` and load it into a matching architecture. Ordinary
forward memory supports continuing a stream; autonomous forecasts reconstruct
delayed feedback queues from history each call. Saving reservoir tensors alone
does not preserve a forecast cursor. See the [complete persistence guide](save-load.md)
for registered components, explicitly trusted notebook bundles and formats.

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

`stateful=False` starts each minibatch from the configured initial state
(zeros by default) and never changes stored
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
