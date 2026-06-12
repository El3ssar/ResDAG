---
description: GPU regimes, model persistence, and embedding frozen reservoirs in larger PyTorch pipelines.
---

<span class="nb-kicker">Work · Scale & deploy</span>

# Scale & deploy

A ResDAG model is a `torch.nn.Module` to the end: it moves to the GPU
like one, checkpoints like one, and drops into a larger network like one.
The reservoir-specific part is knowing when each of those is worth doing.

## GPU

```python
import resdag as rd

model = rd.ott_esn(reservoir_size=3000, feedback_size=3, output_size=3).to("cuda")
x = x.to("cuda")

rd.ESNTrainer(model).fit((warmup.cuda(),), (train.cuda(),), {"output": target.cuda()})
preds = model.forecast(f_warmup.cuda(), horizon=1000)   # stays on cuda
```

`.to("cuda")` moves parameters and buffers as usual. The live reservoir
state needs no handling of its own — it follows the data, re-initializing
on the next forward pass whenever the incoming batch's size, device, or
dtype changes. The flip side: a warmed-up state does not survive a device
hop, so move the model first, warm up after.

**When it pays.** The reservoir loop is sequential in time, so each
timestep is one small kernel launch; the GPU only wins once those kernels
carry real work. In practice:

- **Wins from ~2k units or batched trajectories.** Speedups grow with
  batch and reservoir size — `fit()` reaches up to ~20× at batch 16 with
  3000-unit reservoirs, where the readout's Gram formation becomes heavy
  GPU work.
- **Tiny configs are launch-bound.** A single trajectory through a
  1000-unit reservoir runs at CPU parity or slower; the GPU mostly waits.
- Rule of thumb: big batches, big reservoirs, or many models → GPU; one
  small model → CPU. Measure your own pair with
  `examples/11_gpu_benchmark.py`.

!!! note "Why fit() stays fast on CUDA"
    `CGReadoutLayer`'s `gram_dtype` is automatic: the heavy Gram-matrix
    matmuls run in float64 on CPU (cheap there) but in the input dtype on
    CUDA — consumer GPUs run float64 at 1/32–1/64 throughput, the classic
    reason ESN training used to measure *slower* on GPU. Pass
    `gram_dtype=torch.float64` only for badly scaled states (e.g.
    unnormalized inputs concatenated into the readout); the better fix is
    normalizing the data.

---

## Persistence

`save()` stores the state dict, optional reservoir states, and any
metadata kwargs. The architecture is **not** serialized — keep a build
function and load into a fresh instance:

```python
def build():                       # same factory call = same architecture
    return rd.ott_esn(reservoir_size=500, feedback_size=3, output_size=3)

model.save("model.pt")                                   # weights + metadata
restored = rd.ESNModel.load_from_file("model.pt", model=build())

model.save("ckpt.pt", include_states=True, epoch=10, val_mse=0.012)
restored.load("ckpt.pt", load_states=True)               # weights + live states
```

`include_states=True` makes checkpoints resumable mid-sequence: warm up,
save, and a later process can `load(..., load_states=True)` and call
`forecast(..., reset=False)` to continue exactly where this one stopped.
Metadata is a plain `torch.load` payload — inspectable without building
the model.

## Inside a larger pipeline

A frozen reservoir is a feature extractor with zero trainable parameters
— `sum(p.numel() for p in model.parameters() if p.requires_grad)` is 0 by
design — so optimizers, gradient bookkeeping, and checkpoint diffs all
treat it as the constant it is. Wrap it like any frozen backbone:

<div class="nb-specimen" data-label="reservoir_classifier.py" markdown>

```python
import torch
import torch.nn as nn
import resdag as rd

class ReservoirClassifier(nn.Module):
    """Frozen reservoir features, gradient-trained head."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.features = rd.headless_esn(reservoir_size=500, feedback_size=3)
        self.head = nn.Sequential(
            nn.Linear(500, 128), nn.ReLU(), nn.Linear(128, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (batch, time, 3)
        self.features.reset_reservoirs()                   # independent sequences
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

Gradients flow *through* the reservoir to anything upstream; nothing
inside it moves. Between forward calls the stored state is detached, so
this loop needs no special handling for autograd. To fine-tune a readout
instead of bolting on a head, use the combination pattern from
[Train](train.md): build it with `trainable=True`, solve algebraically,
then continue with a small learning rate.

One operational note: reservoir layers carry mutable state, so a model
instance is not thread-safe — give each thread its own model.
`CoupledEnsembleESNModel.fit(n_workers=...)` parallelizes exactly this
way, one sub-model per thread.

## See also

- [Build · Architectures](../build/architectures.md) — the factories worth scaling
- [Reference · Core](../reference/core.md) — `save`, `load`, `load_from_file` in full
