---
description: Use ResDAG models inside ordinary PyTorch pipelines — frozen feature extractors, algebraic-then-SGD fine-tuning, and fully trainable BPTT.
---

<span class="rd-eyebrow">Cookbook</span>

# PyTorch pipelines & SGD

Three working training loops: a frozen reservoir feeding a trainable head,
an algebraically fitted readout polished with SGD, and a reservoir trained
end-to-end by BPTT. An `ESNModel` is a plain `nn.Module` — autograd,
optimizers, and `nn.Sequential` all just work.

## Pattern 1 — frozen reservoir, trainable head

The reservoir (frozen by default) is a fixed nonlinear feature extractor; a small MLP head does the learning.

<div class="rd-window" data-title="frozen_features.py" markdown>

```python
import torch
import torch.nn as nn
from resdag.models import headless_esn

class ReservoirRegressor(nn.Module):
    def __init__(self, reservoir_size: int = 400, n_features: int = 3):
        super().__init__()
        self.features = headless_esn(             # ESNModel with no readout
            reservoir_size=reservoir_size,
            feedback_size=n_features,
            spectral_radius=0.9,
        )
        self.head = nn.Sequential(
            nn.Linear(reservoir_size, 64), nn.Tanh(), nn.Linear(64, n_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        states = self.features(x)                 # (batch, time, 400) — frozen
        return self.head(states)                  # (batch, time, 3)

model = ReservoirRegressor()
opt = torch.optim.Adam(model.head.parameters(), lr=1e-3)
x, y = torch.randn(32, 100, 3), torch.randn(32, 100, 3)   # (batch, time, features)

for step in range(100):
    pred = model(x)
    loss = nn.functional.mse_loss(pred, y)
    opt.zero_grad(); loss.backward(); opt.step()
```

</div>

No `reset_reservoirs()` in the loop, and no "backward through the graph a
second time" error: the layer detaches its stored state at the end of
every forward call, so gradients never leak across iterations. Carrying
state between batches is a modeling choice, not a gradient hazard.

!!! note "Frozen means *zero* trainable parameters"
    In a default ResDAG model every `requires_grad` is `False`, so
    `Adam([p for p in model.parameters() if p.requires_grad])` raises
    "optimizer got an empty parameter list" — a loud failure by design,
    better than silently training nothing. Hand the optimizer the
    head's parameters, as above.

## Pattern 2 — algebraic fit, then SGD polish

Solve the readout in closed form first, then unfreeze it and let a few
epochs of SGD adapt it to a non-quadratic loss. The ridge solution is an
excellent initialization, so a small learning rate suffices.

```python
from resdag import ESNTrainer, CGReadoutLayer
from resdag.models import classic_esn

warmup, train = torch.randn(1, 100, 3), torch.randn(1, 2000, 3)
target = torch.roll(train, -1, dims=1)             # next-step prediction

model = classic_esn(reservoir_size=300, feedback_size=3, output_size=3)
ESNTrainer(model).fit(
    warmup_inputs=(warmup,),
    train_inputs=(train,),
    targets={"output": target},
)

# Selectively unfreeze just the readout — the reservoir stays fixed.
readout = next(m for m in model.modules() if isinstance(m, CGReadoutLayer))
for p in readout.parameters():
    p.requires_grad_(True)

opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
for epoch in range(5):
    model.warmup(warmup)              # teacher-forced sync, resets state first
    pred = model(train)
    loss = nn.functional.smooth_l1_loss(pred, target)   # any loss you like now
    opt.zero_grad(); loss.backward(); opt.step()
```

## Pattern 3 — fully trainable, from scratch (BPTT)

When the random reservoir is the bottleneck, set `trainable=True` and
backpropagate through time. Clip gradients — recurrence amplifies them —
and remember the Echo State Property is no longer guaranteed once the
weights move; watch long-forecast stability.

```python
from resdag import ESNLayer, ESNModel, reservoir_input

inp = reservoir_input(3)
res = ESNLayer(200, feedback_size=3, spectral_radius=0.9, trainable=True)(inp)
head = nn.Linear(200, 3)(res)
model = ESNModel(inp, head)

opt = torch.optim.Adam(model.parameters(), lr=5e-3)
for step in range(200):                            # x, y as in Pattern 1
    model.reset_reservoirs()
    pred = model(x)
    loss = nn.functional.mse_loss(pred, y)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    opt.step()
```

If you need gradients to flow through state carried *across* forward calls,
set `layer.detach_state_between_calls = False` and manage the retained
graphs yourself.

## Related

- [Save & load](save-load.md) — checkpointing SGD-trained heads.
- [GPU & performance](gpu.md) — where the time actually goes in these loops.
- [Training](../learn/training.md) — the algebraic path these patterns extend.
