# Embedding ResDAG in a larger PyTorch pipeline

ResDAG components are plain `torch.nn.Module`s. A trained reservoir is a
GPU-friendly nonlinear feature extractor that you can drop into any
PyTorch pipeline — classifiers, multi-task heads, downstream RNNs,
diffusion-style score models, whatever your day needs.

This example shows the *frozen-features + SGD head* pattern: train a
reservoir-based feature extractor algebraically (one ridge solve), then
attach a small trainable head and finetune it on a downstream task with
SGD.

The full pattern, including a fully-trainable variant, is in
[`examples/12_pipeline_integration.py`](https://github.com/El3ssar/resdag/blob/main/examples/12_pipeline_integration.py)
in the repo.

## Setup

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from resdag import headless_esn
```

We use [`headless_esn`](../guides/builtin-models.md#headless_esn) — an
`ESNModel` with no readout. Its output is the reservoir state for every
timestep.

## Building the pipeline

The pattern is:

```text
input → headless reservoir (frozen) → torch.nn head (trainable)
```

```python
class ReservoirClassifier(nn.Module):
    """ESN feature extractor + 2-layer MLP head trained with SGD."""

    def __init__(
        self,
        reservoir_size: int = 400,
        input_size: int = 3,
        n_classes: int = 10,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        # 1. Reservoir as feature extractor.  Frozen by default.
        self.reservoir = headless_esn(
            reservoir_size=reservoir_size,
            feedback_size=input_size,
            spectral_radius=0.9,
            leak_rate=0.5,
        )
        # 2. Trainable head.
        self.head = nn.Sequential(
            nn.Linear(reservoir_size, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_size)
        self.reservoir.reset_reservoirs()
        states = self.reservoir(x)              # (B, T, reservoir_size)
        pooled = states.mean(dim=1)             # mean over time
        return self.head(pooled)                # (B, n_classes)
```

The reservoir's weights start frozen (`trainable=False` is the default
on `ESNLayer`) so the SGD optimizer only sees the head's parameters.
That makes training fast and stable — backprop never enters the
recurrent loop.

## Training

```python
model = ReservoirClassifier(reservoir_size=400, input_size=3, n_classes=10)
optim = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],   # head only
    lr=2e-3, weight_decay=1e-4,
)

for epoch in range(20):
    for x_batch, y_batch in dataloader:           # (B, T, 3), (B,)
        optim.zero_grad()
        logits = model(x_batch)
        loss = F.cross_entropy(logits, y_batch)
        loss.backward()
        optim.step()
```

A more advanced variant unfreezes the reservoir for finetuning:

```python
# Phase 2: unfreeze the reservoir for joint finetuning
for p in model.reservoir.parameters():
    p.requires_grad_(True)

optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
# … same training loop …
```

## Inspecting which parameters are trainable

`model.plot_model()` doesn't apply to arbitrary `nn.Module`s, but
ResDAG's helper works on the embedded reservoir:

```python
model.reservoir.plot_model(show_trainable=True, save_path="esn_block.svg")
print("trainable params:",
      sum(p.numel() for p in model.parameters() if p.requires_grad))
print("total params:    ",
      sum(p.numel() for p in model.parameters()))
```

The 🔒/🔓 padlock on the reservoir node confirms its frozen status.

## When this is the right pattern

- **Sequence classification / regression** where you need a single
  embedding per sequence — the mean-pool above is the simplest choice;
  attention pooling or `states[:, -1, :]` work too.
- **Pretraining a reservoir** algebraically (with `ESNTrainer` and a
  `CGReadoutLayer`) and then *reusing* its frozen features for many
  downstream heads.
- **Mixed algebraic + SGD workflows** — e.g. fit one readout
  algebraically as a baseline, then attach a small SGD-trained head
  that predicts the *residual* of the algebraic head.

## When it isn't

- If your downstream task is itself just a regression and you don't
  need backprop, skip the SGD head entirely: a `CGReadoutLayer`
  trained by `ESNTrainer` is faster and matches what classical ESN
  literature does.
- If your data is a single long autoregressive series and you want a
  forecast, you want the [classic_esn workflow](../getting-started/your-first-esn.md),
  not this pattern.

## See also

- [`headless_esn`](../guides/builtin-models.md#headless_esn) reference.
- [`linear_esn`](../guides/builtin-models.md#linear_esn) — same idea
  with a `torch.nn.Linear` readout already attached.
- [Example 12 in the repo](https://github.com/El3ssar/resdag/blob/main/examples/12_pipeline_integration.py)
  for fully-trainable and `nn.Sequential` variants.
