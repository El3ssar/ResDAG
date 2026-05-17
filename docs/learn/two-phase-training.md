# Two-phase training

!!! info "Why this exists"
    Readout weights need reservoir states that reflect **real input dynamics**, not
    zeros. ResDAG separates **warmup** (sync states) from **fitting** (one forward pass
    that solves ridge regression in topological order).

## Phase 1 — Warmup (teacher forcing)

Run the full model on `warmup_inputs` with gradients disabled. Reservoirs update
their internal state step by step following the **true** input sequence — the same
inputs you will use before forecasting.

```python
model.reset_reservoirs()
model.warmup(warmup_feedback, warmup_driver)  # optional drivers
```

After warmup, `reservoir.state` encodes where the system "is" on the attractor.

## Phase 2 — Fit readouts (single forward + hooks)

`ESNTrainer.fit` registers **forward pre-hooks** on each `ReadoutLayer`. On the training
forward pass:

1. When readout A is about to run, its hook collects incoming states and solves ridge
   for its targets.
2. Downstream layers see outputs from **already-fitted** readouts.

No manual "collect states then fit" loop — order follows the symbolic graph.

```python
from resdag.training import ESNTrainer

trainer = ESNTrainer(model)
trainer.fit(
    warmup_inputs=(warmup,),
    train_inputs=(train,),
    targets={"output": target},  # keys = readout names
)
```

### Contracts

- `warmup_inputs` and `train_inputs` must have the **same arity** (feedback first, then drivers).
- `train_inputs` and each target tensor share the same `(batch, time, features)` length.
- Target keys must match `CGReadoutLayer(..., name=...)`.

## Why hooks instead of a manual script?

Multi-readout graphs (e.g. position + velocity heads) need fitting in dependency order.
Hooks mirror `forward()` order automatically and stay correct when you add transforms or
concatenation layers.

## See also

- [Readouts](readouts.md)
- [Forecasting](forecasting.md)
- [`ESNTrainer` API](../reference/training.md)
- [Multi-readout guide](../guides/multi-readout.md)
