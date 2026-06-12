---
description: Save, load, and checkpoint ESN models — weights, reservoir states, and metadata, with the one rule that makes it all work.
---

<span class="rd-eyebrow">Cookbook</span>

# Save, load, checkpoint

By the end of this page you'll have a fitted model on disk and back in
memory, forecasting immediately — no refit. The one rule: **the file holds
weights, not architecture.** You rebuild the graph; the file restores the
numbers.

## The round trip

Keep a build function. It is the architecture's source of truth — the
same factory with the same arguments must run on both sides of the save.

<div class="rd-window" data-title="round_trip.py" markdown>

```python
from resdag import ESNTrainer
from resdag.models import ott_esn

def build_model():
    return ott_esn(reservoir_size=500, feedback_size=3, output_size=3)

# Train and save
model = build_model()
ESNTrainer(model).fit(
    warmup_inputs=(warmup,),
    train_inputs=(train,),
    targets={"output": target},
)
model.save("lorenz.pt")                     # weights only

# Later, in another process
model = build_model()                       # same factory, same args
model.load("lorenz.pt")
prediction = model.forecast(f_warmup, horizon=1000)   # no refit needed
```

</div>

This works because a fitted `CGReadoutLayer` stores its solution in plain
`nn.Linear` parameters (`weight`, `bias`) — the algebraic solve writes
into the `state_dict`, so it round-trips like any PyTorch module. Frozen
reservoir weights ride along the same way.

!!! warning "Architecture is not serialized"
    `save()` stores `state_dict()` plus metadata — layer sizes, topology
    names, and wiring are not in the file. Load into a different
    architecture and `load_state_dict` fails on mismatched keys (loudly,
    with `strict=True` by default). Version-control the build function
    alongside the checkpoint.

## Checkpoints: states and metadata

`save()` accepts arbitrary keyword metadata, and `include_states=True`
snapshots every reservoir's current state:

```python
model.save("checkpoint.pt", include_states=True, epoch=10, val_mse=0.042)

model = build_model()
model.load("checkpoint.pt", load_states=True)   # warns if no states in file
```

The class-method spelling does the same in one line:

```python
from resdag import ESNModel

model = ESNModel.load_from_file("checkpoint.pt", model=build_model(), load_states=True)
```

## Resuming long sequences mid-stream

Reservoirs are stateful, so you can pause a long teacher-forced run and
continue later without replaying the warmup:

```python
states = model.get_reservoir_states()       # {layer_name: state clone}
# ... save them, restart the process, rebuild + load the model ...
model.set_reservoir_states(states)          # strict=True: keys must match exactly
model.warmup(next_chunk, reset=False)       # continue, don't wipe the state
```

`set_reservoir_states` validates that the dict covers every reservoir in
the model — a missing or extra key raises `KeyError` instead of silently
desynchronizing one layer.

!!! tip "States are batch-shaped"
    A saved state has shape `(batch, reservoir_size)` (3-D delay buffer
    for NG-RC). Restore it only into a run with the same batch size — a
    mismatched forward pass re-initializes the state to zeros.

## Related

- [PyTorch pipelines](pipelines.md) — checkpoints for SGD-trained heads use the same `state_dict` machinery.
- [Training](../learn/training.md) — what the fit actually writes into the readout.
- [GPU & performance](gpu.md) — device placement when loading on different hardware.
