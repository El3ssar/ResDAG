---
description: Train over a torch DataLoader of windowed time series — the canonical SGD loop, the algebraic over-the-loader ridge fit with IncrementalRidgeReadoutLayer, and the ESNTrainer.fit_stream path for sequences too long to hold in memory.
---

<span class="nb-kicker">Work · Streaming</span>

# Streaming & DataLoaders

Because ResDAG models are ordinary `torch.nn.Module`s, they slot straight
into the PyTorch data stack. `TimeSeriesWindowDataset` slices a long
trajectory into fixed-length windows and `make_dataloader` wraps it in a
standard `DataLoader` whose batches are `(B, window_len, D)` — exactly what
`ESNLayer` and `ESNModel` consume. The same loader drives gradient descent
*and* an algebraic ridge fit that never holds the whole state matrix in
memory.

The runnable end-to-end recipe behind this page is
[`examples/14_streaming_dataloader.py`](https://github.com/El3ssar/ResDAG/blob/main/examples/14_streaming_dataloader.py).

## Windowing a trajectory

`make_dataloader` builds the dataset and loader in one call. Each window is an
`(input, target, washout)` triple: in the default forecasting mode the target
is the input shifted forward `horizon` steps, and `washout` marks the leading
steps to drop from the loss while the reservoir synchronizes. Windows never
straddle a trajectory boundary, so per-window state resets keep one trajectory
from leaking into the next.

<div class="nb-specimen" data-label="windowing.py" markdown>

```python
import torch
from resdag import make_dataloader, lorenz

series = lorenz(4000, seed=0, normalize="standard")[0]           # (4000, 3) — drop the batch axis: lorenz is (1, T, 3)

loader = make_dataloader(
    series,
    batch_size=16,
    window_len=250,                        # timesteps per window
    horizon=1,                             # one-step-ahead target (input shifted +1)
    stride=125,                            # 50% overlap between successive windows
    washout=50,                            # leading steps excluded from the loss
    shuffle=True,
)

x, y, washout = next(iter(loader))
print(x.shape, y.shape, washout)           # (16, 250, 3) (16, 250, 3) 50
```

</div>

These examples use the entire generated series for training, so their explicit
standardization uses training observations only. For held-out evaluation, split
the raw series first and reuse statistics fitted on the training prefix. The
common `Model.fit` workflow owns that scaling automatically.

`series` may be a single `(T, D)` tensor, a batched `(B, T, D)` tensor (each
slice becomes one trajectory), or a ragged list of `(T, D)` tensors of
differing length — windows are generated per trajectory in every case. For a
regression task, pass an aligned `targets=` tensor instead of relying on
`horizon`. Input `stats` normalize inputs and shifted forecast targets;
external regression targets keep their own units. Pass separate `target_stats`
and `target_norm_method` to normalize external targets. Need the underlying dataset for a custom loader? Build a
`TimeSeriesWindowDataset` directly and wrap it yourself.

## The SGD loop

The canonical loop trains a head on **frozen** reservoir features, minibatched
over the loader. Reset the reservoir at the top of each window so state does
not carry across independent windows; skip the `washout` steps when computing
the loss.

<div class="nb-specimen" data-label="streaming_sgd.py" markdown>

```python
import torch
import torch.nn as nn
from resdag import ESNLayer, make_dataloader, lorenz

series = lorenz(4000, seed=0, normalize="standard")[0]
loader = make_dataloader(series, batch_size=16, window_len=250,
                         horizon=1, stride=125, washout=50, shuffle=True)

reservoir = ESNLayer(200, feedback_size=3, spectral_radius=0.9, trainable=False)
head = nn.Linear(200, 3)
optimizer = torch.optim.Adam(head.parameters(), lr=5e-3)   # head only — reservoir is frozen
criterion = nn.MSELoss()

for epoch in range(6):
    for x, y, washout in loader:
        reservoir.reset_state()                            # each window is independent
        pred = head(reservoir(x))
        loss = criterion(pred[:, washout:], y[:, washout:])  # skip the washout transient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

</div>

`detach_state_between_calls` is on by default, so even without the per-window
`reset_state()` the stored state is severed from the autograd graph at every
forward-call boundary and consecutive `backward()` calls never raise "backward
through the graph a second time". To backpropagate through the recurrence as
well (full BPTT through each window), build the reservoir with
`trainable=True` and add its parameters to the optimizer — see
[Train · Path 3](train.md#path-3-full-bptt).

## Algebraic over the DataLoader

When the reservoir is fixed and you want the classic ridge readout — but the
full state matrix would not fit in memory — accumulate the fit incrementally.
`IncrementalRidgeReadoutLayer` keeps the ridge **sufficient statistics** (the Gram
matrix and cross term), so `partial_fit` per batch followed by a single
`finalize` is algebraically identical to one full-batch solve over the
concatenated data, to floating-point tolerance.

<div class="nb-specimen" data-label="streaming_ridge.py" markdown>

```python
import torch
from resdag import ESNLayer, make_dataloader, lorenz
from resdag.layers import IncrementalRidgeReadoutLayer

series = lorenz(4000, seed=0, normalize="standard")[0]
loader = make_dataloader(series, batch_size=16, window_len=250,
                         horizon=1, stride=125, washout=50)

reservoir = ESNLayer(200, feedback_size=3, spectral_radius=0.9, trainable=False)
readout = IncrementalRidgeReadoutLayer(200, 3, name="output", alpha=1e-4)

readout.reset_accumulators()
with torch.no_grad():
    for x, y, washout in loader:
        reservoir.reset_state()
        states = reservoir(x)
        readout.partial_fit(states[:, washout:], y[:, washout:])  # accumulate, hold no full matrix

print(f"accumulated {readout.n_seen} post-washout samples")
readout.finalize()                          # one ridge solve from the running statistics

with torch.no_grad():
    x, y, washout = next(iter(loader))
    reservoir.reset_state()
    pred = readout(reservoir(x))            # is_fitted is now True
```

</div>

`finalize` must be called once before inference — `forward` on an unfitted
`IncrementalRidgeReadoutLayer` raises with a clear message. Calling `partial_fit`
again after `finalize` keeps accumulating and re-marks the layer unfitted, so
`finalize` again to refresh; `reset_accumulators` starts a fresh fit.

## fit_stream — the trainer over a stream

For a full `ESNModel` whose readouts are all `IncrementalRidgeReadoutLayer`, the
trainer wraps the accumulate-then-finalize pattern in `fit_stream`. It warms
the reservoir once, then consumes chunks one at a time, accumulating each
readout's statistics and solving once at the end — no more than a single
chunk's states are ever in memory.

<div class="nb-specimen" data-label="fit_stream.py" markdown>

```python
import torch
import resdag as rd
from resdag import ESNLayer, ESNModel, reservoir_input, make_dataloader, lorenz
from resdag.layers import IncrementalRidgeReadoutLayer

series = lorenz(4000, seed=0, normalize="standard")                # keep (1, T, D): one continuous stream

inp = reservoir_input(3)
states = ESNLayer(200, feedback_size=3, spectral_radius=0.9)(inp)
out = IncrementalRidgeReadoutLayer(200, 3, name="output", alpha=1e-4)(states)
model = ESNModel(inp, out)

warmup = series[:, :300]                    # (1, 300, 3)

def chunk_stream():
    # Adjacent input chunks, with next-sample targets and fixed batch slots.
    for start in range(300, series.shape[1] - 1, 250):
        stop = min(start + 250, series.shape[1] - 1)
        yield (series[:, start:stop],), {"output": series[:, start+1:stop+1]}

rd.ESNTrainer(model).fit_stream(warmup_inputs=(warmup,), chunks=chunk_stream())
forecast = model.forecast(series[:, -300:], horizon=200)
```

</div>

!!! warning "Chunks must be contiguous and in order"
    `fit_stream` flows reservoir state from the end of one chunk into the start
    of the next, exactly as a single long forward pass would. Do **not**
    shuffle or overlap chunks feeding `fit_stream` — that desynchronizes the
    reservoir. Keep the same batch size and trajectory in each batch slot.
    Ordinary sliding-window batches represent independent windows and should
    use the per-window reset/`partial_fit` path above.
    (The shuffled loader earlier on this page is fine for the SGD and
    per-window `partial_fit` paths, which reset state per window.) Every readout
    in the model must be an `IncrementalRidgeReadoutLayer`; mixed readouts raise, and
    `fit` is the in-memory path for single-pass readouts. Readouts on separate
    branches may share the stream; stacked readouts require separate fitting
    passes because their input features depend on upstream fitted weights.

## When to use which path

| Path | Use when |
| --- | --- |
| **SGD loop** (frozen head) | The head is nonlinear or the loss is not least-squares; you want minibatched gradient training. |
| **SGD loop** (`trainable=True`) | You need full BPTT through the recurrence (rarely worth it — see [Train](train.md#path-3-full-bptt)). |
| **Per-window `partial_fit`** | A fixed reservoir, the classic ridge readout, and a state matrix too large to materialize, with manual control over the loop. |
| **`fit_stream`** | The same algebraic streaming fit for a full `ESNModel`, with warmup and the contiguous-chunk bookkeeping handled for you. |

## Next

- [Train](train.md) — the three in-memory training paths and `prepare_esn_data`
- [Forecast](forecast.md) — the autoregressive rollout after fitting
- [Scale & deploy](deploy.md) — GPU execution and embedding frozen reservoirs in larger pipelines
