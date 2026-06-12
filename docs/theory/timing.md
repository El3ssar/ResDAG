---
description: The one-step-ahead contract, the target shift, the forecast index map with a worked table, driver alignment, and what state persists across calls.
---

<span class="nb-kicker">Theory · Timing</span>

# Timing conventions

Off-by-one errors in autoregressive forecasting are silent — the model
still produces plausible trajectories, just trained on the wrong pairing.
This page fixes every index in the pipeline: which target row a state row
is regressed onto, what `predictions[:, i]` estimates, and which driver
value each step consumes.

## The one-step-ahead contract

A trained model is a one-step predictor. Writing $f_t$ for the feedback
signal and $d_t$ for an optional driver at sample $t$:

$$
\hat y_t = G(f_t, d_t) \approx f_{t+1}
$$

Everything else on this page is bookkeeping that preserves this contract
through training and forecasting.

**The target shift.** `prepare_esn_data` cuts one timeline so the shift
is built into the splits. With `warmup_steps` $= W$ and `train_steps`
$= L$ (after `discard_steps`):

```python
warmup = data[:, :W]            # synchronize state
train  = data[:, W     : W+L]   # model input at step t
target = data[:, W + 1 : W+L+1] # = train shifted forward one step
```

`target[:, t]` is `train[:, t+1]`'s value — the split needs one sample
beyond the training window, which is why `prepare_esn_data` rejects
`W + L >= len(data)`. The forecast warmup is the tail of `train`
(`train[:, -W:]`) and `val` starts at `data[:, W+L]`, immediately after.

**Training alignment.** `ESNTrainer.fit` runs the warmup, then one
forward pass over `train`. The reservoir emits state row $t$ from input
row $t$; each readout's `fit` hook regresses state row $t$ onto target
row $t$. Row for row, no internal re-shifting — the shift lives entirely in
the data preparation above.

---

## The forecast index map

`forecast(warmup_inputs, horizon=H)` runs a teacher-forced warmup, then
closes the loop. Let $T$ index the **last warmup sample** $(f_T, d_T)$.
The invariant:

$$
\texttt{predictions[:, i]} \;\approx\; f_{T+1+i}
$$

Prediction 0 is not produced by the loop at all — it is the model's
output for the last warmup pair, stored before the autoregressive phase
begins. Loop step $t \ge 1$ feeds back prediction $t-1$ and consumes
driver `forecast_inputs[:, t-1]`. Worked out for `horizon=4`:

| Step | Feedback in | Driver in | Stored as | Estimates |
| --- | --- | --- | --- | --- |
| warmup (last) | $f_T$ | $d_T$ | `predictions[:, 0]` | $f_{T+1}$ |
| $t = 1$ | `predictions[:, 0]` | `forecast_inputs[:, 0]` $= d_{T+1}$ | `predictions[:, 1]` | $f_{T+2}$ |
| $t = 2$ | `predictions[:, 1]` | `forecast_inputs[:, 1]` $= d_{T+2}$ | `predictions[:, 2]` | $f_{T+3}$ |
| $t = 3$ | `predictions[:, 2]` | `forecast_inputs[:, 2]` $= d_{T+3}$ | `predictions[:, 3]` | $f_{T+4}$ |

With the `prepare_esn_data` splits, the forecast warmup ends at data
index $W{+}L{-}1$, so $T = W{+}L{-}1$ and `predictions[:, i]` estimates
`data[:, W+L+i]` $=$ `val[:, i]` — the forecast aligns one-to-one with
the validation split, so errors can be computed directly as
`predictions - val` with no re-indexing.

**Driver alignment.** `forecast_inputs` is the driver series for the
forecast window, continuing *exactly where the warmup drivers ended*:
`forecast_inputs[:, t]` is the driver at the $(t{+}1)$-th step after the
warmup window. Because prediction 0 already consumed $d_T$ during warmup,
the loop only reads indices $0$ through $H{-}2$ — **$H-1$ values**. Both
lengths $H-1$ and $H$ are accepted; with $H$ the last value is unused,
which lets you slice drivers over the same window as your validation
targets.

!!! warning "Changed in 0.5"
    Pre-0.5, loop step $t$ consumed `forecast_inputs[:, t]` instead of
    `[:, t-1]`: entry 0 was silently skipped and every prediction was
    paired with the driver one step *ahead* of the training convention.
    Since 0.5 the contract above holds, and the required driver length
    changed from `horizon` to `horizon - 1` (or `horizon`, last value
    unused). If you offset your driver slices to compensate, remove the
    offset.

---

## Statefulness across calls

Reservoir layers carry their state between `forward` calls. This
persistence is what makes the warmup-then-forecast split work; it is also
a common source of errors.

- **Reset behavior.** `warmup()` and `forecast()` reset all
  reservoirs by default (`reset=True`); `ESNTrainer.fit` always resets
  before its warmup. A bare `model(x)` or `layer(x)` never resets —
  consecutive calls continue the trajectory unless you call
  `reset_state()` / `reset_reservoirs()`.
- **Silent re-initialization.** If an incoming batch's size, device, or
  dtype differs from the stored state's, the state is silently replaced
  with zeros of the new configuration (`_maybe_init_state`). Convenient
  for switching from training batches to a single forecast trajectory;
  surprising if you expected state to survive a `.to("cuda")`.
- **Detached state at call boundaries.** At the end of each `forward`,
  the *stored* state is detached from the autograd graph. Gradients flow
  through the returned states within a call, but never across calls —
  truncated BPTT at call boundaries, which is what lets SGD over
  consecutive batches work without "backward through the graph a second
  time" errors. If you intentionally backprop through state carried
  across calls, set `layer.detach_state_between_calls = False` and manage
  the retained graphs yourself.

## Next

[**Design of the library**](design.md) — the architecture decisions
behind the code that enforces these conventions.
