<span class="rd-eyebrow">Under the hood</span>

# Timing & alignment

Every off-by-one bug in reservoir computing is a disagreement about which
timestep an input, a state, and a target belong to. This page fixes ResDAG's
conventions once, end to end.

## The one-step-ahead contract

The model is trained so that its output at step $t$ estimates the feedback at
step $t+1$:

$$
\hat{y}_t = G\big(f_t,\, d_t\big) \approx f_{t+1}
$$

where $f$ is the feedback series and $d$ the (optional) driver series.
`prepare_esn_data` encodes this: `target` is `train` shifted left by one
step. If you build targets yourself, keep the same shift — *everything*
below assumes it.

## Training (`ESNTrainer.fit`)

```text
fit(warmup_inputs, train_inputs, targets)
```

1. **Reset** — all reservoir states cleared.
2. **Warmup** — one teacher-forced pass over `warmup_inputs`. Outputs are
   discarded; only the reservoir states matter. This washes out the
   arbitrary zero initial state (echo state property).
3. **Fit pass** — one forward pass over `train_inputs` with a pre-hook on
   every readout. The hook fires at the moment the readout would execute,
   receives the exact tensor the readout is about to see (after any
   transforms/concatenations upstream in the DAG), solves the
   [ridge problem](readout-fitting.md) against `targets[name]`, and *then*
   lets the now-fitted readout run. Downstream layers therefore consume
   already-fitted outputs — multi-readout DAGs fit in topological order with
   no extra passes.

States at row $t$ of the fit pass were produced from inputs at row $t$, and
are regressed onto `target` row $t$ = feedback row $t+1$: the readout learns
exactly the one-step-ahead contract.

## Forecasting (`ESNModel.forecast`)

Let the warmup window cover times $1 \dots T$. `forecast(warmup_inputs,
horizon=H)` returns `predictions` with `predictions[:, i]` estimating
$f_{T+1+i}$, for $i = 0 \dots H-1$:

1. **Warmup** — teacher-forced pass over the warmup window (reset first by
   default). Its *last* output, produced from $(f_T, d_T)$, is the estimate
   of $f_{T+1}$ → stored as `predictions[:, 0]`.
2. **Autoregression** — for $t = 1 \dots H-1$: feed the previous prediction
   back as feedback, together with the driver for that same step, and store
   the output as `predictions[:, t]`.

Worked indices for `horizon=4`, warmup of length $T$:

| step | feedback in | driver in | output stored as | estimates |
| --- | --- | --- | --- | --- |
| warmup (last) | $f_T$ | $d_T$ | `predictions[:, 0]` | $f_{T+1}$ |
| $t=1$ | `predictions[:, 0]` | `forecast_inputs[:, 0]` $= d_{T+1}$ | `predictions[:, 1]` | $f_{T+2}$ |
| $t=2$ | `predictions[:, 1]` | `forecast_inputs[:, 1]` $= d_{T+2}$ | `predictions[:, 2]` | $f_{T+3}$ |
| $t=3$ | `predictions[:, 2]` | `forecast_inputs[:, 2]` $= d_{T+3}$ | `predictions[:, 3]` | $f_{T+4}$ |

## Driver alignment

From the table: `forecast_inputs` is the driver series **continuing exactly
where the warmup drivers ended** — no overlap, no gap. If your full driver
series is `d` and warmup used `d[:, :T]`, pass `d[:, T:T+horizon-1]` (or
`d[:, T:T+horizon]`; both are accepted and the last step of the longer form
is unused, so you can slice drivers over the same window as your validation
targets).

Only $H-1$ driver values are ever consumed: prediction 0 was already
produced during warmup from $d_T$, and predicting $f_{T+H}$ needs drivers
only up to $d_{T+H-1}$.

!!! warning "Changed in 0.5"
    Before 0.5 the loop consumed `forecast_inputs[:, t]` at step $t$: the
    first driver value was silently skipped and every remaining step saw the
    driver one step *ahead* of its training-time pairing. If you forecast
    with drivers on ≤ 0.4, expect (slightly improved) different results.
    The same fix applies to `CoupledEnsembleESNModel.forecast`.

`predictions[:, 0]` comes from teacher-forced data, so it is the easiest
step; errors compound from index 1 onward. With `return_warmup=True` the
warmup outputs (estimates of $f_2 \dots f_{T+1}$) are prepended.

## Statefulness across calls

Reservoir layers keep `state` between `forward` calls — that is what makes
`warmup → forecast` work. Three consequences:

- **Independent sequences need a reset.** Call `model.reset_reservoirs()`
  (or `warmup(..., reset=True)`, the default) before unrelated data.
- **State re-initializes silently** when the batch size, device, or dtype of
  the incoming feedback changes — continuity is only preserved across calls
  with matching shapes.
- **Gradients do not cross call boundaries.** The stored state is detached
  at the end of each `forward` (truncated BPTT), so SGD over consecutive
  batches works out of the box without `retain_graph` errors. Gradients
  still flow through all timesteps *within* one call. Set
  `layer.detach_state_between_calls = False` only if you intend to backprop
  through state carried across calls and manage the graphs yourself.
