# Mental model

## Components

- **Reservoir layer** — maps `(batch, time, features)` to states. Internal recurrent
  weights are set at initialization and usually not updated by gradient descent.
- **Readout layer** — maps states to outputs. Fitted from collected states and targets
  (implementation may use conjugate-gradient ridge, closed-form solvers, or other
  algebraic methods depending on the class).
- **`ESNModel`** — wraps a `pytorch_symbolic` graph, exposes `warmup`, `forecast`,
  state save/load, and standard `forward`.

## Training

`ESNTrainer.fit` expects:

1. **Warmup** — teacher-forced inputs to align reservoir state with the data segment
   before training.
2. **Train** — teacher-forced inputs on the fitting segment; readout hooks solve for
   weights against `targets` (dict keyed by readout `name`).

No epoch loop over the reservoir is required for the default algebraic readouts.

## Forecasting

`model.forecast(f_warmup, horizon=…)`:

1. Warmup (optional reset) on `f_warmup` and any drivers.
2. Autoregressive steps: the model output at step $t$ becomes feedback at $t+1$.

The feedback channel dimension must match the first output head used for feedback.

## Data splits (`prepare_esn_data`)

Timeline after optional `discard_steps`:

```text
[ warmup | train | val ]
```

- `target` — `train` shifted by one timestep (one-step-ahead supervision).
- **`f_warmup`** — `train[:, -warmup_steps:, :]`, i.e. the final warmup-length suffix
  of **train**. The reservoir has already been driven on the preceding train segment
  during fitting; `f_warmup` is the drive immediately before `val`, which is the
  held-out segment used for scoring forecasts.

Do not take `f_warmup` from an arbitrary later index in the series; use
`prepare_esn_data` or reproduce its definition exactly.

## Next

[Your first ESN](your-first-esn.md)
