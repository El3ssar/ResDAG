# Examples

End-to-end runnable scripts with embedded figures. Each example is a
single page that walks the data, the model, the training call, and the
forecast — concise, with every figure you'd want to see while
following along.

If you prefer the conceptual map, jump to the [User guide](../guides/index.md).

### Forecasting

- **[Sine forecasting](sine.md)** — the simplest possible example.
  `classic_esn` on a 6 000-step sine; the autoregressive prediction
  tracks the truth to MSE ≈ 6×10⁻¹¹.
- **[Lorenz-63](lorenz.md)** — canonical chaotic 3-D system. `ott_esn`
  tracks the attractor for ~5 Lyapunov times before drift.
- **[Mackey–Glass](mackey-glass.md)** — quasi-periodic chaos with
  τ = 17. Different attractor regime, different tuning.

### Architecture

- **[Custom architecture](custom-model.md)** — build a
  parallel-timescale reservoir model by hand with `pytorch_symbolic`.
- **[Custom initializer](custom-initializer.md)** — implement an
  orthogonal input/feedback initializer, register it, and use it like
  any built-in.

### Integration

- **[Coupled ensemble](ensemble.md)** — five `ott_esn` sub-models with
  shared feedback aggregation.
- **[Pipeline integration](pipeline-integration.md)** — embed a frozen
  reservoir as a feature extractor inside a larger PyTorch pipeline
  with an SGD-trained head.

All scripts here are also available in the repo under
[`examples/`](https://github.com/El3ssar/resdag/tree/main/examples) —
the same code, plus a few extras (model visualisation, save/load,
HPO).
