# Forecast targets and feedback connections

A forecast starts with the **first unseen sample** after your history. Two
settings describe how a custom model gets there:

- **Prediction lead** tells a head what to learn. Lead 2 pairs an input at time
  `t` with the observation at `t + 2`.
- **Feedback delay** tells a connection when to reuse an output. Delay 2 feeds
  an output back into its input two model steps after that output was produced.

Both default to 1. For normal self-feedback, use the same lead and delay.
Different values let you build deliberately lagged connections between parts of
a model.

## Start with the default

```python
import torch
import resdag as rd

x = rd.reservoir_input(1, dtype=torch.float64)
y = rd.RidgeReadoutLayer(1, 1, alpha=0.0).double()(x)
model = rd.Model(
    inputs={"state": x},
    outputs={"next_state": y},
    washout=0,
)
observed = torch.arange(8, dtype=torch.float64)[:, None]
model.fit(observed)
prediction = model.forecast(horizon=4)  # approximately 8, 9, 10, 11
```

`fit(observed)` prepares the shifted input/target pairs for you. The default
feedback connection runs from the first output to the first input. Its delay
automatically matches that output's prediction lead.

## Predict two steps ahead

```python
x = rd.reservoir_input(1, dtype=torch.float64)
y = rd.RidgeReadoutLayer(1, 1, alpha=0.0).double()(x)
model = rd.Model(
    inputs={"state": x},
    outputs={"next_state": y},
    prediction_lead=2,
    feedback=[rd.Feedback(output="next_state", input="state", delay=2)],
    washout=0,
)
observed = torch.arange(5, dtype=torch.float64)[:, None]
model.fit(observed)
prediction = model.forecast(horizon=4)  # approximately 5, 6, 7, 8
```

The last two warmup outputs already estimate samples 5 and 6. Forecasting keeps
both pending predictions, then generates 7 and 8. It does not skip sample 5.
You can omit `feedback` here: the automatic connection also uses delay 2.

For a head trained to add 1, choosing lead 1 and delay 2 instead produces
`5, 5, 6, 6` after the history `0, 1, 2, 3, 4`. That repeated pattern is the
declared lag, rather than an indexing correction.

## Route by public names

Each `Feedback` uses the names from the model's `inputs` and `outputs` mappings.
Internal layer names and mapping insertion order do not select connections.

```python
connections = [
    rd.Feedback(output="slow_prediction", input="slow_state", delay=2),
    rd.Feedback(output="fast_prediction", input="fast_state", delay=1),
]
leads = {"slow_prediction": 2, "fast_prediction": 1}
```

Pass these as `feedback=connections, prediction_lead=leads` when constructing
your graph. A lead mapping can override selected heads; other heads use lead 1.
An output can feed several inputs, but each input accepts at most one feedback
connection. Connected widths must match. Add an explicit graph projection when
you need a width conversion.

Inputs without a connection are **external drivers**. Supply them by name in
`forecast(..., future_inputs={"forcing": future_forcing})`. Their first sample
belongs immediately after the observed history. For ordinary next-step heads,
a horizon of 100 needs 99 driver samples: warmup already provides the first
prediction. More generally, the count is
`max(0, horizon - min(model.prediction_leads.values()))`.

Use `feedback=[]` when every input is externally driven. History must contain
at least the largest lead and largest connection delay so every pending value
can be initialized. Names, widths, history and driver lengths are checked before
forecasting changes reservoir state or runs model hooks.

## Scaling and explicit targets

Feedback travels through original units: the source head's inverse transform
is followed by the destination input's transform. Independently fitted target
statistics therefore work without manual packing or scaling. All tensor
operations stay on the model device; only an explicitly selected trajectory
result converts to NumPy.

With `fit(inputs, targets=...)`, target arrays are already aligned with their
input rows. `prediction_lead` records the physical time that each target row
represents; it does not shift explicit arrays a second time. For example, a
lead-2 target supplied at input row `t` must already contain the desired value
at physical time `t + 2`.

All returned heads share the same future physical timeline. With
`return_warmup=True`, the shared historical prefix starts at the largest lead:
for history length `W` and largest lead `L`, the result has `W - L + horizon`
samples per head. Trajectory timestamps follow this same alignment.

## Custom temporal modules

Fast forecasting uses one input frame per step. Built-in pointwise transforms
and explicit reservoir layers support this. A custom module can declare
`supports_framewise = True` when its evaluation is independent across time,
preserves batch/time dimensions, and has no hidden temporal state.

For a pure causal sequence transform such as a prefix sum, choose
`forecast(..., execution="history")`. This recomputes the entire growing
observed/generated input prefix from the configured reservoir reset on each
step. It retains every prefix sample and costs more time and memory. Such
graphs also fit their terminal heads from complete-prefix features before
discarding washout rows, and retain their complete fitting history for cached
forecasting. Ordinary fast models retain only their compact synchronization
tail. History execution currently requires `compile=False`.

Custom temporal modules must be pure across evaluation calls. Expose mutable
streaming state through a `BaseReservoirLayer` adapter; replay cannot discover
or reset arbitrary buffers hidden in a custom module.

## State and checkpoints

Each `forecast` call starts a fresh rollout from its history, rebuilding delayed
queues. `get_state()` and `set_state()` describe reservoir state for ordinary
forward calls; those tensors do not include pending feedback or its clock and
cannot resume a delayed forecast by themselves.

`state_dict()` includes prediction leads, connection names/delays, fitted
scalers and cached history. Loading restores those settings after validating
the graph's named ports. Complete model artifacts are covered by the
[model workflow guide](model-workflow.md).
