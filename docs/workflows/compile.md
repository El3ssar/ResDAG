---
description: How to torch.compile a reservoir without unrolling the time loop — compile the step, not the forward — plus the scan path and its caveats.
---

<span class="nb-kicker">Work · Compile & accelerate</span>

# Compile the step, not the forward

The reflex a PyTorch user has is `torch.compile(model)`. For a reservoir
that is the **wrong** thing to compile, and it is worth understanding why
before reaching for the tools that do the right thing.

## Why `torch.compile(reservoir)` backfires

A reservoir is a sequential recurrence: the state at step `t` depends on
the state at step `t-1`, so the forward pass is a Python loop over
timesteps. When `torch.compile` traces that forward it **unrolls** the
loop — it bakes one graph node per timestep into a single giant graph.
Two consequences follow:

- **First-compile cost scales with sequence length.** Tracing and
  optimizing `T` nodes takes time proportional to `T` — tens of seconds
  at `T = 500`, and longer still as the sequence grows. Worse, a new
  sequence length retriggers the whole compile.
- **The compiled call is often *slower* than eager.** The per-step ESN
  update is launch-bound, not FLOP-bound: each timestep is a couple of
  small matmuls. Unrolling does not remove the launches; it just moves
  them into one enormous graph that the compiler struggles to schedule
  better than the interpreter already does.

So the obvious move buys a long compile and a slower model. The fix is to
compile the *unit of repetition* — the single-step kernel — not the loop
that repeats it.

## The right tool: `compile_reservoirs()`

[`ESNModel.compile_reservoirs()`][resdag.core.ESNModel.compile_reservoirs]
wraps each reservoir's **single-step kernel** (`cell.step`) in
`torch.compile`, leaving the Python time loop to drive it:

```python
import resdag as rd

model = rd.ott_esn(reservoir_size=2000, feedback_size=3, output_size=3).to("cuda")
model.compile_reservoirs()                 # mode="reduce-overhead" by default

preds = model.forecast(warmup.cuda(), horizon=2000)   # first call compiles, then fast
```

The compiled graph is now a *single timestep*. It is traced once, regardless
of sequence length, and replayed every step — so the launch overhead that
dominates the per-step update amortizes across the whole sequence without any
unroll. The default `mode="reduce-overhead"` uses CUDA graphs to collapse the
launches; pass your own kwargs to override:

```python
model.compile_reservoirs(mode="default", fullgraph=True)   # kwargs forwarded to torch.compile
```

`compile_reservoirs()` returns the model, so it chains:

```python
preds = rd.ott_esn(2000, 3, 3).compile_reservoirs().forecast(warmup, horizon=2000)
```

!!! note "Where it pays off"
    The per-step update is launch-bound, so the win is a **GPU** phenomenon —
    compiling `cell.step` amortizes kernel launches that the CPU never pays in
    the same way. On CPU the eager loop is already lean and the compiled step
    can be at parity or slower; measure your own configuration with
    `benchmarks/compile_scan_reservoir.py`.

Compilation is lazy: `torch.compile` traces on the first call with a given input
signature, so the first forward / forecast after `compile_reservoirs()` pays the
one-time compile and subsequent calls are fast. Because it rebinds an instance
attribute, a compiled model is not picklable while compiled — compile *after*
any `save_full` / `deepcopy`, or on the restored model.

## The scan path: one graph region for the whole sequence

The complementary option lowers the loop itself.
`ESNLayer(compile_mode="scan")` expresses the recurrence as a single
`combine_fn` and runs it through
[`torch._higher_order_ops.scan`][torch._higher_order_ops.scan], so the entire
sequence is **one** graph region instead of `T` unrolled nodes:

```python
reservoir = rd.ESNLayer(2000, feedback_size=3, spectral_radius=0.9, compile_mode="scan")
states = reservoir(feedback)        # scan-driven, numerically identical to the loop
```

`compile_mode` takes `"loop"` / `"eager"` (the default Python loop) or `"scan"`.
The scan path is value-identical to the loop — same `cell.step` kernel — and
falls back to the Python loop automatically when the installed torch is too old
(`< 2.10`) or the scan op is unavailable, so it is always safe to request.

!!! warning "Scan is inference-only — autograd caveat"
    `torch._higher_order_ops.scan` is a **prototype** op. In current torch it
    **does not support autograd** and is known to miscompile under
    gradient-clamping
    ([pytorch#153437](https://github.com/pytorch/pytorch/issues/153437)). Use
    `compile_mode="scan"` for **inference / forward-only throughput**. For
    training *through time* (backpropagation through the reservoir, e.g. a
    `trainable=True` reservoir under SGD), keep the default loop, whose autograd
    is well-tested. Algebraic readout training (`ESNTrainer.fit`) is unaffected
    — it never backprops through the reservoir.

## Which one do I use?

| Goal | Use |
|---|---|
| Speed up the autoregressive `forecast` loop on GPU | `model.compile_reservoirs()` |
| One compiled graph region for a forward pass over a fixed-length sequence | `ESNLayer(compile_mode="scan")` |
| Training through the reservoir with autograd | the default loop — neither path |
| A quick `torch.compile(model)` | **avoid** — it unrolls the loop |

Both tools wrap **`cell.step`** (the per-step kernel), never the whole-sequence
`forward`. That single rule is the takeaway: the loop is the thing you do *not*
want the compiler to see all at once.

## See also

- [Work · Scale & deploy](deploy.md) — GPU regimes and persistence
- [Work · Forecast](forecast.md) — `forecast(compile=True)` for the autoregressive engine
- [Reference · Core](../reference/core.md) — `compile_reservoirs` in full
- [Theory · Reservoir dynamics](../theory/dynamics.md) — the recurrence being compiled
