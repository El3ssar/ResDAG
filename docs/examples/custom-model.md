# Custom architecture — parallel-timescale reservoirs

A worked example of building a non-trivial architecture by hand with
the [functional API](../guides/functional-api.md): **two reservoirs
running in parallel with different time constants**, their outputs
concatenated before a single readout. The fast reservoir captures
high-frequency content; the slow reservoir captures long-range
correlations.

This is the pattern that powers hierarchical-reservoir work in the
literature (Gallicchio et al., 2017) but it takes ~10 lines of ResDAG.

## Architecture

<figure markdown>
  ![Parallel reservoirs architecture](../assets/figures/arch_parallel_reservoirs.svg){ width="720" }
  <figcaption>Two <code>ESNLayer</code>s with different
  <code>spectral_radius</code> and <code>leak_rate</code> read the same
  symbolic input. Their outputs concatenate; a single
  <code>CGReadoutLayer</code> regresses against the target.</figcaption>
</figure>

## Build it

```python
import pytorch_symbolic as ps
from resdag import ESNModel, reservoir_input
from resdag.layers import ESNLayer, Concatenate, CGReadoutLayer

def parallel_reservoir_esn(
    feedback_size: int,
    output_size: int,
    fast_size: int = 200,
    slow_size: int = 200,
    fast_sr: float = 0.6,
    slow_sr: float = 0.95,
    fast_leak: float = 1.0,
    slow_leak: float = 0.3,
    readout_alpha: float = 1e-7,
    readout_name: str = "output",
) -> ESNModel:
    """Two parallel reservoirs with different time constants."""
    inp = reservoir_input(feedback_size)

    fast = ESNLayer(
        reservoir_size=fast_size, feedback_size=feedback_size,
        spectral_radius=fast_sr, leak_rate=fast_leak,
    )(inp)

    slow = ESNLayer(
        reservoir_size=slow_size, feedback_size=feedback_size,
        spectral_radius=slow_sr, leak_rate=slow_leak,
    )(inp)

    merged = Concatenate()(fast, slow)
    out = CGReadoutLayer(
        in_features=fast_size + slow_size,
        out_features=output_size,
        alpha=readout_alpha,
        name=readout_name,
    )(merged)

    return ESNModel(inp, out)
```

That's it — the function returns a fully-fledged `ESNModel` you can
treat like any built-in factory.

## Test it on Mackey–Glass

Mackey–Glass τ = 17 has both fast oscillations and slow modulation, so
the multi-timescale architecture has something to do.

```python
import torch
from resdag.training import ESNTrainer
from resdag.utils.data import prepare_esn_data

# Use the helper from the Mackey–Glass example.
data = mackey_glass(8_000)

torch.manual_seed(0)
warmup, train, target, f_warmup, val = prepare_esn_data(
    data,
    warmup_steps=500, train_steps=5_000, val_steps=1_500,
    discard_steps=500, normalize=True,
)

model = parallel_reservoir_esn(
    feedback_size=1, output_size=1,
    fast_size=300, slow_size=300,
)

ESNTrainer(model).fit(
    warmup_inputs=(warmup,),
    train_inputs=(train,),
    targets={"output": target},
)
model.reset_reservoirs()
pred = model.forecast(f_warmup, horizon=val.shape[1])
print("val MSE:", float(((pred - val) ** 2).mean()))
```

## Inspect what you built

```python
model.summary()
model.plot_model(show_shapes=True, show_trainable=True, save_path="parallel.svg")
```

`show_trainable=True` confirms only the readout has trainable
parameters; both reservoirs are 🔒 (frozen by default).

## Where to go from here

- Push the asymmetry harder: a *very* fast reservoir (`leak=1.0,
  sr=0.4`) and a *very* slow one (`leak=0.05, sr=0.99`).
- Stack three reservoirs instead of two, e.g. spanning three orders of
  magnitude in `leak_rate`.
- Add a `Power` or `SelectiveExponentiation` after each reservoir
  before the concatenation. The general pattern is in the
  [functional API guide](../guides/functional-api.md).
- Wrap the function in a [coupled ensemble](../guides/coupled-ensembles.md)
  to get N independently-initialised parallel-reservoir models with
  shared feedback.
