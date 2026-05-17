# Training paths in resdag

`resdag` supports two complementary ways of fitting an ESN model. They share
the same `ESNModel` class — what differs is which parameters are frozen and
which optimiser updates them.

| Path                         | Reservoir | Readout         | Optimiser            | Typical use                        |
| ---------------------------- | --------- | --------------- | -------------------- | ---------------------------------- |
| 1. Classical ESN (algebraic) | frozen    | `CGReadoutLayer` (frozen) | `ESNTrainer.fit`   | Standard ESN forecasting          |
| 2. Mixed (SGD readout)       | frozen    | `nn.Linear` / MLP (trainable) | `torch.optim`     | ESN-as-feature-extractor pipelines |
| 3. Fully trainable (BPTT)    | trainable | trainable       | `torch.optim`        | When the random reservoir is suboptimal |

---

## 1. Classical algebraic readout

The reservoir is left at its random initialisation; only the readout weights
are solved in closed form by ridge regression (conjugate gradient inside
`CGReadoutLayer`). This is the path most ESN papers describe.

```python
import resdag as rd
from resdag.composition import reservoir_input

inp = reservoir_input(feature_size=3)
res = rd.ESNLayer(reservoir_size=500, feedback_size=3,
                  trainable=False)(inp)
out = rd.CGReadoutLayer(in_features=500, out_features=3,
                        name="output")(res)
model = rd.ESNModel(inp, out)

rd.ESNTrainer(model).fit(
    warmup_inputs=(warmup,),
    train_inputs=(train,),
    targets={"output": targets},
)
predictions = model.forecast(forecast_warmup, horizon=100)
```

**Why pick this**
- Fast and deterministic — no learning-rate, no convergence drama.
- The reservoir's *Echo State Property* is the entire point: keep it frozen.
- Works without a GPU (the CG solver is BLAS-bound).

**Constraints**
- The output dimension is fixed by the targets you provide.
- The readout topology is per-timestep linear (or whatever subclass of
  `ReadoutLayer` you write — override `_fit_impl` to add your own algebraic
  solver, see Phase 3.2 of the 0.4.0 release notes).

---

## 2. Mixed: frozen reservoir, SGD readout

Replace the algebraic readout with any `nn.Module` (e.g. an MLP, a
classification head, a probabilistic decoder) and train *only the head* with
a standard PyTorch optimiser. The reservoir is frozen, so its weights never
receive gradients.

```python
import torch.nn as nn
import torch

inp = reservoir_input(feature_size=3)
res = rd.ESNLayer(reservoir_size=500, feedback_size=3,
                  trainable=False)(inp)
feature_model = rd.ESNModel(inp, res)  # outputs the reservoir states

head = nn.Sequential(
    nn.Linear(500, 64),
    nn.Tanh(),
    nn.Linear(64, 3),
)

optim = torch.optim.Adam(head.parameters(), lr=1e-3)  # head params only
criterion = nn.MSELoss()

for batch_x, batch_y in loader:
    feature_model.reset_reservoirs()
    feats = feature_model(batch_x)        # (B, T, R) — no grads through res
    pred = head(feats)                    # (B, T, 3)
    loss = criterion(pred, batch_y)
    optim.zero_grad(); loss.backward(); optim.step()
```

**Why pick this**
- You want a non-linear / non-MSE head (classification, mixture density,
  embedding distillation…).
- You're plugging an ESN feature extractor into an existing PyTorch pipeline.
- You want autograd-friendly training without the cost of BPTT through a
  large reservoir.

**Tip**
- Pass `feature_model` itself as a layer inside `nn.Sequential` — see
  `examples/12_pipeline_integration.py` scenario 3.

---

## 3. Fully trainable (BPTT)

Set `trainable=True` on the reservoir; PyTorch then tracks gradients through
the recurrent update at every timestep. Use this when the random reservoir
initialisation is the bottleneck for a particular task.

```python
inp = reservoir_input(feature_size=3)
res = rd.ESNLayer(reservoir_size=200, feedback_size=3,
                  trainable=True)(inp)
head = nn.Linear(200, 3)(res)
model = rd.ESNModel(inp, head)

optim = torch.optim.Adam(model.parameters(), lr=5e-3)
for step in range(epochs):
    model.reset_reservoirs()
    pred = model(x)
    loss = criterion(pred, y)
    optim.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optim.step()
```

**Why pick this**
- The frozen random reservoir doesn't capture the relevant features.
- You want to combine ESN structure with a downstream task that benefits
  from end-to-end gradients (e.g. a contrastive objective).

**Caveats**
- BPTT through a large reservoir is expensive — memory grows linearly with
  sequence length.
- Gradient clipping (`clip_grad_norm_`) is strongly recommended.
- Reservoir stability (the Echo State Property) is no longer guaranteed
  during training; spectral-radius drift can destabilise long forecasts.

---

## Sanity checks

When in doubt, inspect the partitioning of trainable parameters:

```python
trainable = [n for n, p in model.named_parameters() if p.requires_grad]
print(trainable)
```

- **Path 1**: empty list.
- **Path 2**: only your head's parameters.
- **Path 3**: every reservoir weight + every head weight.

If the partition looks wrong, the wrong path is being trained — re-check the
`trainable=` flags on `ESNLayer` and `CGReadoutLayer` / `ReadoutLayer`.

---

## See also

- `examples/09_training.py` — algebraic training walkthrough (path 1).
- `examples/12_pipeline_integration.py` — paths 2 and 3 end-to-end.
- `tests/test_training/test_sgd_path.py` — assertions on the gradient flow.
